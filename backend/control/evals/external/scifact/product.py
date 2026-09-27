"""SciFact *through the ToxAgent product* — a `published-data-transfer` study (W7-04).

`run.py` measures a model used as a stand-alone verifier. This measures the
product: the corpus is served through its own research provider, ToxAgent writes
its own queries, decides what to read, and its accepted answer's
``evidence_relations`` are read back as abstract-level SUPPORT / CONTRADICT.

Why the label is `published-data-transfer` and never `external-native`
(RETHINK §5.2):

* the task format changes — a SciFact claim becomes a question a researcher
  would ask, because the product answers questions and does not classify;
* retrieval is the product's own BM25 over the pinned corpus, not the
  benchmark's oracle or TF-IDF setting;
* the product selects no rationale sentences, so only ``abstract_label_only``
  is defined. The three other official metrics are not computed at all rather
  than reported as zero.

Comparisons that are fair: the same claim ids run through `run.py`'s judges
(`--compare-with RUN_ID` records which run to compare with in the manifest).
Comparisons that are not: any published full-split number.

Preparing the corpus (the abstracts are ODC-By and never enter this repository)::

    python -m evals.external.scifact.product --write-corpus /srv/scifact-corpus.jsonl
    # prints the sha256; start the control plane with
    #   TOXAGENT_RESEARCH_PROVIDER=corpus
    #   TOXAGENT_RESEARCH_CORPUS_PATH=/srv/scifact-corpus.jsonl
    #   TOXAGENT_RESEARCH_CORPUS_SHA256=<printed hash>
    #   TOXAGENT_FLAG_ANSWER_DRAFT_V2=1   (relations exist only in v2)
    #   TOXAGENT_FLAG_SUBJECTLESS_RESEARCH_V1=1  (a claim names no molecule)

Running::

    TOXAGENT_STUDY_TOKEN=... python -m evals.external.scifact.product \\
        --base-url http://127.0.0.1:8011 --split dev --limit 25 --seed 20260927
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from evals.external.scifact import data as data_module
from evals.external.scifact.metrics import NEI, compute_f1
from evals.external.scifact.run import select
from evals.investigation.adapters.toxagent_api import ToxAgentAPI
from evals.investigation.record import environment

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs"
SCHEMA_VERSION = "scifact-product-run-v1"

#: Bumped when the question wording changes; recorded with every run, because
#: the wording is part of what was measured.
QUESTION_VERSION = "scifact-product-question-1"

#: The claim becomes a question, and nothing else is added: no hint that the
#: corpus contains the answer, no instruction about what to conclude, no
#: mention of SUPPORT/CONTRADICT. What the product does with it is the
#: measurement.
QUESTION_TEMPLATE = (
    "Does the published literature support or contradict this claim? "
    "Claim: \"{claim}\" "
    "Search the literature, and for each paper you read say whether it supports "
    "or contradicts the claim."
)

#: RelationLabel -> the official prediction vocabulary. Only a stance maps: a
#: ``contextual``, ``insufficient`` or ``not_applicable`` relation is the
#: product declining to take a side, which is the official format's "absent",
#: i.e. NEI. Mapping either of those to a label would invent a judgement.
RELATION_TO_LABEL: Mapping[str, str] = {
    "supports": "SUPPORT",
    "contradicts": "CONTRADICT",
}

_EVIDENCE_SOURCE_CLASSES = frozenset(
    {"external_experimental", "external_regulatory", "external_other"}
)


def question_for(claim: str) -> str:
    return QUESTION_TEMPLATE.format(claim=claim.strip())


def corpus_lines(corpus: Mapping[int, data_module.Document]) -> list[str]:
    """The pinned corpus in ``research-corpus-v1`` form, sorted by doc id so the
    file — and therefore its hash — depends only on the release."""
    return [
        json.dumps(
            {
                "record_id": str(doc_id),
                "title": corpus[doc_id].title,
                "sentences": list(corpus[doc_id].sentences),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for doc_id in sorted(corpus)
    ]


def write_corpus(corpus: Mapping[int, data_module.Document], path: Path) -> str:
    payload = ("\n".join(corpus_lines(corpus)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def corpus_doc_ids(evidence: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """evidence_id -> corpus doc id, for the records this run's corpus served.

    A record the corpus did not serve has no doc id and is left out of the
    predictions: the official format only knows this corpus's abstracts.
    """
    mapping: dict[str, int] = {}
    for record in evidence:
        facts = record.get("normalized_facts") or {}
        record_id = facts.get("corpus_record_id")
        evidence_id = record.get("evidence_id")
        if not record_id or not evidence_id:
            continue
        try:
            mapping[str(evidence_id)] = int(record_id)
        except (TypeError, ValueError):
            continue
    return mapping


def predictions_for_claim(
    relations: Iterable[Mapping[str, Any]], doc_ids: Mapping[str, int]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """One claim's abstract labels, plus what was dropped and why.

    The product may assess the same abstract under several propositions. When
    those disagree the abstract is left out (NEI) rather than resolved by a
    rule this module invents; the conflict is reported.
    """
    stances: dict[int, set[str]] = {}
    skipped: list[dict[str, Any]] = []
    for relation in relations:
        source = relation.get("source_ref") or {}
        source_class = source.get("source_class")
        source_id = str(source.get("source_id") or "")
        label = RELATION_TO_LABEL.get(str(relation.get("relation")))
        if source_class not in _EVIDENCE_SOURCE_CLASSES:
            # A predictor fact or a synthesis is not one of this corpus's
            # abstracts; it is not a prediction about any doc id.
            continue
        if source_id not in doc_ids:
            skipped.append({"reason": "source_not_in_corpus", "source_id": source_id,
                            "relation": relation.get("relation")})
            continue
        if label is None:
            skipped.append({"reason": "no_stance", "doc_id": doc_ids[source_id],
                            "relation": relation.get("relation")})
            continue
        stances.setdefault(doc_ids[source_id], set()).add(label)
    predictions: dict[int, dict[str, Any]] = {}
    for doc_id, labels in sorted(stances.items()):
        if len(labels) != 1:
            skipped.append({"reason": "conflicting_stances", "doc_id": doc_id,
                            "labels": sorted(labels)})
            continue
        # The product selects no rationale sentences: an empty list is the
        # honest value, and it makes every sentence-level metric undefined
        # rather than zero.
        predictions[doc_id] = {"label": labels.pop(), "sentences": []}
    return predictions, skipped


def abstract_label_only(
    predictions: Mapping[int, Mapping[int, Mapping[str, Any]]],
    gold: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """The one official metric that is defined here, computed exactly as
    ``metrics.compute_metrics`` computes it — same loop, same NEI rule — with
    the three rationale-dependent metrics left out instead of zeroed."""
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing or extra:
        raise ValueError(
            f"predictions must cover exactly the gold claims; missing {missing[:5]}, extra {extra[:5]}"
        )
    counts = {"relevant": 0, "retrieved": 0, "correct_label_only": 0}
    for claim_id, gold_claim in gold.items():
        counts["relevant"] += len(gold_claim)
        for doc_id, prediction in predictions[claim_id].items():
            if prediction["label"] == NEI:
                continue
            counts["retrieved"] += 1
            if doc_id in gold_claim and prediction["label"] == gold_claim[doc_id]["label"]:
                counts["correct_label_only"] += 1
    return {
        "abstract_label_only": compute_f1(counts, "label_only"),
        "counts": {"abstract": counts},
        "not_computed": {
            "sentence_selection": "the product selects no rationale sentences",
            "sentence_label": "the product selects no rationale sentences",
            "abstract_rationalized": "requires rationale sentences",
        },
    }


async def answer_one(
    api: ToxAgentAPI, claim: data_module.Claim
) -> dict[str, Any]:
    """One claim: one session, one question, then everything the run recorded."""
    question = question_for(claim.claim)
    started = datetime.now(timezone.utc)
    session_id = await api.new_session("en")
    record: dict[str, Any] = {
        "claim_id": claim.id, "claim": claim.claim, "question": question,
        "session_id": session_id, "started_at": started.isoformat(),
    }
    try:
        run = await api.send(session_id, question)
    except Exception as exc:  # noqa: BLE001 - counted, never dropped
        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return record
    run_id = run.get("run_id")
    answer = await api.answer_for_run(session_id, run_id) if run_id else None
    relations = (
        (await api.get(f"/v1/sessions/{session_id}/runs/{run_id}/evidence-relations") or {})
        .get("evidence_relations", [])
        if run_id else []
    )
    evidence = await api.evidence(session_id)
    doc_ids = corpus_doc_ids(evidence)
    predictions, skipped = predictions_for_claim(relations, doc_ids)
    record.update({
        "status": "ok" if run.get("status") == "completed" else "error",
        "run_id": run_id,
        "run_status": run.get("status"),
        "run_error": run.get("error") or run.get("error_code"),
        "answer_markdown": (answer or {}).get("answer_markdown"),
        "answer_id": (answer or {}).get("answer_id"),
        "tool_calls": [
            {"tool_name": c.get("tool_name"), "status": c.get("status"),
             "error_code": c.get("error_code")}
            for c in run.get("tool_calls") or ()
        ],
        "usage": run.get("usage"),
        "evidence_records": len(evidence),
        "corpus_records_served": len(doc_ids),
        "evidence_relations": list(relations),
        "predictions": {str(k): v for k, v in predictions.items()},
        "skipped_relations": skipped,
        "duration_s": (datetime.now(timezone.utc) - started).total_seconds(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    if run.get("status") != "completed":
        record["error"] = f"run {run.get('status')}"
    return record


async def run_study(
    *, claims: list[data_module.Claim], base_url: str, token: str, out: Path,
    corpus_sha256: str, parallel: int = 1, run_timeout_s: float = 1200.0,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, parallel))
    records: list[dict[str, Any]] = []
    async with ToxAgentAPI(base_url, token, run_timeout_s=run_timeout_s) as api:
        product = await api.effective_product()
        problems = product_problems(product, corpus_sha256=corpus_sha256)
        if problems:
            raise SystemExit(
                "this deployment cannot produce the measurement:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )

        async def one(claim: data_module.Claim) -> None:
            async with semaphore:
                record = await answer_one(api, claim)
                records.append(record)
                with (out / "turns.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(claim) for claim in claims))
    predictions = {
        claim.id: {
            int(doc_id): prediction
            for doc_id, prediction in (
                next((r for r in records if r["claim_id"] == claim.id), {}).get("predictions") or {}
            ).items()
        }
        for claim in claims
    }
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for claim in claims:
            evidence = {str(d): p for d, p in sorted(predictions[claim.id].items())}
            handle.write(json.dumps({"id": claim.id, "evidence": evidence}) + "\n")
    return {"records": records, "predictions": predictions, "effective_product": product}


#: Flags the study reads off the deployment. The first two are requirements;
#: the case ones are recorded because they change what is measured.
_RECORDED_FLAGS = (
    "answer_draft_v2", "subjectless_research_v1", "scientific_case_v1", "scientific_skills_v1",
)


def product_state(product: Mapping[str, Any]) -> dict[str, Any]:
    """What this study needs to know about the deployment it is measuring."""
    providers = product.get("providers") or {}
    flags = product.get("flags") or {}
    return {
        "research_provider": providers.get("research_provider"),
        "research_corpus": providers.get("research_corpus"),
        "flags": {
            name: bool((flags.get(name) or {}).get("enabled")) for name in _RECORDED_FLAGS
        },
    }


def product_problems(product: Mapping[str, Any], *, corpus_sha256: str) -> list[str]:
    """Why this deployment cannot produce the measurement, if it cannot.

    Checked before a single claim is sent: a run against a deployment serving
    EuropePMC, or one whose answers carry no relations, would produce a file of
    zeros that looks like a result.
    """
    state = product_state(product)
    problems: list[str] = []
    if state["research_provider"] != "corpus":
        problems.append(
            f"research_provider is {state['research_provider']!r}, not 'corpus': the product would "
            "search the live literature instead of the pinned benchmark corpus"
        )
    else:
        served = (state["research_corpus"] or {}).get("sha256")
        if served != corpus_sha256:
            problems.append(
                f"the deployment serves corpus sha256 {served!r}, not the {corpus_sha256!r} built "
                "from the pinned release"
            )
    if not state["flags"]["answer_draft_v2"]:
        problems.append(
            "answer_draft_v2 is off, so accepted answers carry no evidence_relations and there is "
            "no abstract label to read"
        )
    if not state["flags"]["subjectless_research_v1"]:
        problems.append(
            "subjectless_research_v1 is off, so a claim naming no molecule is answered with a "
            "clarification and no search runs"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write-corpus", type=Path, default=None,
                        help="write the pinned corpus for the provider and print its sha256")
    parser.add_argument("--base-url", default=None, help="the ToxAgent deployment to measure")
    parser.add_argument("--token", default=None, help="defaults to $TOXAGENT_STUDY_TOKEN")
    parser.add_argument("--split", default="dev", choices=data_module.SPLITS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--parallel", type=int, default=1,
                        help="claims at a time; 1 unless the deployment is known to take more")
    parser.add_argument("--run-timeout", type=float, default=1200.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache", type=Path, default=data_module.DEFAULT_CACHE)
    parser.add_argument("--compare-with", default=None,
                        help="a run.py run id over the same claims, recorded for comparison")
    args = parser.parse_args(argv)

    data_dir = data_module.ensure_release(args.cache)
    corpus = data_module.load_corpus(data_dir)
    if args.write_corpus:
        digest = write_corpus(corpus, args.write_corpus)
        print(json.dumps({"path": str(args.write_corpus), "records": len(corpus),
                          "sha256": digest}, indent=2))
        return 0
    if not args.base_url:
        parser.error("--base-url is required (or --write-corpus)")
    token = args.token or os.environ.get("TOXAGENT_STUDY_TOKEN", "")
    if not token:
        parser.error("--token or TOXAGENT_STUDY_TOKEN is required")
    if args.split == "test":
        parser.error("the test split has no public labels; this study scores against gold")

    all_claims = data_module.load_claims(data_dir, args.split)
    claims = select(all_claims, limit=args.limit, seed=args.seed)
    run_id = args.run_id or f"product-{args.split}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = args.out_root / run_id
    corpus_digest = hashlib.sha256(
        ("\n".join(corpus_lines(corpus)) + "\n").encode("utf-8")
    ).hexdigest()
    result = asyncio.run(run_study(
        claims=claims, base_url=args.base_url, token=token, out=out,
        corpus_sha256=corpus_digest, parallel=args.parallel, run_timeout_s=args.run_timeout,
    ))
    records = result["records"]
    metrics = abstract_label_only(result["predictions"], data_module.gold_of(claims))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        # Never `external-native`: see this module's docstring.
        "result_label": "published-data-transfer",
        "benchmark": {
            "name": "SciFact", "release_url": data_module.RELEASE_URL,
            "release_sha256": data_module.RELEASE_SHA256,
            "split": args.split, "claims_in_split": len(all_claims),
            "claims_run": len(claims),
            "subset_seed": None if len(claims) == len(all_claims) else args.seed,
            "claim_ids": [c.id for c in claims],
            "corpus_records": len(corpus),
            "corpus_sha256_expected": corpus_digest,
            "metrics": "abstract_label_only only; port of verisci/evaluate/lib/metrics.py @ 68b98a56",
        },
        "system": {
            "description": "the ToxAgent product answering each claim as a question, "
                           "retrieving over the pinned corpus with its own queries",
            "base_url": args.base_url,
            "question_version": QUESTION_VERSION,
            "question_template_sha256": hashlib.sha256(
                QUESTION_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "effective_product": result["effective_product"],
            "requirements_seen": product_state(result["effective_product"]),
        },
        "turns": {
            "total": len(records),
            "ok": sum(1 for r in records if r.get("status") == "ok"),
            "errors": sum(1 for r in records if r.get("status") != "ok"),
            "claims_with_no_relation": sum(1 for r in records if not r.get("evidence_relations")),
            "relations_skipped": sum(len(r.get("skipped_relations") or ()) for r in records),
        },
        "compare_with": args.compare_with,
        "metrics": metrics,
        "environment": environment(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ("run_id", "result_label", "turns")}, indent=2))
    print(json.dumps(metrics["abstract_label_only"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
