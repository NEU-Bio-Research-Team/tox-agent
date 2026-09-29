"""Run SciFact under its own protocol and score it with the official metrics.

    python -m evals.external.scifact.run --split dev --retrieval oracle \\
        --judge claude:opus --limit 50 --seed 20260925 --out-root evals/external/scifact/runs

Writes ``<run>/predictions.jsonl`` (the official submission format),
``judgments.jsonl`` (every judge call: prompt hash, raw output, model id,
timing, usage, parse errors), ``metrics.json`` and ``manifest.json``.

Labels (RETHINK §5.2): a full dev split run is ``external-native``; a subset
is ``external-native-subset`` and not comparable with published full-split
numbers; the test split has no public labels, so its predictions are written
for a leaderboard submission and not scored here. A judge output that cannot
be parsed counts as NOT_ENOUGH_INFO and is counted in the manifest, never
dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.external.scifact import data as data_module
from evals.external.scifact import judges as judge_module
from evals.external.scifact import retrieval
from evals.external.scifact.metrics import NEI, compute_metrics
from evals.investigation.record import environment

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs"
SCHEMA_VERSION = "scifact-run-v1"


def select(claims: list[data_module.Claim], *, limit: int | None, seed: int) -> list[data_module.Claim]:
    if not limit or limit >= len(claims):
        return claims
    chosen = set(random.Random(seed).sample([c.id for c in claims], limit))
    return [c for c in claims if c.id in chosen]


async def run(*, claims: list[data_module.Claim], corpus: dict[int, data_module.Document],
              retrieved: dict[int, list[int]], judge, out: Path, parallel: int = 3) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, parallel))
    judgments: list[dict[str, Any]] = []
    predictions: dict[int, dict[int, dict[str, Any]]] = {c.id: {} for c in claims}

    async def one(claim: data_module.Claim, doc_id: int) -> None:
        async with semaphore:
            document = corpus[doc_id]
            record: dict[str, Any] = {"claim_id": claim.id, "doc_id": doc_id,
                                      "prompt_sha256": judge_module.sha256(judge_module.render(claim, document))}
            try:
                result = await judge.judge(claim, document)
            except Exception as exc:  # noqa: BLE001 - counted, never dropped
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "label": NEI})
            else:
                record.update({"status": "ok", "label": result.label, "sentences": result.sentences,
                               "raw_text": result.raw_text, "model": result.model, "usage": result.usage,
                               "duration_s": result.duration_s, "parse_error": result.parse_error})
                if result.label != NEI:
                    predictions[claim.id][doc_id] = {"label": result.label, "sentences": result.sentences}
            judgments.append(record)
            with (out / "judgments.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(c, d) for c in claims for d in retrieved.get(c.id, [])))
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for claim in claims:
            evidence = {str(d): p for d, p in sorted(predictions[claim.id].items())}
            handle.write(json.dumps({"id": claim.id, "evidence": evidence}) + "\n")
    return {"predictions": predictions, "judgments": judgments}


def models_that_answered(judgments: list[dict[str, Any]], requested: str | None) -> dict[str, int]:
    """How many judgments each model actually wrote, from the stored per-model
    usage when the judge recorded it (resolver in ``platform_cli._main_model``)."""
    from collections import Counter

    from evals.investigation.adapters.platform_cli import _main_model

    counts: Counter = Counter()
    for judgment in judgments:
        model = judgment.get("model") or {}
        usage = model.get("model_usage")
        counts[str(_main_model(usage, requested) if usage else model.get("model_id_resolved"))] += 1
    return dict(sorted(counts.items()))


def summarise_existing(out: Path) -> dict[str, Any]:
    """Recompute a finished run's model counts and metrics from what it stored.

    No judge is called: predictions and judgments on disk are the record. Used
    when the model resolver improves after a run.
    """
    manifest = json.loads((out / "manifest.json").read_text())
    judgments = [json.loads(line) for line in (out / "judgments.jsonl").read_text().splitlines()]
    requested = manifest["system"].get("model_requested")
    # The field an earlier resolver wrote is superseded, not kept beside the new count.
    manifest["judgments"].pop("models_reported", None)
    manifest["judgments"]["models_answered"] = models_that_answered(judgments, requested)
    manifest["judgments"]["models_answered_note"] = (
        "per judgment, the model that wrote the output, from the stored per-model usage"
    )
    if "metrics" in manifest:
        data_dir = data_module.ensure_release(data_module.DEFAULT_CACHE)
        ids = set(manifest["benchmark"]["claim_ids"])
        claims = [c for c in data_module.load_claims(data_dir, manifest["benchmark"]["split"]) if c.id in ids]
        predictions = {}
        for line in (out / "predictions.jsonl").read_text().splitlines():
            entry = json.loads(line)
            predictions[int(entry["id"])] = {int(d): v for d, v in entry["evidence"].items()}
        manifest["metrics"] = compute_metrics(predictions, data_module.gold_of(claims))
    manifest["summarised_at"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=data_module.SPLITS)
    parser.add_argument("--retrieval", default="oracle", choices=("oracle", "tfidf"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-nei", action="store_true", help="oracle without NEI abstracts")
    parser.add_argument("--judge", default=None, help="claude[:model] or codex")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache", type=Path, default=data_module.DEFAULT_CACHE)
    parser.add_argument("--summarise", metavar="RUN_ID", default=None,
                        help="recompute an existing run's manifest from its stored outputs")
    args = parser.parse_args(argv)
    if args.summarise:
        manifest = summarise_existing(args.out_root / args.summarise)
        print(json.dumps(manifest["judgments"], indent=2))
        return 0

    if not args.judge:
        parser.error("--judge is required unless --summarise is given")
    data_dir = data_module.ensure_release(args.cache)
    corpus = data_module.load_corpus(data_dir)
    all_claims = data_module.load_claims(data_dir, args.split)
    claims = select(all_claims, limit=args.limit, seed=args.seed)
    retrieved = (retrieval.oracle(claims, include_nei=not args.no_nei) if args.retrieval == "oracle"
                 else retrieval.tfidf(claims, corpus, k=args.k))
    judge = judge_module.make_judge(args.judge)
    run_id = args.run_id or (f"{args.split}-{args.retrieval}-{judge.name.replace(':', '-')}-"
                             f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    out = args.out_root / run_id
    result = asyncio.run(run(claims=claims, corpus=corpus, retrieved=retrieved, judge=judge,
                             out=out, parallel=args.parallel))
    full = len(claims) == len(all_claims)
    label = ("external-native" if full else "external-native-subset")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "result_label": label,
        "benchmark": {"name": "SciFact", "release_url": data_module.RELEASE_URL,
                      "release_sha256": data_module.RELEASE_SHA256,
                      "metrics": "port of verisci/evaluate/lib/platform/metrics.py @ 68b98a56",
                      "split": args.split, "claims_in_split": len(all_claims),
                      "claims_run": len(claims), "subset_seed": None if full else args.seed,
                      "claim_ids": [c.id for c in claims]},
        "retrieval": {"setting": args.retrieval, "include_nei": not args.no_nei,
                      "k": args.k if args.retrieval == "tfidf" else None,
                      "abstracts_judged": sum(len(v) for v in retrieved.values())},
        "system": {"description": f"{judge.name} as a stand-alone claim verifier; not the ToxAgent product",
                   **judge.describe(), "prompt_version": judge_module.PROMPT_VERSION,
                   "prompt_sha256": judge_module.template_sha256()},
        "judgments": {
            "total": len(result["judgments"]),
            "errors": sum(1 for j in result["judgments"] if j["status"] == "error"),
            "parse_errors": sum(1 for j in result["judgments"] if j.get("parse_error")),
            "models_answered": models_that_answered(
                [j for j in result["judgments"] if j.get("model")],
                getattr(judge, "model", None),
            ),
        },
        "environment": environment(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.split != "test":
        metrics = compute_metrics(result["predictions"], data_module.gold_of(claims))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        manifest["metrics"] = metrics
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("run_id", "result_label", "judgments")}, indent=2))
    if "metrics" in manifest:
        print(json.dumps({m: {k: round(v, 4) for k, v in manifest["metrics"][m].items()}
                          for m in manifest["metrics"] if m != "counts"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
