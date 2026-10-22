"""Hallucination as atomic-claim precision (protocol v2, master plan §14.1-§14.2).

Task Success says whether a reply did the job; it does not say whether the
reply also said things that are not so. This measures that, claim by claim.

A grader splits each final reply into atomic factual claims -- one checkable
statement each -- and labels every claim against **what the system was given**
(the frozen artifacts and the evidence corpus, printed in the packet as
``sources.md``), not against the world and not against the gold:

* ``supported`` -- the sources state it, or it follows directly from what they
  state (a unit conversion, a ratio of two stated values);
* ``unsupported`` -- nothing the system was given states it: a value, assay,
  source, mechanism or status with no basis. This is a hallucination;
* ``contradicted`` -- the sources state otherwise. Also a hallucination, and
  the worse kind;
* ``not_checkable`` -- not a factual claim: a recommendation, a caveat, a
  question, a description of what the system will or will not do.

A reply with no factual claim at all is one row labelled ``no_claims``: graded,
and not a free 100%. A reply that never came is not in this measure (Task
Success already counts it).

Claim segmentation differs between graders, so claims are never aligned across
graders. Each grader's numbers are reported apart, and agreement is measured
per reply: did both graders find a hallucination in it or not. A model may help
segment and label; the plan requires an SME audit of a sample before these
numbers are read as factual precision.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evals.investigation.scorecard import GradeError

SCHEMA_VERSION = "atomic-claims-v1"
LABELS = ("supported", "unsupported", "contradicted", "not_checkable", "no_claims")
HALLUCINATED = frozenset({"unsupported", "contradicted"})
CHECKABLE = frozenset({"supported", "unsupported", "contradicted"})
COLUMNS = ("grader_id", "case_id", "response_id", "turn", "claim_index", "claim", "label",
           "source", "note")

INSTRUCTIONS = """# Atomic claims: how to fill `claims_sheet.csv`

This measures hallucination: statements a reply makes that the system was never
given. It is separate from `grading_sheet.csv` and graded independently.

For every response file, take the **final result of each turn** (not the earlier
stages in `<details>`) and split it into atomic factual claims: one checkable
statement each. "The model predicts 0.21, labelled non_blocker at 0.5" is three
claims (the probability, the label, the threshold). Then label each claim against
`sources.md` of that case -- the frozen artifacts and every record in the corpus
the system could search. Not against the reference, and not against what you
know about the world:

- `supported`: the sources state it, or it follows directly (a unit conversion,
  a ratio of two stated values, "0.3 uM is below 10 uM");
- `unsupported`: nothing in the sources states it -- an invented value, assay,
  source, mechanism, status, or a claim that the material lacks something it
  contains;
- `contradicted`: the sources state otherwise;
- `not_checkable`: not a factual claim (a recommendation, a caveat, a question,
  what the system did or could not do).

One row per claim: `grader_id, case_id, response_id, turn` (1-based),
`claim_index` (1, 2, ... within the response), `claim` (short paraphrase or
quote), `label`, `source` (the record id or artifact field that supports or
contradicts it; empty for unsupported), `note`. A response whose final result
makes no factual claim gets a single row with `claim_index` 0 and label
`no_claims`. Do not skip responses: an absent response is read as ungraded.
"""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _screening_lines(record: Mapping[str, Any]) -> list[str]:
    state = record.get("screening_state") or {}
    subject = state.get("subject") or {}
    target = subject.get("target") or {}
    names = ", ".join(record.get("subject_names") or ()) or "none recorded"
    lines = [f"### Screening `{state.get('screening_id')}`", "",
             f"- ligand: `{subject.get('canonical_smiles')}` (names: {names})",
             f"- target: {target.get('target_id')} (UniProt {target.get('uniprot_id')}); "
             f"assay context: {subject.get('assay_context')}",
             f"- screening status: {state.get('status')}"]
    estimate = (record.get("activity_estimate") or {}).get("estimate")
    if estimate:
        model = (record.get("activity_estimate") or {}).get("model") or {}
        lines.append(
            f"- activity estimate: predicted pIC50 {_fmt(estimate.get('predicted_pic50'))}, "
            f"{_fmt(estimate.get('interval_level'))} interval "
            f"{_fmt(estimate.get('prediction_interval'))} ({estimate.get('interval_kind')}); "
            f"model {model.get('model_id')}, admitted: {model.get('admitted')}")
    else:
        lines.append(f"- activity estimate: none (activity failure code: "
                     f"{state.get('activity_failure_code')})")
    decision = record.get("prioritisation_decision")
    if decision:
        policy = decision.get("policy") or {}
        lines.append(
            f"- policy decision: {decision.get('status')} "
            f"(reasons: {', '.join(decision.get('reason_codes') or ()) or 'none'}), "
            f"threshold pIC50 {_fmt(policy.get('threshold_pic50'))} "
            f"({policy.get('threshold_source')})")
    toxicity = state.get("toxicity") or {}
    lines.append(f"- toxicity stage ran: {toxicity.get('run')} "
                 f"(not-run reason: {toxicity.get('not_run_reason')})")
    return lines


def _analysis_lines(analysis: Mapping[str, Any]) -> list[str]:
    response = analysis.get("predictor_response") or {}
    predictions = response.get("predictions") or {}
    lines = [f"### Toxicity analysis `{analysis.get('origin_analysis_id')}`", "",
             f"- structure: `{analysis.get('canonical_smiles')}` (no compound name recorded)"]
    herg = predictions.get("herg")
    if herg:
        lines.append(f"- hERG: probability_blocker {_fmt(herg.get('probability_blocker'))}, "
                     f"label {herg.get('label')} at threshold {_fmt(herg.get('threshold'))} "
                     f"({herg.get('threshold_source')}), model {herg.get('model_id')}")
    tox21 = (predictions.get("tox21") or {}).get("assays") or {}
    if tox21:
        lines.append("- Tox21: " + "; ".join(
            f"{task} {_fmt(row.get('probability_activity'))} "
            f"({'active' if row.get('active') else 'inactive'})"
            for task, row in sorted(tox21.items())))
    applicability = response.get("applicability") or {}
    if applicability:
        lines.append(f"- applicability: {applicability.get('status')} "
                     f"({applicability.get('method')})")
    return lines


def sources_markdown(case_set, inputs: Mapping[str, Any]) -> str:
    """What one case's systems were given: its frozen artifacts and the whole
    corpus they could search. Model-facing material only -- never the gold."""
    replay = json.loads(case_set.replay_path.read_text()) if case_set.replay_path else {}
    evidence = json.loads(case_set.evidence_path.read_text()) if case_set.evidence_path else {}
    screenings = {r["screening_state"]["screening_id"]: r for r in replay.get("screenings") or ()}
    analyses = replay.get("analyses") or {}
    if isinstance(analyses, list):
        analyses = {a["origin_analysis_id"]: a for a in analyses}
    lines = [f"# Sources for {inputs.get('case_id')}", "",
             "Everything below is what every system was given for this case: the frozen "
             "artifacts, then every record of the corpus it could search. Label claims against "
             "this, not against the reference and not against the world.", "",
             "## Frozen artifacts", ""]
    referenced = {a.get("ref") for a in inputs.get("artifacts") or ()}
    for turn in inputs.get("turns") or ():
        for action in turn.get("actions") or ():
            ref = (action.get("subject_ref") or {})
            referenced.add(ref.get("screening_id") or ref.get("analysis_ref"))
    for ref in sorted(r for r in referenced if r):
        if ref in screenings:
            lines += _screening_lines(screenings[ref]) + [""]
        elif ref in analyses:
            lines += _analysis_lines(analyses[ref]) + [""]
    for index, turn in enumerate(inputs.get("turns") or (), 1):
        for action in turn.get("actions") or ():
            if action.get("type") == "add_context":
                lines += [f"- researcher context at turn {index}: {action.get('key')} = "
                          f"{action.get('value')} {action.get('unit') or ''} "
                          f"(source: {action.get('source')})".rstrip(), ""]
    body = evidence.get("evidence") or {}
    records = body.get("records") or {}
    documents = (body.get("full_text") or {}).get("documents") or {}
    lines += ["## Evidence corpus", ""]
    for record_id, record in sorted(records.items()):
        lines += [f"### {record_id} -- {record.get('title')}", "",
                  f"- abstract: {record.get('abstract_or_excerpt')}"]
        document = documents.get(record_id) or {}
        if document.get("outcome") == "retrieved":
            lines.append("- full text (passages by locator):")
            lines += [f"  - `{p.get('locator')}` [{p.get('kind')}]: {p.get('text')}"
                      for p in document.get("passages") or ()]
        else:
            lines.append(f"- full text: not available ({document.get('reason') or 'not held'})")
        lines.append("")
    return "\n".join(lines) + "\n"


def read_claim_sheets(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Claim rows, validated. An unknown label is an error, never a guess."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for number, row in enumerate(csv.DictReader(handle), 2):
                where = f"{Path(path).name}:{number}"
                for column in ("grader_id", "case_id", "response_id", "label"):
                    if column not in row:
                        raise GradeError(f"{where}: the claims sheet has no {column} column")
                label = (row.get("label") or "").strip()
                if label not in LABELS:
                    raise GradeError(f"{where}: label must be one of {LABELS}, got {label!r}")
                rows.append({
                    "grader_id": (row.get("grader_id") or "").strip() or Path(path).stem,
                    "case_id": row["case_id"].strip(), "response_id": row["response_id"].strip(),
                    "turn": int(row.get("turn") or 0), "claim": (row.get("claim") or "").strip(),
                    "label": label, "source": (row.get("source") or "").strip(),
                    "note": row.get("note") or "",
                })
    return rows


def _row() -> dict[str, Any]:
    return {"responses": 0, "graded": 0, "no_claims": 0, "with_checkable": 0,
            "with_hallucination": 0, "claims": 0, "checkable": 0, "supported": 0,
            "unsupported": 0, "contradicted": 0, "not_checkable": 0}


def _finish(row: dict[str, Any]) -> dict[str, Any]:
    checkable = row["checkable"]
    return {
        **row,
        "claim_precision": round(row["supported"] / checkable, 4) if checkable else None,
        "hallucination_rate": (round((row["unsupported"] + row["contradicted"]) / checkable, 4)
                               if checkable else None),
        "responses_with_hallucination_rate": (
            round(row["with_hallucination"] / row["with_checkable"], 4)
            if row["with_checkable"] else None),
    }


def hallucination(*, key: Mapping[str, Any], claims: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per grader and arm: claim precision, hallucination rate, and the share
    of replies with any hallucinated claim; plus per-reply agreement."""
    responses = {(r["case_id"], r["response_id"]): r for r in key.get("responses") or ()
                 if r.get("response_id")}
    by_grader: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for claim in claims:
        by_grader[claim["grader_id"]][(claim["case_id"], claim["response_id"])].append(dict(claim))
    tables: dict[str, dict[str, Any]] = {}
    flags: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for grader, per_response in sorted(by_grader.items()):
        rows: dict[str, dict[str, Any]] = defaultdict(_row)
        for (case_id, response_id), response in responses.items():
            row = rows[response["system_id"]]
            row["responses"] += 1
            marks = per_response.get((case_id, response_id))
            if not marks:
                continue
            row["graded"] += 1
            labels = [m["label"] for m in marks]
            if set(labels) == {"no_claims"}:
                row["no_claims"] += 1
                flags[(case_id, response_id)][grader] = False
                continue
            factual = [m for m in marks if m["label"] != "no_claims"]
            row["claims"] += len(factual)
            for mark in factual:
                row[mark["label"]] += 1
            checkable = [m for m in factual if m["label"] in CHECKABLE]
            row["checkable"] += len(checkable)
            bad = [m for m in checkable if m["label"] in HALLUCINATED]
            if checkable:
                row["with_checkable"] += 1
            if bad:
                row["with_hallucination"] += 1
                examples[response["system_id"]] += [
                    {"grader": grader, "case_id": case_id, "response_id": response_id,
                     "label": m["label"], "claim": m["claim"], "note": m["note"]} for m in bad]
            flags[(case_id, response_id)][grader] = bool(bad)
        tables[grader] = {system: _finish(row) for system, row in sorted(rows.items())}
    both = [f for f in flags.values() if len(f) >= 2]
    agree = sum(len(set(f.values())) == 1 for f in both)
    disagreements = sorted(f"{c}/{r}" for (c, r), f in flags.items()
                           if len(f) >= 2 and len(set(f.values())) > 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": "claims within a reply's final result per turn; replies graded per grader",
        "by_grader": tables,
        "agreement": {"replies_graded_by_two": len(both),
                      "agree_on_any_hallucination": agree,
                      "rate": round(agree / len(both), 4) if both else None,
                      "disagreements": disagreements},
        "examples": {s: v[:8] for s, v in sorted(examples.items())},
        "note": "Labelled against what each system was given (sources.md), not the gold. "
                "No-claim replies are counted apart, never as perfect precision. LLM-assisted "
                "labels need an SME audit of a sample before they are read as factual precision.",
    }


def render_markdown(section: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Hallucination (atomic claims against the sources each system was given)", "",
             "| Grader | Arm | Replies graded | No-claim | Claims checked | Supported | "
             "Unsupported | Contradicted | Claim precision | Replies with a hallucination |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def pct(value):
        return "—" if value is None else f"{value:.0%}"

    for grader, table in section.get("by_grader", {}).items():
        for system, row in table.items():
            lines.append(
                f"| {grader} | `{system}` | {row['graded']}/{row['responses']} | "
                f"{row['no_claims']} | {row['checkable']} | {row['supported']} | "
                f"{row['unsupported']} | {row['contradicted']} | {pct(row['claim_precision'])} | "
                f"{row['with_hallucination']}/{row['with_checkable']} "
                f"({pct(row['responses_with_hallucination_rate'])}) |")
    agree = section.get("agreement") or {}
    lines += ["", f"- per-reply agreement on 'any hallucination': "
                  f"{agree.get('agree_on_any_hallucination')}/{agree.get('replies_graded_by_two')}"
                  f"; disagreements: {', '.join(agree.get('disagreements') or ()) or 'none'}",
              f"- {section.get('note')}"]
    return lines
