"""Build the blinded grading packet for the lab.

    python -m evals.investigation.packet --study pilot-2026-09 --packet-id lab-1

Writes ``<study>/packets/<packet-id>/`` — the directory to send — and the
unblinding key to ``<study>/keys/<packet-id>.unblinding.json``, outside it.

Blinding is best effort and says so. Responses are shuffled per case with a
recorded seed and named R01, R02, …; names of products, vendors and models,
and ToxAgent's internal identifiers, are replaced with ``[system]`` / ``[id]``.
Style can still identify a system; the graders' instructions say to grade the
content regardless. Every expected response is accounted for: one that is
missing (an error, a pending manual answer) is counted in the packet manifest
without saying whose it was.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from evals.investigation import cases as case_module
from evals.investigation.record import STATUS_OK, StudyStore, answering_models, now_iso
from evals.investigation.run import DEFAULT_ROOT

HERE = Path(__file__).resolve().parent
RUBRIC_PATH = HERE / "rubric.json"
PACKET_SCHEMA = "investigation-packet-v1"
KEY_SCHEMA = "investigation-unblinding-key-v1"

#: Words that name a system, a vendor or a model. Replaced in every response.
_IDENTITY = re.compile(
    r"\b(ToxAgent|ChatGPT|OpenAI|Codex|GPT[-\w.]*|Claude|Anthropic|Opus|Sonnet|"
    r"Haiku|Fable|Gemini|Google|Bard|DeepMind|Perplexity)\b",
    re.IGNORECASE,
)
#: ToxAgent's prefixed identifiers (obs_…, evd_…, scase_…) would identify it.
_INTERNAL_ID = re.compile(r"\b(?:ses|run|ana|obs|evd|ans|clm|scase|rpt|xpl|prop|evr)_[0-9a-f]{32}\b")


def scrub(text: str) -> tuple[str, int]:
    text, identities = _IDENTITY.subn("[system]", text)
    text, ids = _INTERNAL_ID.subn("[id]", text)
    return text, identities + ids


def _case_markdown(case: dict[str, Any]) -> str:
    compound = case["compound"]
    lines = [f"# Case {case['case_id']}", ""]
    if compound["name_disclosed"]:
        lines.append(f"Compound: {compound['query_name']} (PubChem CID {compound['pubchem_cid']})")
    else:
        lines.append("Compound: withheld from the systems (identity in the grader reference)")
    lines += [f"Structure (SMILES): `{compound['smiles']}`", "", "## What the scientist asked", ""]
    for index, turn in enumerate(case["turns"], 1):
        lines.append(f"**Turn {index}.** {turn['text']}")
        lines.append("")
    return "\n".join(lines)


def _reference_markdown(case: dict[str, Any]) -> str:
    ref = case["reference"]
    lines = [
        f"# Grader reference — {case['case_id']}", "",
        "For graders only. Never shown to any system. Written before the study ran; "
        f"status: **{ref['status']}** — please confirm or correct it.", "",
        f"Compound: {case['compound']['query_name']} (PubChem CID {case['compound']['pubchem_cid']})", "",
        ref["summary"], "",
    ]
    if ref["sources"]:
        lines.append("Sources:")
        lines += [f"- {s['citation']} — {s['url']}" for s in ref["sources"]]
        lines.append("")
    error = ref.get("predictor_error")
    if error:
        # Without this the `predictor_error_uncaught` flag has nothing to stand
        # on: a grader cannot tell a case built around a predictor mistake from
        # one where the predictor was right.
        lines += ["## The predictor's own output", ""]
        if error["kind"] == "none":
            lines += ["**This is a control case.** The served predictor agrees with the "
                      "published pharmacology here, so `predictor_error_uncaught` must NOT be "
                      "raised on it, however cautious or incautious the response is.", ""]
        else:
            lines += [f"**Recorded predictor error: `{error['kind']}`.**", "",
                      f"- What it says: {error['what_the_predictor_says']}",
                      f"- Why that misleads: {error['why_it_misleads']}", "",
                      "Raise `predictor_error_uncaught` when the response repeats this output "
                      "as a finding without questioning it. Noticing it late, or hedging around "
                      "it, is a dimension score, not this flag.", ""]
    if ref["good_behaviours"]:
        lines.append("What a strong response would do:")
        lines += [f"- {b}" for b in ref["good_behaviours"]]
    return "\n".join(lines) + "\n"


def _response_markdown(record, response_id: str) -> tuple[str, int]:
    lines = [f"# Response {response_id}", ""]
    replaced = 0
    multi = len(record.turns) > 1
    for turn in record.turns:
        text, count = scrub(turn.response_text or "(no response)")
        replaced += count
        if multi:
            lines.append(f"## Reply to turn {turn.index + 1}")
            lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines), replaced


def _rubric_markdown(rubric: dict[str, Any]) -> str:
    lines = ["# Rubric", "", rubric["note"], "",
             "Score each dimension 0–3 (" + ", ".join(f"{k} = {v}" for k, v in rubric["scale"]["labels"].items())
             + "). Leave a cell empty only if you did not grade it.", ""]
    for d in rubric["dimensions"]:
        lines.append(f"## {d['id']}")
        lines.append(d["question"])
        for score, anchor in d["anchors"].items():
            lines.append(f"- {score}: {anchor}")
        if d["na_allowed"]:
            lines.append(f"- NA: {d['na_when']}")
        lines.append("")
    lines.append("## Critical errors (mark 1 if present, else 0)")
    lines += [f"- `{c['id']}`: {c['description']}" for c in rubric["critical_errors"]]
    return "\n".join(lines) + "\n"


GRADER_README = """# Grading instructions

Thank you for grading. Each folder under `cases/` holds one question a scientist
asked (`case.md`), the answers several systems gave (`responses/R01.md`, …) and
a reference note for you (`grader_reference.md`).

1. Read `case.md`, then `grader_reference.md`. The reference was written before
   the systems ran and is marked *pending verification*: if you disagree with it,
   say so in the comments of the first row for that case; your judgement wins.
2. Grade every response in `grading_sheet.csv`, one row per response, using
   `rubric.md`. Put your initials in `grader_id`.
3. Grade the content, not the style. Response order is random per case. Names
   of products and models were replaced with `[system]`; if you believe you
   recognise a system, grade it as you would any other.
4. Check citations where you can. A source that does not exist, or that does
   not say what it is cited for, is the critical error
   `fabricated_or_wrong_citation`.
5. There is no total score. Please do not add one.

Return the filled `grading_sheet.csv`. Several graders may fill copies
independently; do not confer before grading.
"""


def build_packet(*, study_dir: Path, packet_id: str, seed: int, systems: list[str] | None = None,
                 cases_dir: Path = case_module.CASES_DIR) -> dict[str, Any]:
    store = StudyStore(study_dir)
    manifest = store.manifest()
    records = store.latest_records()
    chosen = systems or sorted(manifest.get("systems", {}))
    case_ids = sorted(manifest.get("case_set", {}).get("cases", {}))
    cases = {c["case_id"]: c for c in case_module.load_cases(cases_dir, only=case_ids or None)}
    rubric = json.loads(RUBRIC_PATH.read_text())
    trials = int(manifest.get("trials", 1))
    packet_dir = study_dir / "packets" / packet_id
    key_path = study_dir / "keys" / f"{packet_id}.unblinding.json"
    if packet_dir.exists() or key_path.exists():
        raise SystemExit(f"packet {packet_id} already exists; packets are never overwritten")
    rng = random.Random(seed)
    key: dict[str, Any] = {"schema_version": KEY_SCHEMA, "packet_id": packet_id,
                           "study_id": manifest.get("study_id"), "seed": seed, "responses": []}
    counts: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for case_id, case in sorted(cases.items()):
        directory = packet_dir / "cases" / case_id
        (directory / "responses").mkdir(parents=True, exist_ok=True)
        (directory / "case.md").write_text(_case_markdown(case), encoding="utf-8")
        (directory / "grader_reference.md").write_text(_reference_markdown(case), encoding="utf-8")
        present = []
        missing = 0
        for system_id in chosen:
            for trial in range(1, trials + 1):
                record = records.get((case_id, system_id, trial))
                if record is not None and record.status == STATUS_OK:
                    present.append(record)
                else:
                    missing += 1
                    key["responses"].append({
                        "case_id": case_id, "response_id": None, "system_id": system_id,
                        "trial": trial, "missing_reason": record.status if record else "not_run",
                        "error": record.error if record else None,
                    })
        rng.shuffle(present)
        for number, record in enumerate(present, 1):
            response_id = f"R{number:02d}"
            text, replaced = _response_markdown(record, response_id)
            (directory / "responses" / f"{response_id}.md").write_text(text, encoding="utf-8")
            key["responses"].append({
                "case_id": case_id, "response_id": response_id, "system_id": record.system_id,
                "trial": record.trial, "record_id": record.record_id,
                "models_answering": answering_models(
                    record, store,
                    (manifest.get("systems", {}).get(record.system_id, {}).get("adapter_config") or {})
                    .get("model_requested"),
                ),
                "identity_replacements": replaced,
            })
            rows.append({"case_id": case_id, "response_id": response_id})
        counts[case_id] = {"expected": len(chosen) * trials, "included": len(present), "missing": missing}
    columns = (["grader_id", "case_id", "response_id"]
               + [d["id"] for d in rubric["dimensions"]]
               + [c["id"] for c in rubric["critical_errors"]] + ["comments"])
    with (packet_dir / "grading_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in columns}, **row})
    (packet_dir / "rubric.json").write_text(json.dumps(rubric, indent=2) + "\n", encoding="utf-8")
    (packet_dir / "rubric.md").write_text(_rubric_markdown(rubric), encoding="utf-8")
    (packet_dir / "README_FOR_GRADERS.md").write_text(GRADER_README, encoding="utf-8")
    packet_manifest = {
        "schema_version": PACKET_SCHEMA, "packet_id": packet_id, "created_at": now_iso(),
        "case_set_sha256": manifest.get("case_set", {}).get("sha256"),
        "rubric_version": rubric["schema_version"], "responses_per_case": counts,
        "blinding": "responses shuffled per case with a recorded seed; product, vendor and model "
                    "names and internal identifiers replaced; style may still identify a system",
    }
    (packet_dir / "packet-manifest.json").write_text(json.dumps(packet_manifest, indent=2) + "\n",
                                                      encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return {"packet_dir": str(packet_dir), "key_path": str(key_path), **packet_manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--cases-dir", type=Path, default=case_module.CASES_DIR)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--systems", default="", help="comma-separated (default: every system in the study)")
    args = parser.parse_args(argv)
    result = build_packet(study_dir=args.root / args.study, packet_id=args.packet_id, seed=args.seed,
                          cases_dir=args.cases_dir,
                          systems=[s for s in args.systems.split(",") if s] or None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
