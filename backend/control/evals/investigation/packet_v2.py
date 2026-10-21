"""The blinded grading packet of a protocol v2 study (#53).

    python -m evals.investigation.packet_v2 --study pilot-2026-11 \\
        --case-set evals/investigation/cases_protocol_v2_technical --packet-id lab-1

Writes ``<study>/packets/<packet-id>/`` -- the directory a grader is sent --
and the unblinding key beside it in ``<study>/keys/``, never inside it.

What a grader gets per case: the turns exactly as every arm received them
(the researcher's own words plus the context their actions carried), and a
reference built from the case's gold -- the reference outcome, the acceptable
conclusions, the facts the answer owes, the limitations it owes, the
inferences it must not make, and the spans the sources hold -- with the
gold's verification status printed at the top, because a synthetic fixture's
reference is machinery, not science.

What a grader gets per response: every stage the product recorded, in order
-- the raw draft as submitted, the first hard-valid candidate, the repaired
candidate, and the result the researcher actually saw (an answer, or the
incomplete result the release policy showed instead). Grading the final
result is the task; the earlier stages are there so a repair's effect can be
judged rather than guessed.

Blinding is best effort and says so: responses are shuffled per case with a
recorded seed, and product, vendor, model and internal identifiers are
replaced. Style can still identify an arm, and the instructions tell graders
to grade the content anyway. Every expected response is accounted for: one
that never came is counted in the manifest without saying whose it was.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

from evals.investigation import turn_actions
from evals.investigation.case_inputs import load_case_set
from evals.investigation.packet import RUBRIC_PATH, scrub
from evals.investigation.record import STATUS_OK, StudyStore, now_iso
from evals.investigation.run import DEFAULT_ROOT

PACKET_SCHEMA = "investigation-packet-v2"
KEY_SCHEMA = "investigation-unblinding-key-v2"

#: The columns a grader fills. The list columns take semicolon-separated ids
#: or codes; an empty cell is "not graded", never a zero.
LIST_COLUMNS = ("mandatory_facts_met", "limitations_stated", "forbidden_inferences_committed")

GRADER_README = """# Grading instructions (protocol v2)

Each folder under `cases/` is one task a researcher put to several systems:

* `case.md` -- the turns as every system received them, in order;
* `grader_reference.md` -- what the task requires. Read it after `case.md`;
* `responses/R01.md`, … -- one system's reply per file, in random order.

For every response, fill one row of `grading_sheet.csv`:

1. `task_success` is 1 when the response meets what the reference says the
   task requires -- an acceptable conclusion, with the claims that decide it
   sourced -- and 0 otherwise. On a case the reference marks as *needs user
   input*, naming the missing input and not over-concluding is a success; on
   a case the sources settle, a generic "not enough data" is not.
2. `mandatory_facts_met` -- the ids of the facts in the reference the
   response states correctly, separated by `;`. Leave empty for none.
3. `limitations_stated` -- the limitation codes (L1…L9) the response actually
   states in substance. A response that uses the word but asserts the
   opposite has not stated it.
4. `forbidden_inferences_committed` -- the codes from the reference the
   response commits, separated by `;`.
5. The critical-error columns -- 1 when present, 0 when you checked and it is
   absent, empty when you did not judge it. A critical error makes the
   response unsuccessful whatever you put in `task_success`.
6. `comments` -- anything the columns cannot carry, and any disagreement with
   the reference. The reference may be wrong; say so and your judgement is
   what gets adjudicated.

Only the **final result** of each turn is the task. The stages above it (raw
draft, candidate, repaired candidate) are there so you can see what a repair
removed; do not grade them as the answer.

There is no total score, and please do not add one. Several graders fill
copies independently -- do not confer before grading.
"""


def case_markdown(loaded, inputs: Mapping[str, Any]) -> str:
    """The case as every arm received it: its turns, with the context the
    researcher's actions carried, and nothing from the gold."""
    case = loaded.case
    subject = (inputs.get("subject") or {}).get("identity") or {}
    lines = [f"# Case {case['case_id']}", "",
             f"Workflow context: {case.get('workflow_context')}.", ""]
    if subject.get("name_disclosed") and subject.get("name"):
        lines.append(f"Compound: {subject['name']}")
    lines += [f"Structure (SMILES): `{subject.get('canonical_smiles')}`", "",
              "The system was given the frozen artifacts this case pins (a screening or a "
              "toxicity analysis, the evidence corpus and its full text). It was not given "
              "anything in the reference note.", "", "## What the researcher asked", ""]
    current = dict((inputs.get("subject") or {}).get("subject_ref") or {})
    for index, turn in enumerate(inputs.get("turns") or (), 1):
        before = current
        current = turn_actions.subject_for_turn(turn, current)
        if index > 1 and turn_actions.subject_key(current) != turn_actions.subject_key(before):
            lines += [f"*(The researcher now selects another subject: "
                      f"{turn_actions.subject_key(current)}.)*", ""]
        lines += [f"**Turn {index}.** {turn_actions.sent_text(turn)}", ""]
    return "\n".join(lines)


def reference_markdown(loaded, gold: Mapping[str, Any]) -> str:
    """The grader reference, from the case's gold file."""
    case = loaded.case
    status = str(gold.get("status"))
    lines = [f"# Grader reference — {case['case_id']}", "",
             "For graders only. Never shown to any system.", "",
             f"Gold status: **{status}**."]
    if status == "synthetic_engineering_gold":
        lines += ["", "> This case is a synthetic engineering fixture: its compounds, numbers "
                  "and sources are invented to test the machinery. Grade whether the response "
                  "does what the task asks of it; do not read any scientific claim into it."]
    elif status == "pending_sme_verification":
        lines += ["", "> This reference has not been verified yet. If you disagree with it, "
                  "say so in `comments`; your judgement is what gets adjudicated."]
    lines += ["", f"Task group: {case.get('task_group')}. Answerability: "
              f"**{case.get('answerability')}**.", "", "## Reference outcome", "",
              str(gold.get("reference_outcome")), "", "## Acceptable conclusions", ""]
    for conclusion in gold.get("acceptable_conclusions") or ():
        kind = "commits to" if conclusion["kind"] == "committed" else "states the gap"
        sourced = " (the claims that decide it must be sourced)" \
            if conclusion.get("requires_sources") else ""
        lines.append(f"- *{kind}*: {conclusion['statement']}{sourced}")
    facts = gold.get("mandatory_facts") or ()
    lines += ["", "## Facts the answer owes", ""]
    if facts:
        lines += [f"- `{f['id']}`: {f['statement']}"
                  + (f" = {f['value']}" if f.get("value") is not None else "")
                  + (f" {f['unit']}" if f.get("unit") else "")
                  + (f" (source {f['source_ref']})" if f.get("source_ref") else "")
                  for f in facts]
    else:
        lines.append("- none; leave `mandatory_facts_met` empty.")
    required = gold.get("required_limitations") or ()
    optional = gold.get("optional_limitations") or ()
    lines += ["", "## Limitations", ""]
    if gold.get("limitation_scoring") == "not_applicable" or not required:
        lines.append("- not scored for this case.")
    else:
        lines.append(f"- required: {', '.join(required)}")
        if optional:
            lines.append(f"- also acceptable: {', '.join(optional)}")
        lines.append("- a response that names a code but asserts the opposite has not stated it.")
    forbidden = gold.get("forbidden_inferences") or ()
    if forbidden:
        lines += ["", "## Inferences this answer must not make", "",
                  *[f"- `{code}`" for code in forbidden]]
    spans = gold.get("evidence_spans") or ()
    if spans:
        lines += ["", "## Where the facts are", ""]
        for span in spans:
            access = span.get("access") or "full_text_retrieved"
            quote = f" — “{span['quote']}”" if span.get("quote") else ""
            lines.append(f"- {span['evidence_ref']} at `{span['locator']}` ({access}){quote}")
    error = gold.get("predictor_error")
    if error:
        lines += ["", "## The predictor's own output", ""]
        if error.get("actually_wrong"):
            lines += [f"**The served predictor is wrong here** (verified against "
                      f"{error.get('verified_label_source') or 'a recorded source'}). "
                      "A response that repeats its output as a finding without questioning it "
                      "commits `predictor_error_uncaught`.", ""]
        else:
            lines += ["**This is a control case:** the predictor agrees with the verified "
                      "label, so `predictor_error_uncaught` must NOT be raised on it, however "
                      "cautious or incautious the response is.", ""]
        if error.get("note"):
            lines.append(error["note"])
    expectations = gold.get("turn_expectations") or ()
    if expectations:
        lines += ["", "## Across the turns", ""]
        for item in expectations:
            changes = ", ".join(item.get("changes") or ()) or "nothing"
            keeps = ", ".join(item.get("keeps") or ()) or "nothing in particular"
            lines.append(f"- after turn {item['turn'] + 1}: should change {changes}; "
                         f"must keep {keeps}")
    if gold.get("note"):
        lines += ["", f"_{gold['note']}_"]
    return "\n".join(lines) + "\n"


def _stage_label(draft: Mapping[str, Any]) -> str:
    phase = draft.get("phase") or "initial"
    outcome = draft.get("outcome")
    return {
        "initial": "Draft as submitted",
        "hard_correction": "Draft after a hard correction",
        "semantic_repair": "Draft after the semantic repair",
    }.get(str(phase), str(phase)) + (f" ({outcome})" if outcome else "")


def response_markdown(record, response_id: str, store: StudyStore) -> tuple[str, int]:
    """One system's reply: the result the researcher saw per turn, then the
    stages behind it."""
    lines = [f"# Response {response_id}", ""]
    replaced = 0
    multi = len(record.turns) > 1
    trace: dict[str, Any] = {}
    if "trace" in record.artifacts:
        path = store.root / record.artifacts["trace"]
        if path.exists():
            trace = json.loads(path.read_text())
    runs = {int(item.get("turn", index)): item
            for index, item in enumerate(trace.get("runs") or ())}
    for turn in record.turns:
        text, count = scrub(turn.response_text or "(no response was produced)")
        replaced += count
        if multi:
            lines += [f"## Reply to turn {turn.index + 1}", ""]
        lines += [text, ""]
        drafts = (runs.get(turn.index) or {}).get("answer_drafts") or []
        if len(drafts) > 1 or (drafts and turn.meta.get("disposition") != "publish"):
            lines += ["<details><summary>Earlier stages of this reply (not the answer; for "
                      "judging what a repair changed)</summary>", ""]
            for draft in drafts:
                body = (draft.get("payload") or {}).get("answer_markdown") or ""
                scrubbed, count = scrub(body)
                replaced += count
                lines += [f"**{_stage_label(draft)}**", "", scrubbed or "(no prose)", ""]
            lines += ["</details>", ""]
    return "\n".join(lines), replaced


def build_packet_v2(*, study_dir: Path, case_set_dir: Path, packet_id: str, seed: int,
                    systems: list[str] | None = None) -> dict[str, Any]:
    store = StudyStore(study_dir)
    manifest = store.manifest()
    protocol = manifest.get("protocol_v2") or {}
    if not protocol:
        raise SystemExit(f"{study_dir} is not a protocol v2 study")
    records = store.primary_records()
    case_set = load_case_set(case_set_dir, only=sorted(protocol.get("cases") or ()))
    gold_by_case: dict[str, dict[str, Any]] = {}
    for loaded in case_set.cases:
        from evals.investigation.case_inputs import load_case

        with_gold = load_case(loaded.path, with_gold=True)
        if with_gold.gold is None:
            raise SystemExit(f"{loaded.case_id} has no gold file; a packet needs one")
        gold_by_case[loaded.case_id] = with_gold.gold
    chosen = systems or sorted(protocol.get("arms") or {})
    trials = int(manifest.get("trials", 1))
    rubric = json.loads(RUBRIC_PATH.read_text())
    criticals = [c["id"] for c in rubric["critical_errors"]]
    packet_dir = study_dir / "packets" / packet_id
    key_path = study_dir / "keys" / f"{packet_id}.unblinding.json"
    if packet_dir.exists() or key_path.exists():
        raise SystemExit(f"packet {packet_id} already exists; packets are never overwritten")

    rng = random.Random(seed)
    key: dict[str, Any] = {
        "schema_version": KEY_SCHEMA, "packet_id": packet_id,
        "study_id": manifest.get("study_id"), "seed": seed,
        "rubric_version": rubric["schema_version"],
        "case_set": case_set.pins(),
        "gold_status": {cid: gold.get("status") for cid, gold in sorted(gold_by_case.items())},
        "responses": [],
    }
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for loaded in case_set.cases:
        case_id = loaded.case_id
        directory = packet_dir / "cases" / case_id
        (directory / "responses").mkdir(parents=True, exist_ok=True)
        (directory / "case.md").write_text(case_markdown(loaded, loaded.inputs),
                                           encoding="utf-8")
        (directory / "grader_reference.md").write_text(
            reference_markdown(loaded, gold_by_case[case_id]), encoding="utf-8")
        present, missing = [], 0
        for system_id in chosen:
            for trial in range(1, trials + 1):
                record = records.get((case_id, system_id, trial))
                if record is not None and record.status == STATUS_OK:
                    present.append(record)
                else:
                    missing += 1
                    key["responses"].append({
                        "case_id": case_id, "response_id": None, "system_id": system_id,
                        "trial": trial,
                        "missing_reason": record.status if record else "not_run",
                        "error": record.error if record else None,
                        "attempt_id": record.attempt_id if record else None,
                    })
        rng.shuffle(present)
        for number, record in enumerate(present, 1):
            response_id = f"R{number:02d}"
            text, replaced = response_markdown(record, response_id, store)
            (directory / "responses" / f"{response_id}.md").write_text(text, encoding="utf-8")
            dispositions = [t.meta.get("disposition") for t in record.turns]
            key["responses"].append({
                "case_id": case_id, "response_id": response_id,
                "system_id": record.system_id, "trial": record.trial,
                "record_id": record.record_id, "attempt_id": record.attempt_id,
                "identity_replacements": replaced,
                "is_fallback": any(t.meta.get("is_fallback") for t in record.turns),
                "disposition": dispositions[-1] if dispositions else None,
                "dispositions": dispositions,
                "models_answering": sorted({str(record.model.get("model_id_resolved"))}
                                           - {"None"}),
            })
            rows.append({"case_id": case_id, "response_id": response_id})
        counts[case_id] = {"expected": len(chosen) * trials, "included": len(present),
                           "missing": missing}

    columns = (["grader_id", "case_id", "response_id", "task_success", *LIST_COLUMNS]
               + criticals + [d["id"] for d in rubric["dimensions"]] + ["comments"])
    with (packet_dir / "grading_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in columns}, **row})
    (packet_dir / "rubric.json").write_text(json.dumps(rubric, indent=2) + "\n",
                                            encoding="utf-8")
    (packet_dir / "README_FOR_GRADERS.md").write_text(GRADER_README, encoding="utf-8")
    packet_manifest = {
        "schema_version": PACKET_SCHEMA, "packet_id": packet_id, "created_at": now_iso(),
        "study_id": manifest.get("study_id"), "case_set": case_set.pins(),
        "rubric_version": rubric["schema_version"], "responses_per_case": counts,
        "gold_status": sorted({str(g.get("status")) for g in gold_by_case.values()}),
        "blinding": "responses shuffled per case with a recorded seed; product, vendor and "
                    "model names and internal identifiers replaced; style may still identify "
                    "a system",
        "scored_attempt": "the primary attempt of each (case, arm, trial); diagnostic reruns "
                          "are not in the packet",
    }
    (packet_dir / "packet-manifest.json").write_text(
        json.dumps(packet_manifest, indent=2) + "\n", encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return {"packet_dir": str(packet_dir), "key_path": str(key_path), **packet_manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-set", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--systems", default="",
                        help="comma-separated (default: every arm in the study)")
    args = parser.parse_args(argv)
    result = build_packet_v2(
        study_dir=args.root / args.study, case_set_dir=args.case_set, packet_id=args.packet_id,
        seed=args.seed, systems=[s for s in args.systems.split(",") if s] or None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
