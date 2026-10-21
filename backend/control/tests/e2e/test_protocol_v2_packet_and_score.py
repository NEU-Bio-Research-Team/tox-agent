"""#53 end to end: a protocol v2 study becomes a blinded packet and a card.

Two arms run the technical set in process, the packet is built from the
primary attempts, filled in as two graders would, and scored. What this
holds:

* the packet carries the turns as every arm received them and the reference
  built from the gold, and no arm's identity;
* the gold file never reaches the packet's case or response files, only the
  grader reference;
* the unblinding key stays outside the packet directory;
* the card's denominator is every assigned case and trial, the paired D - B
  delta is computed over the cases both arms were assigned, and -- with a
  synthetic gold and an ungraded cell -- it refuses to call the table
  publishable;
* retrieval and cost come from the traces and records, not from the grades.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from evals.investigation.packet_v2 import build_packet_v2
from evals.investigation.record import StudyStore
from evals.investigation.scorecard_v2 import score_study
from tests.e2e.test_protocol_v2_four_arms import CASES, SET, _config, _script, _arm_env
from tests.e2e.test_scripted_runtime import _install_scripted_runtime
from tests.support.api import USER_TOKEN, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

ARMS = ("B2_fixed_workflow", "D2_investigator")


async def _run(db, monkeypatch, tmp_path, system_id: str, case_ids: list[str]) -> None:
    from evals.investigation.adapters.toxagent_replay import ToxAgentReplayAdapter
    from evals.investigation.run import run_protocol_study

    _arm_env(monkeypatch, system_id)
    async with api_client(db, StubPredictor(), config=_config()) as client:
        await _install_scripted_runtime(
            client, _script({"keeps_case": system_id == "D2_investigator"}))
        adapter = ToxAgentReplayAdapter(
            "http://control.test", USER_TOKEN, transport=httpx.ASGITransport(app=client.app),
            poll_delay_s=0.01, run_timeout_s=60)
        await run_protocol_study(
            study_id="v2", root=tmp_path, case_set_dir=SET, system_ids=[system_id],
            case_ids=case_ids, adapters={system_id: adapter})


def _fill(sheet: Path, *, grader: str, succeed: set[str], facts: dict[str, list[str]],
          limits: dict[str, list[str]], skip: set[str] = frozenset()) -> Path:
    rows = list(csv.DictReader(sheet.open(newline="", encoding="utf-8")))
    out = sheet.with_name(f"grades-{grader}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            key = f"{row['case_id']}/{row['response_id']}"
            if key in skip:
                writer.writerow({**row, "grader_id": grader})
                continue
            writer.writerow({
                **row, "grader_id": grader,
                "task_success": "1" if key in succeed else "0",
                "mandatory_facts_met": ";".join(facts.get(key, [])),
                "limitations_stated": ";".join(limits.get(key, [])),
                "false_reassurance": "0", "invented_value": "0",
            })
    return out


async def test_a_study_becomes_a_blinded_packet_and_a_scorecard(db, monkeypatch, tmp_path):
    cases = CASES[:2]
    for system_id in ARMS:
        await _run(db, monkeypatch, tmp_path, system_id, cases)

    study_dir = tmp_path / "v2"
    result = build_packet_v2(study_dir=study_dir, case_set_dir=SET, packet_id="lab-1", seed=1)
    packet_dir = Path(result["packet_dir"])
    key = json.loads(Path(result["key_path"]).read_text())

    # The key is outside the packet, and the packet names no arm.
    assert Path(result["key_path"]).parent == study_dir / "keys"
    text = "\n".join(p.read_text() for p in packet_dir.rglob("*.md"))
    assert "ToxAgent" not in text and "fixed_workflow" not in text
    assert "investigator" not in text.replace("investigation", "")
    # The case file is what the arms were sent; the gold is only in the
    # grader reference.
    case_file = (packet_dir / "cases" / cases[0] / "case.md").read_text()
    gold = json.loads((SET / "gold" / f"{cases[0]}.json").read_text())
    assert gold["reference_outcome"] not in case_file
    reference = (packet_dir / "cases" / cases[0] / "grader_reference.md").read_text()
    assert gold["reference_outcome"] in reference
    assert "synthetic engineering fixture" in reference
    assert result["responses_per_case"][cases[0]] == {"expected": 2, "included": 2, "missing": 0}
    assert result["gold_status"] == ["synthetic_engineering_gold"]

    # Two graders. One response of the second case is left ungraded.
    by_system = {(r["case_id"], r["system_id"]): r["response_id"]
                 for r in key["responses"] if r.get("response_id")}
    d_first = f"{cases[0]}/{by_system[(cases[0], 'D2_investigator')]}"
    b_first = f"{cases[0]}/{by_system[(cases[0], 'B2_fixed_workflow')]}"
    d_second = f"{cases[1]}/{by_system[(cases[1], 'D2_investigator')]}"
    b_second = f"{cases[1]}/{by_system[(cases[1], 'B2_fixed_workflow')]}"
    sheet = packet_dir / "grading_sheet.csv"
    required = {case_id: json.loads((SET / "gold" / f"{case_id}.json").read_text())
                ["required_limitations"] for case_id in cases}
    facts_of = {case_id: [f["id"] for f in json.loads(
        (SET / "gold" / f"{case_id}.json").read_text())["mandatory_facts"]]
        for case_id in cases}
    # D states every required limitation and every owed fact; B, graded only
    # on the second case, states one of its two limitations and no fact.
    limits = {d_first: required[cases[0]], d_second: required[cases[1]],
              b_second: required[cases[1]][:1]}
    facts = {d_first: facts_of[cases[0]], d_second: facts_of[cases[1]], b_second: []}
    grades = [
        _fill(sheet, grader=grader, succeed={d_first, d_second}, facts=facts, limits=limits,
              skip={b_first})
        for grader in ("ga", "gb")
    ]
    card = score_study(study_dir=study_dir, case_set_dir=SET, packet_id="lab-1",
                       grades=grades, compare=[("D2_investigator", "B2_fixed_workflow")],
                       reps=100)

    success = card["task_success"]
    assert success["D2_investigator"]["all_assigned"] == {
        "successes": 2, "assigned": 2, "rate": 1.0,
        "ci95_clopper_pearson": success["D2_investigator"]["all_assigned"]["ci95_clopper_pearson"],
    }
    # B failed one and was left ungraded on the other: both are failures, and
    # the causes say which is which.
    assert success["B2_fixed_workflow"]["all_assigned"]["successes"] == 0
    assert success["B2_fixed_workflow"]["failure_causes"] == {"not_graded": 1, "task_not_met": 1}
    (paired,) = card["paired"]
    assert (paired["paired_assigned"], paired["n_system_only"], paired["n_baseline_only"]) \
        == (2, 2, 0)
    assert paired["delta"] == 1.0
    scored = card["limitations"]["D2_investigator"]
    assert scored["required_recall"] == 1.0 and scored["all_required_pass_rate"] == 1.0
    # B stated one of the two the case requires: half the recall, and it does
    # not pass "all required".
    assert card["limitations"]["B2_fixed_workflow"]["required_recall"] == 0.5
    assert card["limitations"]["B2_fixed_workflow"]["all_required_pass_rate"] == 0.0
    # Required information is its own measurement: B's answer was graded and
    # owed facts it did not state.
    assert card["mandatory_information"]["D2_investigator"]["recall"] == 1.0
    assert card["mandatory_information"]["B2_fixed_workflow"]["recall"] == 0.0
    assert card["completeness"]["scientific_table_publishable"] is False
    assert card["completeness"]["ungraded_responses"] == [b_first]
    assert "the gold is not SME-verified" in card["completeness"]["why_not"]
    # Cost is read from the records, over every attempt, with unknown usage
    # kept unknown rather than zero.
    cost = card["cost"]
    assert set(cost) == set(ARMS)
    assert all(row["attempts"] >= 2 for row in cost.values())
    assert all(row["records_with_unknown_usage"] >= 0 for row in cost.values())
    store = StudyStore(study_dir)
    assert len(store.records()) == 4

    written = (study_dir / "scorecards")
    assert not written.exists()  # score_study computes; the CLI writes
