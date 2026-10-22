"""The audit view's stage lineage (AG-UI, §14.2): drafts, review rounds, the
repair's effect claim by claim, and what the reader did not get. Built from
what a run recorded; nothing is re-judged."""
from __future__ import annotations

from types import SimpleNamespace

from toxagent.application.investigation.stage_lineage import stage_lineage

RUN = "run_" + "1" * 32
A1 = "ans_" + "a" * 32
A2 = "ans_" + "b" * 32


def _answer(answer_id, claims, limitations=("evidence_scope_limited",)):
    return SimpleNamespace(
        id=answer_id,
        claims=tuple(SimpleNamespace(claim_id=f"clm_{answer_id[-4:]}_{ref}", local_ref=ref,
                                     kind="scientific", text=text)
                     for ref, text in claims.items()),
        limitations=tuple(SimpleNamespace(code=code) for code in limitations),
    )


def _draft(generation, phase, outcome, answer_id=None, claims=(), violations=()):
    return {
        "generation": generation, "phase": phase, "outcome": outcome, "answer_id": answer_id,
        "violations": [{"code": code} for code in violations],
        "payload": {"answer_markdown": " ".join(text for _, text in claims),
                    "claims": [{"local_ref": ref, "kind": "scientific", "text": text}
                               for ref, text in claims],
                    "limitations": [{"code": "evidence_scope_limited"}]},
        "created_at": "2026-10-22T00:00:00+00:00",
    }


FIRST = {"lit": "A study reports blockade in a series.",
         "overreach": "This exact compound blocks hERG."}
REPAIRED = {"lit": "A study reports blockade in a series."}


def _repaired_run():
    answers = {A1: _answer(A1, FIRST), A2: _answer(A2, REPAIRED, limitations=())}
    drafts = [
        _draft(1, "initial", "rejected", claims=tuple(FIRST.items()),
               violations=("claim_citation_missing",)),
        _draft(2, "hard_correction", "accepted", A1, tuple(FIRST.items())),
        _draft(3, "semantic_repair", "accepted", A2, tuple(REPAIRED.items())),
    ]
    result = {
        "release_policy": "strict-reviewed-v1", "disposition": "publish",
        "review_status": "repaired_and_passed", "execution_outcome": "completed",
        "public_answer_id": A2, "candidate_answer_id": A2,
        "detail": {
            "reviews": [
                {"status": "completed", "answer_id": A1,
                 "counts": {"supported": 1, "not_supported": 1},
                 "blocking_claim_ids": [f"clm_{A1[-4:]}_overreach"],
                 "verdicts": [
                     {"claim_id": f"clm_{A1[-4:]}_lit", "verdict": "supported", "reason": "ok"},
                     {"claim_id": f"clm_{A1[-4:]}_overreach", "verdict": "not_supported",
                      "reason": "the source tested a series"}]},
                {"status": "completed", "answer_id": A2, "counts": {"supported": 1},
                 "blocking_claim_ids": [],
                 "verdicts": [{"claim_id": f"clm_{A2[-4:]}_lit", "verdict": "supported",
                               "reason": "ok"}]},
            ],
            "repair": {"removed_claim_ids": ["overreach"], "removal_only": True,
                       "lost_limitation_codes": ["evidence_scope_limited"]},
            "needed_inputs": [], "unresolved_claim_ids": [], "confirmed_facts": [],
        },
    }
    return stage_lineage(run_id=RUN, drafts=drafts, answers=answers, result=result)


def test_every_stage_is_listed_in_order_with_the_public_one_marked():
    lineage = _repaired_run()
    assert [(s["phase"], s["outcome"], s["hard_valid"], s["is_public"])
            for s in lineage["stages"]] == [
        ("initial", "rejected", False, False),
        ("hard_correction", "accepted", True, False),
        ("semantic_repair", "accepted", True, True),
    ]
    # A draft the validator refused keeps what it said and why it was refused.
    raw = lineage["stages"][0]
    assert raw["violation_codes"] == ["claim_citation_missing"]
    assert raw["claim_count"] == 2 and "exact compound" in raw["answer_markdown"]


def test_verdicts_are_read_against_the_claim_they_judged():
    lineage = _repaired_run()
    first, second = lineage["reviews"]
    assert (first["phase"], second["phase"]) == ("review_initial", "review_final")
    blocked = [v for v in first["verdicts"] if v["verdict"] == "not_supported"]
    assert blocked == [{"claim_id": f"clm_{A1[-4:]}_overreach", "key": "overreach",
                        "text": FIRST["overreach"], "verdict": "not_supported",
                        "reason": "the source tested a series"}]


def test_a_repair_that_only_removed_is_shown_claim_by_claim():
    lineage = _repaired_run()
    claims = {c["key"]: c for c in lineage["claims"]}
    assert claims["overreach"]["change"] == "removed"
    assert claims["overreach"]["final_text"] is None
    assert claims["overreach"]["verdicts"] == [
        {"round": 1, "verdict": "not_supported", "reason": "the source tested a series"}]
    assert claims["lit"]["change"] == "kept"
    assert [v["round"] for v in claims["lit"]["verdicts"]] == [1, 2]
    assert lineage["repair"]["removal_only"] is True
    assert lineage["missing"]["lost_limitation_codes"] == ["evidence_scope_limited"]


def test_a_withheld_run_has_no_public_stage_and_says_what_is_missing():
    answers = {A1: _answer(A1, FIRST)}
    lineage = stage_lineage(
        run_id=RUN, drafts=[_draft(1, "initial", "accepted", A1, tuple(FIRST.items()))],
        answers=answers,
        result={"release_policy": "strict-reviewed-v1", "disposition": "withhold",
                "review_status": "not_completed", "execution_outcome": "cancelled",
                "public_answer_id": None, "candidate_answer_id": A1,
                "detail": {"reason": "the run ended before an answer could be published",
                           "failure_code": "cancelled"}},
    )
    assert not any(s["is_public"] for s in lineage["stages"])
    assert lineage["reviews"] == []
    assert {c["change"] for c in lineage["claims"]} == {"kept"}
    assert lineage["missing"]["failure_code"] == "cancelled"
    assert lineage["execution_outcome"] == "cancelled"


def test_a_run_before_any_release_policy_is_unknown_not_passed():
    lineage = stage_lineage(run_id=RUN, drafts=[], answers={}, result=None)
    assert lineage["recorded"] is False
    assert lineage["release_policy"] == "legacy"
    assert lineage["review_status"] is None
    assert lineage["stages"] == [] and lineage["claims"] == []
