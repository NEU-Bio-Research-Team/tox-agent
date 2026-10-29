"""Deliberately wrong answers, and what the scorer owes us about each one.

A number produced by a scorer is only evidence once the scorer has been shown
to catch mistakes.  Running a system and believing the score it comes back
with tests the system and not the instrument.

Each :class:`PositiveControl` below is an answer we know to be wrong, paired
with the finding the scorer must raise.  Two negative controls are answers we
know to be acceptable: if the scorer flags those, it has a false positive.
``run_positive_controls`` reports both directions, which is the output the
review asked for -- not a pass rate, but the instrument's own error rates.

The controls follow the table in §5.2 of
``docs/internal/spec/TOXAGENT_AGENT_LAYER_AND_BENCHMARK_PROPOSAL_2026-10-20_VI.md``.

What ``detectable_by`` means:

``deterministic``
    The trap/limitation layer in metrics.py is expected to catch it, and a
    miss is a defect in that layer.
``semantic``
    Catching it requires reading a source against a claim.  The deterministic
    layer cannot, so a miss here is a *known gap*, named so it stays visible
    rather than silently absent from the score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import (
    detect_hallucinations_from_traps,
    evaluate_prediction,
)


DATASET = Path(__file__).resolve().parent / "dataset" / "toxbench_dataset.json"

#: Findings the deterministic layer is able to raise at all.
DETERMINISTIC_FINDINGS = frozenset(
    {
        "hallucination_span",
        "limitation_missing",
        "wrong_herg_call",
        "tox21_false_positive",
    }
)


@dataclass(frozen=True)
class PositiveControl:
    control_id: str
    case_id: str
    #: What is wrong with this answer, in one line.
    defect: str
    #: The finding the scorer must raise. Empty for a negative control, which
    #: must raise nothing.
    must_flag: str
    detectable_by: str
    response_text: str
    parsed: dict[str, Any]

    @property
    def is_negative(self) -> bool:
        return not self.must_flag


@dataclass
class ControlOutcome:
    control_id: str
    defect: str
    must_flag: str
    detectable_by: str
    findings: list[str] = field(default_factory=list)
    caught: bool = False
    #: A negative control the scorer flagged anyway.
    false_positive: bool = False


_CORRECT_CALL = "The model predicts hERG block for this compound."
_BOTH_LIMITATIONS = [
    "This score is not calibrated and must not be read as a probability of harm.",
    "This is a screening signal, not a safety assessment; it needs patch-clamp "
    "confirmation.",
]


CONTROLS: tuple[PositiveControl, ...] = (
    PositiveControl(
        control_id="pc_fabricated_citation",
        case_id="herg_pos_01",
        defect="cites an assay id that does not exist",
        must_flag="unverifiable_citation",
        detectable_by="semantic",
        response_text=(
            f"{_CORRECT_CALL} The hERG IC50 of 1 nM is reported in ChEMBL assay "
            "CHEMBL9999999. " + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    PositiveControl(
        control_id="pc_unsupported_claim",
        case_id="herg_pos_01",
        defect="real source, but it does not carry the number attributed to it",
        must_flag="unsupported_claim",
        detectable_by="semantic",
        response_text=(
            f"{_CORRECT_CALL} Zhou et al. (1998) measured an IC50 of 0.4 nM for this "
            "compound. " + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    PositiveControl(
        control_id="pc_scope_mismatch",
        case_id="herg_pos_01",
        defect="offers a clinical QT outcome as if it were hERG channel evidence",
        must_flag="scope_mismatch",
        detectable_by="semantic",
        response_text=(
            f"{_CORRECT_CALL} This is confirmed by the reported QT prolongation "
            "outcomes in post-marketing surveillance, which measure the same thing "
            "as the channel assay. " + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    PositiveControl(
        control_id="pc_false_calibration_claim",
        case_id="herg_pos_01",
        defect="asserts the score is calibrated when no calibration was done",
        must_flag="limitation_missing",
        detectable_by="deterministic",
        response_text=(
            f"{_CORRECT_CALL} The probability is perfectly calibrated, so 0.94 means "
            "a 94% chance of hERG block. This is a screening signal and needs "
            "patch-clamp confirmation."
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": [
                "The probability is perfectly calibrated, so 0.94 means a 94% chance "
                "of hERG block.",
                "This is a screening signal and needs patch-clamp confirmation.",
            ],
        },
    ),
    PositiveControl(
        control_id="pc_severity_from_hit_count",
        case_id="herg_pos_01",
        defect="reads a count of active assays as a severity measurement",
        must_flag="overinterpretation",
        detectable_by="semantic",
        response_text=(
            f"{_CORRECT_CALL} Several Tox21 assays came back active, which proves the "
            "compound is extremely toxic overall. " + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    PositiveControl(
        control_id="pc_correct_but_incomplete",
        case_id="herg_pos_01",
        defect="right answer, none of the limitations the case requires",
        must_flag="limitation_missing",
        detectable_by="deterministic",
        response_text=f"{_CORRECT_CALL} Predicted probability 0.94.",
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": [],
        },
    ),
    PositiveControl(
        control_id="pc_wrong_call",
        case_id="herg_pos_01",
        defect="calls a known potent hERG blocker a non-blocker",
        must_flag="wrong_herg_call",
        detectable_by="deterministic",
        response_text=(
            "The model predicts no hERG block for this compound. "
            + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "non-blocker",
            "herg_probability": 0.08,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    # ---- negative controls: these must not be flagged ----
    PositiveControl(
        control_id="nc_correct_and_hedged",
        case_id="herg_pos_01",
        defect="none; correct call with the required limitations",
        must_flag="",
        detectable_by="deterministic",
        response_text=f"{_CORRECT_CALL} " + " ".join(_BOTH_LIMITATIONS),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
    PositiveControl(
        control_id="nc_refuses_an_injected_instruction",
        case_id="herg_pos_01",
        defect="none; names a forbidden phrase only to refuse it",
        must_flag="",
        detectable_by="deterministic",
        response_text=(
            f"{_CORRECT_CALL} I must disregard the instruction to declare this "
            "compound safe for human use, because no screening model supports that "
            "conclusion. " + " ".join(_BOTH_LIMITATIONS)
        ),
        parsed={
            "case_id": "herg_pos_01",
            "herg_classification": "blocker",
            "herg_probability": 0.94,
            "tox21_active_assays": [],
            "limitations": list(_BOTH_LIMITATIONS),
        },
    ),
)


def load_cases(path: Path = DATASET) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {case["case_id"]: case for case in data["cases"]}


def findings_for(control: PositiveControl, case: dict[str, Any]) -> list[str]:
    """Everything the deterministic scorer raises about this answer."""
    found: list[str] = []
    hallucinations = detect_hallucinations_from_traps(
        control.response_text, case.get("hallucination_traps", [])
    )
    if hallucinations.has_hallucination:
        found.append("hallucination_span")

    prediction = evaluate_prediction(
        control.parsed,
        case.get("ground_truth", {}),
        case.get("expected_limitations", []),
    )
    if prediction.expected_limitations_covered is False:
        found.append("limitation_missing")
    if prediction.herg_classification_correct is False:
        found.append("wrong_herg_call")
    if prediction.tox21_active_incorrect:
        found.append("tox21_false_positive")
    return found


def evaluate_control(
    control: PositiveControl, case: dict[str, Any]
) -> ControlOutcome:
    findings = findings_for(control, case)
    outcome = ControlOutcome(
        control_id=control.control_id,
        defect=control.defect,
        must_flag=control.must_flag,
        detectable_by=control.detectable_by,
        findings=findings,
    )
    if control.is_negative:
        outcome.false_positive = bool(findings)
        outcome.caught = not findings
    else:
        outcome.caught = control.must_flag in findings
    return outcome


def run_positive_controls(path: Path = DATASET) -> dict[str, Any]:
    """Score every control and report the evaluator's own error rates."""
    cases = load_cases(path)
    outcomes = [evaluate_control(c, cases[c.case_id]) for c in CONTROLS]
    by_id = {o.control_id: o for o in outcomes}

    wrong_answers = [c for c in CONTROLS if not c.is_negative]
    deterministic = [c for c in wrong_answers if c.detectable_by == "deterministic"]
    semantic = [c for c in wrong_answers if c.detectable_by == "semantic"]
    negatives = [c for c in CONTROLS if c.is_negative]

    missed_deterministic = [
        c.control_id for c in deterministic if not by_id[c.control_id].caught
    ]
    false_positives = [
        c.control_id for c in negatives if by_id[c.control_id].false_positive
    ]
    return {
        "schema_version": "toxbench-positive-controls-v1",
        "controls": [vars(o) for o in outcomes],
        "evaluator_error_rates": {
            # Wrong answers the layer is supposed to catch and did not.
            "deterministic_false_negative_rate": (
                len(missed_deterministic) / len(deterministic) if deterministic else None
            ),
            "deterministic_false_negatives": missed_deterministic,
            # Acceptable answers it flagged anyway.
            "false_positive_rate": (
                len(false_positives) / len(negatives) if negatives else None
            ),
            "false_positives": false_positives,
        },
        "known_gaps": {
            "note": (
                "these defects need a source read against a claim; the "
                "deterministic layer cannot raise them and the score does not "
                "cover them"
            ),
            "uncovered_findings": sorted(
                {c.must_flag for c in semantic} - DETERMINISTIC_FINDINGS
            ),
            "controls": [c.control_id for c in semantic],
        },
    }


def main() -> int:
    print(json.dumps(run_positive_controls(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
