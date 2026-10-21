"""Synthetic frozen activity screenings, for replay (#44, #46).

Built through the domain and frozen with ``freeze_screening`` exactly as a
live run would freeze them, so a replay exercises the records the product
would actually hold -- not hand-written JSON that only looks right. Every
number here is synthetic: the model id is the SAIR B0 baseline's name, but no
model produced these estimates.

    python -m evals.investigation.activity_fixtures --write evals/fixtures/activity/activity-replay-synthetic.json

prints the sha256 a deployment is pinned to (TOXAGENT_ACTIVITY_REPLAY_SHA256).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from toxagent.application.activity.projections import freeze_screening
from toxagent.application.activity.replay import write_replay_file
from toxagent.domain.ids import RUN, SESSION, new_id
from toxagent.domain.screening import (
    ActivityApplicability,
    ActivityEstimate,
    ActivityGuidedScreening,
    ActivityModelProvenance,
    ActivitySnapshot,
    ActivitySubject,
    TargetIdentity,
    TargetPriorityPolicy,
    TargetRole,
)

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)
LIGAND = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
CONTEXT = "IC50|wild_type"


def frozen_screening(
    *,
    target_id: str = "EGFR",
    uniprot_id: str = "P00533",
    predicted: float = 7.1,
    interval: tuple[float, float] = (6.8, 7.4),
    threshold: float = 7.0,
    activity_fails: bool = False,
    close_without_toxicity: bool = True,
) -> dict:
    """One screening taken as far as its policy decision -- by default a
    REVIEW, because the interval crosses the threshold -- and, when
    ``close_without_toxicity``, closed with the toxicity stage not run."""
    session_id, run_id = new_id(SESSION), new_id(RUN)
    target = TargetIdentity(target_id=target_id, uniprot_id=uniprot_id)
    screening = ActivityGuidedScreening.create(
        session_id=session_id, run_id=run_id, input_smiles=LIGAND,
        subject=ActivitySubject(canonical_smiles=LIGAND, target=target, assay_context=CONTEXT),
        intended_target_role=TargetRole.INTENDED_TARGET, now=NOW,
    ).start_activity(now=NOW)
    if activity_fails:
        screening = screening.fail_activity(failure_code="activity_model_unavailable", now=NOW)
        return freeze_screening(
            screening=screening, snapshot=None, decision=None, model_admitted=False,
            run_id=run_id, now=NOW,
        )
    snapshot = ActivitySnapshot.create(
        session_id=session_id, run_id=run_id, input_smiles=LIGAND, canonical_smiles=LIGAND,
        target=target, assay_context=CONTEXT,
        estimate=ActivityEstimate(
            predicted_pic50=predicted, uncertainty=0.2,
            interval_low=interval[0], interval_high=interval[1],
        ),
        ligand_applicability=ActivityApplicability.IN_DOMAIN,
        target_applicability=ActivityApplicability.IN_DOMAIN,
        joint_applicability=ActivityApplicability.IN_DOMAIN,
        provenance=ActivityModelProvenance(
            model_id="sair-descriptor-hgb-b0", dataset_id="sair-phase1",
            split_id="exact_sequence_cold_target",
        ),
        now=NOW,
    )
    screening = screening.record_activity(snapshot, now=NOW)
    policy = TargetPriorityPolicy(
        policy_id=f"activity-priority-{target_id.lower()}-v1", target_id=target_id,
        assay_context=CONTEXT, threshold_pic50=threshold,
        threshold_source="synthetic_test_policy", max_uncertainty=0.5,
    )
    decision = policy.decide(snapshot, now=NOW)
    screening = screening.record_decision(decision, now=NOW)
    if close_without_toxicity:
        screening = screening.complete_without_toxicity(
            reason="review_required_awaiting_researcher", now=NOW,
        )
    return freeze_screening(
        screening=screening, snapshot=snapshot, decision=decision, model_admitted=False,
        run_id=run_id, now=NOW,
    )


def screening_id(record: dict) -> str:
    return record["screening_state"]["screening_id"]


def replay_file(tmp_path: Path, *records: dict) -> tuple[Path, str]:
    path = tmp_path / "screenings.json"
    return path, write_replay_file(records, path)


def synthetic_set() -> list[dict]:
    """The three shapes the #46 user stories need: a REVIEW whose interval
    crosses the threshold with toxicity not run, the same ligand on a second
    target estimated well below its threshold (low priority), and a screening
    whose activity stage failed. Every one is screened as an intended target:
    the policy is defined for that role only and refuses the others."""
    return [
        frozen_screening(),
        frozen_screening(target_id="KCNH2", uniprot_id="Q12809", predicted=5.2,
                         interval=(4.9, 5.5)),
        frozen_screening(target_id="ABL1", uniprot_id="P00519", activity_fails=True),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    records = synthetic_set()
    digest = write_replay_file(records, args.write)
    for record in records:
        state = record["screening_state"]
        print(state["screening_id"], state["subject"]["target"]["target_id"], state["status"])
    print("sha256", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
