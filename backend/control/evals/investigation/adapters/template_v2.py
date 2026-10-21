"""Arm A under protocol v2: the frozen artifacts through a fixed template (#50).

The product floor every other arm has to clear. It reads the case's frozen
inputs -- the same screening records and toxicity analyses the other arms are
given, from the same pinned replay file -- and renders, whatever was asked:
the activity estimate and its interval, the policy decision and its reason,
each stage's status, and the toxicity predictor's output, with the caveats the
product states for each. It never calls a predictor, never searches, never
computes anything the artifacts do not hold, and follows a turn's
``switch_subject`` because the subject is an input, not a judgement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from evals.investigation import turn_actions
from evals.investigation.adapters import AdapterResult
from evals.investigation.adapters.predictor_template import render as render_prediction
from evals.investigation.record import STATUS_ERROR, STATUS_OK, StudyStore, TurnRecord, now_iso
from evals.investigation.systems import SystemSpec

TEMPLATE_VERSION = "predictor-template-v2"


def render_screening(record: Mapping[str, Any]) -> str:
    state = record.get("screening_state") or {}
    subject = state.get("subject") or {}
    target = subject.get("target") or {}
    lines = [
        f"Activity-guided screening of {subject.get('canonical_smiles')} on "
        f"{target.get('target_id')} ({subject.get('assay_context')}).",
        f"- Screening status: {state.get('status')}.",
    ]
    estimate = record.get("activity_estimate")
    if estimate is None:
        failure = state.get("activity_failure_code")
        lines.append(
            "- Activity estimate: none -- the activity stage "
            + (f"failed ({failure})" if failure else "did not produce one")
            + ". This is not an inactive or negative result."
        )
    else:
        value = estimate.get("estimate") or {}
        interval = value.get("prediction_interval")
        lines.append(
            f"- Predicted pIC50 {value.get('predicted_pic50')}"
            + (f", {int((value.get('interval_level') or 0) * 100)}% interval "
               f"{interval[0]}-{interval[1]}" if interval else "")
            + f" (model {(estimate.get('model') or {}).get('model_id')}, "
            + ("admitted" if (estimate.get("model") or {}).get("admitted") else "not admitted")
            + ")."
        )
        applicability = estimate.get("applicability") or {}
        lines.append(f"- Applicability: ligand {applicability.get('ligand')}, target "
                     f"{applicability.get('target')}, joint {applicability.get('joint')}.")
    decision = record.get("prioritisation_decision")
    if decision is not None:
        policy = decision.get("policy") or {}
        lines.append(
            f"- Policy decision: {decision.get('status')} "
            f"({', '.join(decision.get('reason_codes') or ()) or 'no reason codes'}), threshold "
            f"pIC50 {policy.get('threshold_pic50')} ({policy.get('threshold_source')})."
        )
    toxicity = state.get("toxicity") or {}
    if toxicity.get("run"):
        lines.append("- Toxicity stage: completed.")
    elif toxicity.get("failure_code"):
        lines.append(f"- Toxicity stage: failed ({toxicity['failure_code']}).")
    else:
        lines.append(
            "- Toxicity stage: not run"
            + (f" ({toxicity['not_run_reason']})" if toxicity.get("not_run_reason") else "")
            + ". Not running it is not a negative toxicity result."
        )
    frozen = record.get("toxicity_snapshot")
    if frozen is not None:
        lines += ["", render_prediction(dict(frozen.get("predictor_response") or {}))]
    lines += [
        "",
        "A pIC50 is an in-vitro potency estimate for this target and context only, not "
        "efficacy. The interval is marginal coverage on validation residuals, not a "
        "per-compound guarantee. The policy decision is the output of a fixed rule, not a "
        "measurement.",
    ]
    return "\n".join(lines)


class TemplateV2Adapter:
    """A, reading the pinned replay file the deployments replay from."""

    def __init__(self, replay_path: Path, *, sha256: str) -> None:
        """``sha256`` is the case set's pin: A reads exactly the file the
        deployments replay from, or nothing."""
        from toxagent.application.activity.replay import FrozenReplayStore

        self._store = FrozenReplayStore.load(Path(replay_path), sha256=sha256)
        self.replay_sha256 = self._store.sha256

    def describe(self) -> dict[str, Any]:
        return {"adapter": "template_v2", "template_version": TEMPLATE_VERSION,
                "replay_sha256": self.replay_sha256}

    def _render(self, subject_ref: Mapping[str, Any]) -> tuple[str, str | None]:
        """The rendered artifact and its content hash, or ``("", why)``."""
        if subject_ref.get("kind") == "activity_screening":
            record = self._store.get(str(subject_ref.get("screening_id")))
            if record is None:
                return "", "the screening is not in the pinned replay file"
            return render_screening(record), str(record.get("content_sha256"))
        frozen = self._store.get_analysis(str(subject_ref.get("analysis_ref")))
        if frozen is None:
            return "", "the analysis is not in the pinned replay file"
        return render_prediction(dict(frozen.get("predictor_response") or {})), str(
            frozen.get("content_sha256"))

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore,
                  inputs: dict[str, Any] | None = None,
                  attempt_id: str | None = None) -> AdapterResult:
        del snapshot
        if inputs is None:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="template v2 reads a protocol v2 case's inputs")
        declared = turn_actions.declared_digests(inputs)
        subject = dict((inputs.get("subject") or {}).get("subject_ref") or {})
        turns: list[TurnRecord] = []
        observed: dict[str, str] = {}
        mismatches: list[str] = []
        for index, turn in enumerate(inputs.get("turns") or ()):
            subject = turn_actions.subject_for_turn(turn, subject)
            key = turn_actions.subject_key(subject)
            text, digest_or_error = self._render(subject)
            if not text:
                return AdapterResult(status=STATUS_ERROR, turns=turns, error=digest_or_error)
            observed[key] = str(digest_or_error)
            if key in declared and declared[key] != observed[key]:
                mismatches.append(f"turn {index}: {key} is {observed[key]!r} in the replay "
                                  f"file, the inputs declare {declared[key]!r}")
            stamp = now_iso()
            turns.append(TurnRecord(
                index=index, user_text=turn["text"],
                sent_text="(template: the question is not read)", response_text=text,
                started_at=stamp, ended_at=stamp, duration_s=0.0,
                meta={"subject": key},
            ))
        artifact = store.write_raw(spec.system_id, case["case_id"], trial, "template.json",
                                   {"observed_source_sha256": observed,
                                    "template_version": TEMPLATE_VERSION,
                                    "replay_sha256": self.replay_sha256},
                                   attempt_id=attempt_id)
        return AdapterResult(
            status=STATUS_ERROR if mismatches else STATUS_OK, turns=turns,
            final_text=turns[-1].response_text if turns else "",
            model={"model_id_resolved": None, "template_version": TEMPLATE_VERSION},
            artifacts={"template": artifact},
            toxagent={"observed_source_sha256": dict(sorted(observed.items()))},
            error=("artifact_digest_mismatch: " + "; ".join(mismatches)) if mismatches else None,
        )
