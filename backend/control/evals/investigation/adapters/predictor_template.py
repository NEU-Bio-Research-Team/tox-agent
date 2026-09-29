"""System A: the predictor's output through a fixed template, no language model.

The floor every other arm has to clear (RETHINK §5.4): the same snapshot the
``with_snapshot`` platform arms receive, rendered the same way whatever the
question was. It states what the predictor measured and the caveats the
product states for every score; it cannot frame a question, weigh evidence,
or propose anything.
"""
from __future__ import annotations

from typing import Any

from evals.investigation.adapters import AdapterResult
from evals.investigation.record import STATUS_ERROR, STATUS_OK, StudyStore, TurnRecord, now_iso
from evals.investigation.systems import SystemSpec

TEMPLATE_VERSION = "predictor-template-v1"


def render(snapshot: dict[str, Any]) -> str:
    predictions = snapshot.get("predictions") or {}
    lines = ["Toxicity predictor output for the submitted structure."]
    herg = predictions.get("herg")
    if herg:
        lines.append(
            f"- hERG channel blockade: model probability {herg['probability_blocker']:.3f}, "
            f"label '{herg['label']}' at threshold {herg['threshold']:.3f} "
            f"({herg['threshold_source']}), model {herg['model_id']}."
        )
    tox21 = predictions.get("tox21")
    if tox21:
        lines.append(f"- Tox21 (twelve independent assays, model {tox21['model_id']}):")
        for task, assay in sorted((tox21.get("assays") or {}).items()):
            lines.append(
                f"  - {task}: probability {assay['probability_activity']:.3f}, "
                f"{'active' if assay['active'] else 'inactive'} at threshold {assay['threshold']:.3f}"
            )
    for endpoint in snapshot.get("unavailable_endpoints") or ():
        lines.append(f"- {endpoint}: not available from this predictor deployment.")
    applicability = snapshot.get("applicability") or {}
    if applicability:
        reasons = ", ".join(applicability.get("reasons") or ()) or "none"
        lines.append(
            f"- Applicability (rule-based element check): {applicability.get('status')} "
            f"(reasons: {reasons})."
        )
    lines += [
        "",
        "Probabilities are uncalibrated model scores, not measured activity or clinical risk. "
        "Endpoints are separate measurements and are not combined. This output is a screening "
        "signal and not a safety assessment.",
    ]
    return "\n".join(lines)


class PredictorTemplateAdapter:
    def describe(self) -> dict[str, Any]:
        return {"adapter": "predictor_template", "template_version": TEMPLATE_VERSION}

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore) -> AdapterResult:
        if not snapshot:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="no predictor snapshot for this case")
        text = render(snapshot)
        stamp = now_iso()
        turns = [
            TurnRecord(index=i, user_text=t["text"], sent_text="(template: the question is not read)",
                       response_text=text, started_at=stamp, ended_at=stamp, duration_s=0.0)
            for i, t in enumerate(case["turns"])
        ]
        artifact = store.write_raw(spec.system_id, case["case_id"], trial, "snapshot.json", snapshot)
        return AdapterResult(status=STATUS_OK, turns=turns, final_text=text,
                             model={"model_id_resolved": None, "template_version": TEMPLATE_VERSION},
                             artifacts={"snapshot": artifact})
