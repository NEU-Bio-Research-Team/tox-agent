"""What the served explainer has been measured to do, stated with every attribution.

An attribution heatmap reads as an explanation whether or not it is one. The
predictor's XAI benchmark (``backend/predictor/evals/xai``) measured the served
method on a 42-molecule golden panel: it is deterministic and invariant to how
a SMILES is spelled, and on the two measured targets it was **not** measurably
more faithful than deleting arbitrary atoms. The RETHINK review (§2, §4.3)
requires the agent to publish the reliability of each explanation layer rather
than leave a reader to infer it from a picture.

This module is that statement, as data. It is copied from the predictor's
result files rather than read from them at run time because the control plane
talks to the predictor only over ``/v1`` (ADR 0001);
``tests/unit/test_explainer_validation.py`` fails when the copy and the result
files disagree, so a new benchmark run is a reviewed change here.

The model view is qualitative on purpose. The counts are in the UI view and the
docs; a number in a model view that is not in any observation's canonical
payload is a number the model cannot cite, and the answer validator rejects
unclaimed numbers in prose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "explainer-validation-v1"

#: The method the benchmark measured: the predictor's default attribution,
#: whose metadata names it ``grad_x_input_v2`` (``/v1/attributions``) or
#: ``grad_x_input_v2+token_structure_align_v2`` (``/v1/explanations``).
#: ``integrated_gradients_v1`` was never measured and resolves to not_measured.
SERVED_METHOD = "grad_x_input_v2"


class Faithfulness:
    #: Top-k deletion moved the score more than a random-atom control, on most molecules.
    BETTER_THAN_RANDOM = "better_than_random_control"
    #: It did not: the highlight is not shown to be more faithful than chance.
    NOT_BETTER_THAN_RANDOM = "not_better_than_random_control"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True, slots=True)
class ExplainerMeasurement:
    model_id: str
    endpoint: str
    task: str | None
    method: str
    measured_on: str
    source: str
    molecules_attempted: int
    determinism_checked: int
    determinism_failures: int
    spelling_checked: int
    spelling_failures: int
    faithfulness_checked: int
    faithfulness_wins: int
    median_top_k_delta: float
    median_random_delta: float

    @property
    def faithfulness(self) -> str:
        if self.faithfulness_checked == 0:
            return Faithfulness.NOT_MEASURED
        # A majority of wins *and* a larger median effect. Either alone is what
        # a coin flip or one outlier produces on a panel this small.
        if (
            self.faithfulness_wins * 2 > self.faithfulness_checked
            and self.median_top_k_delta > self.median_random_delta
        ):
            return Faithfulness.BETTER_THAN_RANDOM
        return Faithfulness.NOT_BETTER_THAN_RANDOM

    def counts(self) -> dict[str, Any]:
        return {
            "molecules_attempted": self.molecules_attempted,
            "determinism": f"{self.determinism_checked - self.determinism_failures}/"
                           f"{self.determinism_checked}",
            "spelling_invariance": f"{self.spelling_checked - self.spelling_failures}/"
                                   f"{self.spelling_checked}",
            "faithfulness_wins": f"{self.faithfulness_wins}/{self.faithfulness_checked}",
            "median_delta_top_k": round(self.median_top_k_delta, 4),
            "median_delta_random": round(self.median_random_delta, 4),
        }


#: Copied from backend/predictor/evals/xai/results/*.json (2026-09-09).
MEASUREMENTS: tuple[ExplainerMeasurement, ...] = (
    ExplainerMeasurement(
        model_id="herg-tox21-chemberta-v1", endpoint="herg", task=None,
        method=SERVED_METHOD, measured_on="2026-09-09",
        source="backend/predictor/evals/xai/results/herg-20260909.json",
        molecules_attempted=42, determinism_checked=42, determinism_failures=0,
        spelling_checked=40, spelling_failures=0,
        faithfulness_checked=34, faithfulness_wins=15,
        median_top_k_delta=0.03396926634013653, median_random_delta=0.02937954105436802,
    ),
    ExplainerMeasurement(
        model_id="herg-tox21-chemberta-v1", endpoint="tox21", task="NR-AR",
        method=SERVED_METHOD, measured_on="2026-09-09",
        source="backend/predictor/evals/xai/results/tox21-NR-AR-20260909.json",
        molecules_attempted=42, determinism_checked=42, determinism_failures=0,
        spelling_checked=40, spelling_failures=0,
        faithfulness_checked=35, faithfulness_wins=14,
        median_top_k_delta=0.03827609121799469, median_random_delta=0.044924989342689514,
    ),
)

_READING = {
    Faithfulness.BETTER_THAN_RANDOM: (
        "On the measured panel the highlighted atoms moved this score more than random atoms "
        "did. That makes them a model-attribution signal worth reading, still never a "
        "mechanism or independent evidence."
    ),
    Faithfulness.NOT_BETTER_THAN_RANDOM: (
        "On the measured panel the highlighted atoms did not move this score more than "
        "deleting random atoms. Present highlights only as what the model's gradient points "
        "at, say that their faithfulness is not established, and never as mechanism or as "
        "evidence for or against a hazard."
    ),
    Faithfulness.NOT_MEASURED: (
        "No faithfulness measurement exists for this target. Treat highlights as an "
        "unvalidated model-attribution signal, never as mechanism or evidence."
    ),
}


def _base_method(method: str | None) -> str | None:
    # The predictor reports "grad_x_input+<alignment>"; the measurement is of
    # the attribution method, whichever alignment projected it onto atoms.
    return method.split("+", 1)[0] if method else None


def lookup(model_id: str | None, endpoint: str | None, task: str | None,
           method: str | None) -> ExplainerMeasurement | None:
    base = _base_method(method)
    for item in MEASUREMENTS:
        if (item.model_id == model_id and item.endpoint == endpoint and item.task == task
                and (base is None or item.method == base)):
            return item
    return None


def model_view(model_id: str | None, endpoint: str | None, task: str | None,
               method: str | None) -> dict[str, Any]:
    """The qualitative statement a model sees beside an attribution."""
    item = lookup(model_id, endpoint, task, method)
    if item is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "faithfulness_vs_random_control": Faithfulness.NOT_MEASURED,
            "determinism": "not_measured",
            "spelling_invariance": "not_measured",
            "reading": _READING[Faithfulness.NOT_MEASURED],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "measured_on": item.measured_on,
        "determinism": "passed" if item.determinism_failures == 0 else "failed",
        "spelling_invariance": "passed" if item.spelling_failures == 0 else "failed",
        "faithfulness_vs_random_control": item.faithfulness,
        "reading": _READING[item.faithfulness],
    }


def ui_view(model_id: str | None, endpoint: str | None, task: str | None,
            method: str | None) -> dict[str, Any]:
    """The model view plus the counts and the result file, for a human reader."""
    view = model_view(model_id, endpoint, task, method)
    item = lookup(model_id, endpoint, task, method)
    if item is not None:
        view = {**view, "counts": item.counts(), "source": item.source}
    return view


def matrix() -> list[Mapping[str, Any]]:
    """Every measurement, for the capability matrix."""
    return [
        {
            "model_id": item.model_id, "endpoint": item.endpoint, "task": item.task,
            "method": item.method, "measured_on": item.measured_on, "source": item.source,
            "faithfulness_vs_random_control": item.faithfulness, **item.counts(),
        }
        for item in MEASUREMENTS
    ]
