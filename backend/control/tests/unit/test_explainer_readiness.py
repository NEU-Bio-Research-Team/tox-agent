"""What `/v1/models` actually tells us about explainability — and what it does not.

This file exists because the first version of the readiness probe got the
semantics of one field wrong. `ModelInfo.capabilities` lists the **endpoints** a
model serves (`["herg", "tox21"]`); the probe read it as a list of explainer
features and looked for `token_attribution` in it. No model reports that, so on a
live stack a perfectly healthy ChemBERTa explainer — one that returns thirteen
signed atom contributions and a red/green depiction when you ask it — was
reported as `attribution_unsupported`.

A probe that cries outage over a working service is worse than no probe: it
trains whoever reads it to ignore the field. So the rule these tests pin is
narrow on purpose — readiness means load state, and nothing else is inferred.
"""
from __future__ import annotations

import pytest

from toxagent.application.explanation.service import ExplanationUnavailable
from toxagent.application.explanation.readiness import (
    explainer_readiness,
    readiness_for_model,
    smoke_explanation,
)
from toxagent.predictor.schemas import ModelInfo

pytestmark = pytest.mark.anyio


def _model(**overrides) -> ModelInfo:
    return ModelInfo.model_validate({
        "model_id": "herg-tox21-chemberta-v1",
        # The real shape, as the running predictor reports it.
        "capabilities": ["herg", "tox21"],
        "loaded": True,
        "required": True,
        "detail": "",
        "blocked_reason": None,
        **overrides,
    })


def test_a_loaded_model_is_ready_even_though_it_names_no_explainer_feature():
    """The regression. `capabilities` is an endpoint list; requiring a
    `token_attribution` entry in it declares every real model unexplainable."""
    readiness = readiness_for_model(_model())
    assert readiness.ready is True
    assert readiness.reason is None
    assert readiness.endpoints == ("herg", "tox21")


def test_an_unloaded_model_is_the_one_thing_this_probe_can_refuse():
    readiness = readiness_for_model(_model(
        model_id="clintox-smilesgnn-v1",
        capabilities=["clintox"],
        loaded=False,
        blocked_reason="the pinned tokenizer.pkl is absent",
    ))
    assert readiness.ready is False
    assert readiness.reason == ExplanationUnavailable.MODEL_UNAVAILABLE
    assert "tokenizer" in (readiness.detail or "")


def test_an_unloaded_model_with_no_stated_reason_still_gets_one():
    readiness = readiness_for_model(_model(loaded=False, detail="", blocked_reason=None))
    assert readiness.detail


async def test_the_aggregate_says_no_model_is_loaded_not_that_attribution_is_missing():
    """Two different claims. Only the first is one this endpoint can make."""
    class _Predictor:
        async def models(self):
            class _Response:
                models = [_model(loaded=False), _model(model_id="other", loaded=False)]
            return _Response()

    state = await explainer_readiness(_Predictor())
    assert state["available"] is False
    assert state["reason"] == ExplanationUnavailable.MODEL_UNAVAILABLE
    # And it says out loud how shallow the probe is, so nobody reads
    # `available: true` as a promise that an explanation will succeed.
    assert "load state" in state["probe"]


async def test_one_loaded_model_is_enough_for_the_deployment_to_be_available():
    class _Predictor:
        async def models(self):
            class _Response:
                models = [_model(model_id="clintox-smilesgnn-v1", loaded=False), _model()]
            return _Response()

    state = await explainer_readiness(_Predictor())
    assert state["available"] is True
    assert state["reason"] is None
    # Per model, so "which one is down" is answerable without a second request.
    assert [m["ready"] for m in state["models"]] == [False, True]


async def test_an_unreachable_predictor_is_reported_not_raised():
    class _Predictor:
        async def models(self):
            raise ConnectionError("no route to host")

    state = await explainer_readiness(_Predictor())
    assert state["available"] is False
    assert state["detail"] == "ConnectionError"
    assert state["models"] == []


# --- the diagnostic that *can* prove attribution ----------------------------

async def test_the_smoke_diagnostic_is_what_proves_attribution_works():
    """Readiness cannot establish this; running one can."""
    class _Response:
        @staticmethod
        def model_dump(mode="json"):
            return {
                "status": "completed",
                "atoms": [{"atom_index": 0, "signed_contribution": 0.4}],
                "method": "grad_x_input_v2+token_structure_align_v2",
                "depiction_svg": "<svg/>",
                "metadata": {"model_id": "herg-tox21-chemberta-v1"},
            }

    class _Predictor:
        async def explain(self, smiles, endpoint, task=None, *, model_id=None):
            return _Response()

    result = await smoke_explanation(_Predictor(), endpoint="herg")
    assert result["ok"] is True
    assert result["atom_count"] == 1
    assert result["has_depiction"] is True


async def test_a_depiction_with_no_numbers_is_not_a_pass():
    """A picture of nothing. Reporting it as ok would hide exactly the failure
    this diagnostic exists to surface."""
    class _Response:
        @staticmethod
        def model_dump(mode="json"):
            return {
                "status": "completed", "atoms": [], "bonds": [],
                "depiction_svg": "<svg/>", "metadata": {},
            }

    class _Predictor:
        async def explain(self, smiles, endpoint, task=None, *, model_id=None):
            return _Response()

    result = await smoke_explanation(_Predictor(), endpoint="herg")
    assert result["ok"] is False
    assert result["reason"] == ExplanationUnavailable.ALIGNMENT_FAILED


async def test_the_diagnostic_reports_a_raised_error_rather_than_propagating_it():
    class _Predictor:
        async def explain(self, smiles, endpoint, task=None, *, model_id=None):
            raise RuntimeError("the model is not loaded")

    result = await smoke_explanation(_Predictor(), endpoint="herg")
    assert result["ok"] is False
    assert "RuntimeError" in result["detail"]
