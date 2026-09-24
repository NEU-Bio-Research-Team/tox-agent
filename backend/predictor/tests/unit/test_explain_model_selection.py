"""Attribution and explanation resolve a model explicitly, or refuse.

I10/I11: both used `registry.for_capability`, which picks the single provider
serving an endpoint and raises once there are two. That is safe today and
wrong tomorrow — the caller had no way to say *which* model, so a prediction
made by B could be explained by A the moment a second model was admitted.

They now take `model_id` and go through `registry.resolve`, which either
returns the named model or raises. There is deliberately no fallback: an
attribution computed by a different model than the prediction it explains is
worse than no attribution.
"""
from __future__ import annotations

import pytest

from toxpred.application.attribution import AttributionService
from toxpred.application.explain import ExplainService
from toxpred.scientific.artifacts import ArtifactError


class FakeProvider:
    def __init__(self, model_id: str, capabilities: set[str]) -> None:
        self.model_id = model_id
        self.capabilities = frozenset(capabilities)

    def token_attribution(self, canonical_smiles, *, head, task_index=None, method="grad_x_input"):
        return {
            "method": "grad_x_input_v2",
            "target": "logit",
            "mapping_version": "token-offset-v2",
            "model_id": self.model_id,
            "probability": 0.5,
            "tokens": [
                {"token": "C", "position": 0, "signed_contribution": 0.1,
                 "magnitude": 0.1, "importance": 0.1, "relative_importance": 1.0,
                 "offsets": [0, 1]},
            ],
        }


class FakeRegistry:
    """Only what these services use, with the real resolution semantics."""

    def __init__(self, *providers: FakeProvider) -> None:
        self._providers = {p.model_id: p for p in providers}

    def get(self, model_id):
        try:
            return self._providers[model_id]
        except KeyError:
            raise ArtifactError(f"unknown model {model_id!r}") from None

    def for_capability(self, capability):
        matches = [p for p in self._providers.values() if capability in p.capabilities]
        if not matches:
            raise ArtifactError(f"no model provides {capability!r}")
        if len(matches) > 1:
            raise ArtifactError(f"capability {capability!r} is ambiguous")
        return matches[0]

    def resolve(self, *, capability, model_id=None):
        if model_id is None:
            return self.for_capability(capability)
        provider = self.get(model_id)
        if capability not in provider.capabilities:
            raise ArtifactError(f"incompatible_model: {model_id!r} does not provide {capability!r}")
        return provider


A = FakeProvider("herg-v1", {"herg"})
B = FakeProvider("herg-v2", {"herg"})


def attribution(*providers) -> AttributionService:
    return AttributionService(FakeRegistry(*providers))


# --- one model per endpoint: unchanged behaviour -----------------------------

def test_a_single_model_still_resolves_without_being_named():
    result = attribution(A).attribute("CCO", "herg")
    assert result["metadata"]["model_id"] == "herg-v1"


# --- two models: the caller must choose --------------------------------------

def test_naming_the_model_selects_it():
    result = attribution(A, B).attribute("CCO", "herg", model_id="herg-v2")
    assert result["metadata"]["model_id"] == "herg-v2"

    other = attribution(A, B).attribute("CCO", "herg", model_id="herg-v1")
    assert other["metadata"]["model_id"] == "herg-v1"


def test_two_models_and_no_choice_is_refused_rather_than_guessed():
    with pytest.raises(ArtifactError, match="ambiguous"):
        attribution(A, B).attribute("CCO", "herg")


def test_an_unknown_model_is_refused_rather_than_falling_back():
    with pytest.raises(ArtifactError, match="unknown model"):
        attribution(A, B).attribute("CCO", "herg", model_id="herg-v9")


def test_a_model_that_does_not_serve_the_endpoint_is_refused():
    tox_only = FakeProvider("tox21-v1", {"tox21"})
    with pytest.raises(ArtifactError, match="incompatible_model"):
        attribution(A, tox_only).attribute("CCO", "herg", model_id="tox21-v1")


# --- the explanation layer forwards the same choice --------------------------

def test_explain_forwards_the_model_to_attribution():
    service = ExplainService(attribution(A, B))
    result = service.explain("CCO", "herg", model_id="herg-v2")
    assert result["metadata"]["model_id"] == "herg-v2"


def test_explain_is_refused_for_an_ambiguous_endpoint_too():
    service = ExplainService(attribution(A, B))
    with pytest.raises(ArtifactError, match="ambiguous"):
        service.explain("CCO", "herg")
