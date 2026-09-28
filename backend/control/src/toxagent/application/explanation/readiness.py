"""Can this deployment actually explain a prediction? (XAI-01)

A loaded checkpoint proves that weights and a tokenizer could be read. It does
not prove that an explanation can be produced: the explainer additionally needs
a provider exposing ``token_attribution``, the model pinned for the endpoint
actually loaded, token-offset alignment that succeeds on this tokenizer, a
budget to run inside, an object store for the figure, and a depiction step that
survives sanitization. The report builder used to discover all of that one
endpoint at a time, at report time, and report it as an undifferentiated
``explanation_failed``.

This module answers what it *can* answer up front and per model, in the same
reason-code vocabulary a gap later carries — so "no model is loaded here" is a
fact a monitor can read rather than a diagnosis somebody makes from a failed
report.

What it can answer is narrower than it first looks, and the narrowing is the
point. ``/v1/models`` reports each model's load state and the endpoints it
serves; it does not advertise explainer features. An earlier version of this
module read that endpoint list as a feature list and looked for a
``token_attribution`` entry in it, which no model reports — so a healthy
ChemBERTa explainer came back ``attribution_unsupported`` on a live stack. A
probe that cries outage over a working service is worse than no probe.

So readiness here means load state, and nothing further is inferred.
:func:`smoke_explanation` is the end-to-end proof, offered as a diagnostic
rather than as part of readiness: running a real backward pass on every
readiness check would make an inexpensive endpoint expensive and, on a loaded
deployment, would itself be the thing that exhausts the explanation budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .service import ExplanationUnavailable

#: What ``ModelInfo.capabilities`` actually contains: the **endpoints** a model
#: serves — ``["herg", "tox21"]`` — not a list of explainer features. An earlier
#: version of this module read it as the latter and looked for a
#: ``token_attribution`` entry, which no model reports, so every healthy
#: explainer was declared ``attribution_unsupported``. A false negative there is
#: worse than no probe at all: it reads exactly like a real outage.
#:
#: Attribution support is therefore *not* derivable from ``/v1/models``. It is
#: proven only by running one, which is what :func:`smoke_explanation` is for.


@dataclass(frozen=True, slots=True)
class ExplainerReadiness:
    model_id: str
    #: "Nothing known to this probe stops an explanation." Not a promise that one
    #: will succeed — see the module docstring.
    ready: bool
    #: One of ``ExplanationUnavailable``'s codes, or ``None`` when ready.
    reason: str | None = None
    detail: str | None = None
    #: The endpoints this model serves, as ``/v1/models`` reports them.
    endpoints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ready": self.ready,
            "reason": self.reason,
            "detail": self.detail or None,
            "endpoints": list(self.endpoints),
        }


def readiness_for_model(model: Any) -> ExplainerReadiness:
    """One model's explainability, as far as its ``/v1/models`` entry can say.

    Which is: whether its checkpoint loaded. That is a real and useful fact — an
    unloaded model is the most common reason an explanation cannot be produced,
    and it is the one this endpoint can establish cheaply and truthfully.
    Anything beyond it (does the provider implement attribution, does token
    alignment succeed on this tokenizer) is not advertised here, and is not
    guessed at.
    """
    model_id = getattr(model, "model_id", "") or ""
    endpoints = tuple(getattr(model, "capabilities", ()) or ())
    if not getattr(model, "loaded", False):
        return ExplainerReadiness(
            model_id=model_id,
            ready=False,
            reason=ExplanationUnavailable.MODEL_UNAVAILABLE,
            detail=getattr(model, "blocked_reason", None) or getattr(model, "detail", "")
            or "the predictor reports this model as not loaded",
            endpoints=endpoints,
        )
    return ExplainerReadiness(model_id=model_id, ready=True, endpoints=endpoints)


async def explainer_readiness(predictor) -> dict[str, Any]:
    """Per-model explainer state, plus whether *any* model can explain.

    A deployment where no admitted model can attribute is not unhealthy — it is
    a deployment without XAI, which is a legitimate product. So this never
    decides the readiness status code; it makes the state legible.
    """
    try:
        models = await predictor.models()
    except Exception as exc:  # noqa: BLE001 — reported as a dependency state
        return {
            "available": False,
            "reason": ExplanationUnavailable.MODEL_UNAVAILABLE,
            "detail": type(exc).__name__,
            "models": [],
        }
    per_model = [readiness_for_model(model) for model in models.models]
    available = any(item.ready for item in per_model)
    return {
        "available": available,
        # "No model is loaded", not "attribution is unsupported": the second is a
        # claim about the provider's features that this endpoint cannot make.
        "reason": None if available else ExplanationUnavailable.MODEL_UNAVAILABLE,
        "probe": "model load state only; attribution support is proven by running "
                 "one, not advertised on /v1/models",
        "models": [item.to_dict() for item in per_model],
    }


async def smoke_explanation(
    predictor, *, smiles: str = "CCO", endpoint: str, task: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """One real explanation of a trivial molecule, as a diagnostic.

    Offered for the debug sequence in the handoff plan (call ``/v1/models``,
    then explain directly, then check the payload) so that the sequence is one
    request rather than three by hand. Never called from ``/health/ready``: a
    backward pass per probe is a cost a readiness check must not impose.
    """
    try:
        response = await predictor.explain(smiles, endpoint, task, model_id=model_id)
    except Exception as exc:  # noqa: BLE001 — the diagnostic *is* the report
        return {
            "ok": False,
            "reason": ExplanationUnavailable.MODEL_UNAVAILABLE,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    payload = response.model_dump(mode="json")
    from .service import classify_failure, has_numeric_attribution

    if payload.get("status") == "failed":
        return {
            "ok": False,
            "reason": classify_failure(payload) or ExplanationUnavailable.ALIGNMENT_FAILED,
            "detail": (payload.get("metadata") or {}).get("error"),
            "status": payload.get("status"),
        }
    return {
        # Numeric attribution is the thing being probed. A payload with a
        # depiction and no numbers is a picture of nothing, and reporting it as
        # ok would hide precisely the failure this exists to surface.
        "ok": has_numeric_attribution(payload),
        "reason": None if has_numeric_attribution(payload)
        else ExplanationUnavailable.ALIGNMENT_FAILED,
        "status": payload.get("status"),
        "method": payload.get("method"),
        "model_id": (payload.get("metadata") or {}).get("model_id"),
        "atom_count": len(payload.get("atoms") or ()),
        "has_depiction": bool(payload.get("depiction_svg")),
    }
