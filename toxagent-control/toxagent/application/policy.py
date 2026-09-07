"""Product policy decisions that are not the predictor's business.

Kept out of the workflows so that "may this caller move the operating point?"
has one answer in one place, and so the answer is recorded in the snapshot that
the resulting numbers live in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..config import PolicySettings
from ..domain.errors import Forbidden, InvalidRequest
from ..predictor.contract import ENDPOINTS, TOX21_TASKS


@dataclass(frozen=True)
class Actor:
    """Who is asking. Roles come from the authenticated principal, never from
    the request body and never from a runtime."""

    subject_id: str
    roles: frozenset[str] = frozenset()

    def has_role(self, role: str) -> bool:
        return role in self.roles


def resolve_endpoints(
    requested: tuple[str, ...] | None, settings: PolicySettings
) -> tuple[str, ...]:
    endpoints = tuple(requested) if requested else settings.default_endpoints
    unknown = sorted(set(endpoints) - set(ENDPOINTS))
    if unknown:
        raise InvalidRequest(f"unknown endpoint(s): {unknown}", allowed=list(ENDPOINTS))
    if not endpoints:
        raise InvalidRequest("at least one endpoint must be requested")
    # Deduplicate while keeping the caller's order stable in the snapshot.
    seen: list[str] = []
    for endpoint in endpoints:
        if endpoint not in seen:
            seen.append(endpoint)
    return tuple(seen)


def authorise_threshold_overrides(
    overrides: Mapping[str, Any] | None, actor: Actor, settings: PolicySettings
) -> dict[str, Any] | None:
    """DEC-09. Overrides are off by default; where enabled they need an explicit
    role. A moved threshold changes every label downstream, so the decision is
    an authorisation decision, not a convenience parameter."""
    if not overrides:
        return None
    if not settings.allow_threshold_overrides:
        raise Forbidden(
            "threshold overrides are disabled for this deployment",
            code_hint="set TOXAGENT_ALLOW_THRESHOLD_OVERRIDES to enable them",
        )
    if not (actor.roles & set(settings.threshold_override_roles)):
        raise Forbidden(
            "threshold overrides require an expert role",
            required_roles=sorted(settings.threshold_override_roles),
        )
    cleaned: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in ("herg", "clintox"):
            cleaned[key] = _fraction(key, value)
        elif key == "tox21":
            if not isinstance(value, Mapping):
                raise InvalidRequest("tox21 threshold overrides must be a mapping of task -> value")
            unknown = sorted(set(value) - set(TOX21_TASKS))
            if unknown:
                raise InvalidRequest(f"unknown Tox21 task(s): {unknown}")
            cleaned["tox21"] = {task: _fraction(task, v) for task, v in value.items()}
        else:
            raise InvalidRequest(f"unknown threshold override target {key!r}")
    return cleaned


def policy_snapshot(
    *, endpoints: tuple[str, ...], overrides: Mapping[str, Any] | None, actor: Actor
) -> dict[str, Any]:
    """What is recorded alongside the numbers so a reader can reproduce them.

    ``threshold_source`` in the predictor payload already says ``request_override``
    when an override was applied; this records who was allowed to ask.
    """
    return {
        "requested_endpoints": list(endpoints),
        "threshold_overrides": dict(overrides) if overrides else None,
        "threshold_override_source": "request_override" if overrides else "model_default",
        "authorised_roles": sorted(actor.roles),
    }


def _fraction(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidRequest(f"threshold for {name} must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise InvalidRequest(f"threshold for {name} must lie in [0, 1], got {number}")
    return number
