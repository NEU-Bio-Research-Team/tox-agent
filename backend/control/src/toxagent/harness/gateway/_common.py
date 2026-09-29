"""Module-level helpers the gateway class and its mixins share."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ...application.runs.scheduler import RunContext
from ...domain.runtime import AuthMode


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _no_commit(context: RunContext) -> None:
    """A turn whose product is committed by its caller, not by the gateway."""
    return None


@dataclass(frozen=True)
class ResolvedProfile:
    """Everything a run's AI profile actually configured.

    A tuple of (provider, model, connection_id) was what the gateway used to
    resolve, which is why the endpoint and the credential never reached the
    runtime (I12).
    """

    provider_id: str
    model_id: str
    connection_id: str | None
    base_url: str | None
    credential: str | None
    auth_mode: AuthMode


log = logging.getLogger("toxagent.gateway")


#: Live sweep 2026-09-06 (progress log section 14): a live run can end in
#: TURN_IDLE having called only tools, no submit_grounded_answer, and no
#: product record exists of what the runtime actually said instead — the
#: exception raised for it (below) names the symptom, not the cause. This
#: caps how much of the runtime's own final text a diagnostic log line may
#: hold: enough to see whether the model wrote a plain-prose answer instead
#: of calling the tool, never the full turn.
_DIAGNOSTIC_DELTA_PREVIEW_CHARS = 400
