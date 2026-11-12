"""What a provider's refusal of a turn means, read from the runtime's own report.

OpenCode reports a failed turn as ``{"name": "APIError", "data": {"message":
..., "statusCode": ...}}`` (``session.error`` event, or ``info.error`` of a
synchronous message). Until this module existed every such failure collapsed
into ``runtime_protocol_error`` / "the runtime turn failed", which told the user
the runtime was broken when the provider had said something specific: "the
'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account".

Kept below the harness (like ``runtime_install``) so the connection checks and
the run gateway classify with one definition.

Only the provider's message text and status code are read; response headers and
bodies, which can echo request material, are never copied out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["model_not_supported", "signin_rejected", "rate_limited", "other"]

#: Longest provider message carried into a typed error. A refusal is a sentence,
#: not a document.
MAX_MESSAGE = 300

#: ``.`` is allowed between "model" and the verdict on purpose: a model id is
#: full of dots ("gpt-5.6-sol"), and a pattern that stopped at the first one
#: would miss exactly the new models this exists for.
_MODEL_REFUSED = re.compile(
    r"model.{0,120}?(is\s+not\s+supported|not\s+supported\s+when\s+using|does\s+not\s+exist"
    r"|not[\s_]+found|is\s+not\s+available|unknown\s+model)"
    r"|(unknown|unsupported|invalid)\s+model"
    r"|do(es)?\s+not\s+have\s+access\s+to\s+(the\s+)?model",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class Refusal:
    kind: Kind
    message: str
    status_code: int | None = None

    @property
    def retryable(self) -> bool:
        return self.kind == "rate_limited"


def _text(value: Any) -> str:
    return " ".join(value.split())[:MAX_MESSAGE] if isinstance(value, str) else ""


def classify(error: Any) -> Refusal | None:
    """The refusal in an OpenCode error object, or ``None`` when it holds none."""
    if not isinstance(error, dict):
        return None
    name = error.get("name")
    data = error.get("data") if isinstance(error.get("data"), dict) else {}
    message = _text(data.get("message")) or _text(error.get("message"))
    status = data.get("statusCode")
    status = status if isinstance(status, int) else None
    if not (name or message or status):
        return None
    if _MODEL_REFUSED.search(message):
        return Refusal("model_not_supported", message, status)
    if name == "ProviderAuthError" or status in (401, 403):
        return Refusal("signin_rejected", message or "the provider rejected this sign-in", status)
    if status == 429:
        return Refusal("rate_limited", message or "the provider's rate limit was reached", status)
    return Refusal("other", message or "the provider refused the turn", status)
