"""Typed product errors.

Plan section 6.6. Every failure the API can produce is a named code with a
fixed HTTP status and a fixed retryability, decided here rather than at each
raise site. The predictor learned this lesson the expensive way: an unparseable
molecule that answers ``{"label": "PARSE_ERROR", "p_toxic": 0.0}`` is a payload
a caller reads as "predicted non-toxic". Nothing in this layer may answer a
failure with a success-shaped body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final


class ToxAgentError(Exception):
    """Base for every failure that reaches a client as a typed envelope."""

    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def envelope(self, *, run_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if run_id is not None:
            body["run_id"] = run_id
        body["details"] = dict(self.detail)
        return {"error": body}


# --- request and routing ---------------------------------------------------

class InvalidRequest(ToxAgentError):
    code, http_status = "invalid_request", 400


class InvalidSmiles(ToxAgentError):
    """SCI-08: never a zero-risk prediction, always a validation failure."""

    code, http_status = "invalid_smiles", 400


class Conflict(ToxAgentError):
    code, http_status = "conflict", 409


class AdmissionBusy(Conflict):
    """A different control-plane instance holds the session admission lock."""

    retryable = True


class NotFound(ToxAgentError):
    """Generic 404. Ownership failures reuse this so a foreign session id is
    indistinguishable from a nonexistent one (plan section 14.1)."""

    code, http_status = "not_found", 404


class SessionNotFound(NotFound):
    code = "session_not_found"


class AnalysisNotFound(NotFound):
    code = "analysis_not_found"


class EvidenceNotFound(NotFound):
    code = "evidence_not_found"


class AttachmentNotFound(NotFound):
    """Raised only inside application/recognize_structure.py, caught there
    and turned into a graceful run completion — never reaches an HTTP
    boundary directly, the same shape as EvidenceNotFound."""

    code = "attachment_not_found"


class AttachmentUnavailable(ToxAgentError):
    """The product cannot durably retain an upload, so OCR must not queue."""

    code, http_status, retryable = "attachment_unavailable", 503, True


class Unauthenticated(ToxAgentError):
    code, http_status = "unauthenticated", 401


class Forbidden(ToxAgentError):
    code, http_status = "forbidden", 403


# --- predictor -------------------------------------------------------------

class PredictorNotReady(ToxAgentError):
    code, http_status, retryable = "predictor_not_ready", 503, True


class EndpointUnavailable(ToxAgentError):
    """SCI-06: an endpoint this build does not serve fails loudly. There is no
    substitute endpoint and no borrowed probability."""

    code, http_status = "endpoint_unavailable", 422


class PredictorProtocolError(ToxAgentError):
    """The predictor answered something its own contract does not describe."""

    code, http_status = "predictor_protocol_error", 502


# --- structure recognition (stateless proxy) -----------------------------

class CapabilityUnavailable(ToxAgentError):
    """A stateless capability whose backing service this deployment does not
    run at all — distinct from a configured service that is momentarily
    unreachable."""

    code, http_status, retryable = "capability_unavailable", 503, True


class StructureRecognitionUnavailable(ToxAgentError):
    """The OCR service is configured but did not answer."""

    code, http_status, retryable = "structure_recognition_unavailable", 503, True


class SmilesNotDetected(ToxAgentError):
    """The OCR service answered but found no structure in the image — never
    retryable, and never a guessed SMILES."""

    code, http_status = "smiles_not_detected", 422


# --- runtime and tools -----------------------------------------------------

class RuntimeUnavailable(ToxAgentError):
    code, http_status, retryable = "runtime_unavailable", 503, True


class RuntimeProtocolError(ToxAgentError):
    code, http_status = "runtime_protocol_error", 502


class ToolDenied(ToxAgentError):
    """PROD-06: refused at the transport, not merely hidden from the model."""

    code, http_status = "tool_denied", 403


class ToolTimeout(ToxAgentError):
    code, http_status, retryable = "tool_timeout", 504, True


# --- evidence --------------------------------------------------------------

class ProviderRateLimited(ToxAgentError):
    code, http_status, retryable = "provider_rate_limited", 429, True

    def __init__(self, message: str, retry_after_ms: int | None = None, **detail: Any) -> None:
        super().__init__(message, **detail)
        self.retry_after_ms = retry_after_ms
        if retry_after_ms is not None:
            self.detail["retry_after_ms"] = retry_after_ms


class EvidenceUnavailable(ToxAgentError):
    code, http_status, retryable = "evidence_unavailable", 503, True


# --- answer lifecycle ------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Violation:
    """One reason a candidate answer was rejected.

    ``path`` points into the candidate so the correction attempt knows exactly
    what to change; ``expected``/``actual`` are filled for value mismatches.
    """

    code: str
    message: str
    path: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path:
            out["path"] = self.path
        if self.expected is not None:
            out["expected"] = self.expected
        if self.actual is not None:
            out["actual"] = self.actual
        return out


class AnswerValidationFailed(ToxAgentError):
    """Raised only for a candidate generation below the run's cap (plan section
    9.5) — the generation that exhausts the cap instead commits a server-built
    fallback and never raises. So whenever this is raised, a correction attempt
    genuinely remains, and ``retryable`` must say so: the tool error envelope
    (``tools/envelope.py``) forwards this flag to the model verbatim, and a
    live Phase 3 run showed a model reading the base class's ``retryable:
    False`` default and simply not calling ``submit_grounded_answer`` again
    (progress log §4.6) — the correction loop existing in code was invisible to
    the model that needed to use it."""

    code, http_status, retryable = "answer_validation_failed", 422, True

    def __init__(self, message: str, violations: list[Violation] | None = None, **detail: Any):
        super().__init__(message, **detail)
        self.violations = violations or []
        self.detail["violations"] = [v.to_dict() for v in self.violations]


class DeadlineExceeded(ToxAgentError):
    code, http_status = "deadline_exceeded", 504


# The set the API documents. A code not listed here cannot leave the process:
# `api.errors` maps anything else to `internal_error` without a message body.
PUBLIC_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "invalid_request", "invalid_smiles", "not_found", "session_not_found",
        "analysis_not_found", "attachment_unavailable", "unauthenticated", "forbidden", "conflict",
        "capability_unavailable", "structure_recognition_unavailable", "smiles_not_detected",
        "endpoint_unavailable", "predictor_not_ready", "predictor_protocol_error",
        "runtime_unavailable", "runtime_protocol_error", "tool_denied",
        "tool_timeout", "provider_rate_limited", "evidence_unavailable",
        "answer_validation_failed", "deadline_exceeded", "internal_error",
    }
)
