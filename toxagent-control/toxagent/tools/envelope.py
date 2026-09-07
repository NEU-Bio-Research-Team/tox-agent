"""The tool result envelope (plan section 8.2).

One shape for success and one for failure, and a failure is never dressed as
prose inside a success. A model that reads "I could not reach the provider" in
a result body has no reliable way to distinguish it from a finding; a model that
receives ``status: error`` with a code does.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "tool-result-v1"


def completed(
    *,
    call_id: str,
    tool_name: str,
    canonical: dict[str, Any],
    model_view: dict[str, Any],
    ui_view: dict[str, Any],
    observation_ids: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
    attachments: tuple[dict[str, Any], ...] = (),
    duration_ms: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "call_id": call_id,
        "tool_name": tool_name,
        "status": "completed",
        "observation_ids": list(observation_ids),
        "canonical": canonical,
        "model_view": model_view,
        "ui_view": ui_view,
        "attachments": list(attachments),
        "provenance": provenance or {},
        "duration_ms": duration_ms,
    }


def failed(
    *,
    call_id: str,
    tool_name: str,
    code: str,
    message: str,
    retryable: bool = False,
    retry_after_ms: int | None = None,
    details: dict[str, Any] | None = None,
    duration_ms: int = 0,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "retryable": retryable}
    if retry_after_ms is not None:
        error["retry_after_ms"] = retry_after_ms
    if details:
        error["details"] = details
    return {
        "schema_version": SCHEMA_VERSION,
        "call_id": call_id,
        "tool_name": tool_name,
        "status": "error",
        "error": error,
        "duration_ms": duration_ms,
    }


def is_error(envelope: dict[str, Any]) -> bool:
    return envelope.get("status") == "error"


def model_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    """What is actually handed to the runtime.

    The canonical payload and the UI view stay on the server: the first is
    unbounded, and the second exists for a human. Sending either would spend
    prompt budget on content the model has no use for and would let it quote
    values no slice ever returned.
    """
    if is_error(envelope):
        return {
            "schema_version": envelope["schema_version"],
            "call_id": envelope["call_id"],
            "tool_name": envelope["tool_name"],
            "status": "error",
            "error": envelope["error"],
        }
    return {
        "schema_version": envelope["schema_version"],
        "call_id": envelope["call_id"],
        "tool_name": envelope["tool_name"],
        "status": "completed",
        "observation_ids": envelope["observation_ids"],
        "result": envelope["model_view"],
    }
