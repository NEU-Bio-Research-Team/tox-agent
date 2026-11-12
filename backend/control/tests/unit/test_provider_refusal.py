"""What a provider's refusal means (connections/refusal.py).

The run failure that started this: OpenAI answered "The 'gpt-5.4-mini' model is
not supported when using Codex with a ChatGPT account." and the product reported
``runtime_protocol_error`` -- "the runtime turn failed" -- as if the runtime were
broken. These pin the sentences providers really send to the code they map to.
"""
from __future__ import annotations

import pytest

from toxagent.connections.refusal import MAX_MESSAGE, classify


def _api_error(message: str, status: int | None = 400, **extra) -> dict:
    data = {"message": message, "isRetryable": False, **extra}
    if status is not None:
        data["statusCode"] = status
    return {"name": "APIError", "data": data}


@pytest.mark.parametrize(
    "message",
    [
        # The exact sentence in the production OpenCode log.
        "The 'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account.",
        "The model `gpt-9` does not exist or you do not have access to it.",
        "Unknown model: gpt-nope",
        "model_not_found: the model gpt-5.6-sol is not available",
        "Invalid model 'x'",
    ],
)
def test_a_refused_model_is_a_model_refusal(message):
    refusal = classify(_api_error(message))
    assert refusal is not None and refusal.kind == "model_not_supported"
    assert refusal.message.startswith(message[:20])
    assert refusal.retryable is False


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_sign_in_is_not_a_model_refusal(status):
    refusal = classify(_api_error('Unauthorized: {"detail":"Could not parse your authentication token."}', status))
    assert refusal is not None and refusal.kind == "signin_rejected" and refusal.status_code == status


def test_a_provider_auth_error_is_a_rejected_sign_in_without_a_status():
    refusal = classify({"name": "ProviderAuthError", "data": {"message": "token expired"}})
    assert refusal is not None and refusal.kind == "signin_rejected"


def test_a_rate_limit_is_retryable_and_not_a_model_refusal():
    refusal = classify(_api_error("Too many requests", 429))
    assert refusal is not None and refusal.kind == "rate_limited" and refusal.retryable is True


def test_a_model_sentence_wins_over_a_status():
    # A 403 whose text says the model is unavailable is about the model.
    refusal = classify(_api_error("The 'gpt-5.4-mini' model is not supported when using Codex", 403))
    assert refusal is not None and refusal.kind == "model_not_supported"


def test_anything_else_stays_other():
    refusal = classify(_api_error("upstream exploded", 500))
    assert refusal is not None and refusal.kind == "other" and refusal.message == "upstream exploded"


@pytest.mark.parametrize("value", [None, {}, "boom", 3, [], {"data": {}}])
def test_no_refusal_in_nothing(value):
    assert classify(value) is None


def test_headers_and_bodies_never_leave_the_error():
    refusal = classify(_api_error(
        "nope", 500,
        responseHeaders={"set-cookie": "session=SECRET"}, responseBody='{"token": "SECRET"}',
    ))
    assert refusal is not None and "SECRET" not in repr(refusal)


def test_a_long_message_is_cut_and_whitespace_collapsed():
    refusal = classify(_api_error("model " + "is not supported " * 100 + "\n\n x"))
    assert refusal is not None and len(refusal.message) <= MAX_MESSAGE and "\n" not in refusal.message
