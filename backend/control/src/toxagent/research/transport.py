"""Shared HTTP discipline for the research and compound providers.

Both providers stream a response so they can stop at a byte cap rather than
after it, then rebuild a non-streaming ``httpx.Response`` so everything
downstream — status checks, ``retry-after``, ``.json()`` — is unchanged.

That rebuild is where the audit's PubChem failure lived (P2-5). ``aiter_bytes``
yields bytes httpx has *already* decompressed, but the rebuilt response was
handed the original headers, which still said ``Content-Encoding: gzip``. Any
later read of ``.text`` or ``.json()`` therefore tried to gunzip a body that
was already plain JSON, and identity resolution failed with a decompression
error on a response that had arrived perfectly intact.

The rule this module encodes: **the entity headers must describe the bytes you
are actually carrying.** Once a body has been decoded, the headers that
described its encoding are no longer true of it, and keeping them is not
conservative — it is a lie the next reader acts on.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, TypeVar

import httpx

#: Entity headers that describe the *transferred* representation. After a
#: transparent decode they no longer describe the bytes in hand.
_STALE_ENTITY_HEADERS = ("content-encoding", "content-length")

T = TypeVar("T")


def headers_for_decoded_body(headers: httpx.Headers, body: bytes) -> httpx.Headers:
    """Headers that truthfully describe ``body``.

    ``Content-Encoding`` is dropped rather than rewritten: there is no encoding
    left to name. ``Content-Length`` is restated from the bytes actually held,
    because the original counted the compressed form.
    """
    rebuilt = httpx.Headers(headers)
    for name in _STALE_ENTITY_HEADERS:
        if name in rebuilt:
            del rebuilt[name]
    rebuilt["content-length"] = str(len(body))
    return rebuilt


def rebuild_decoded_response(
    *, response: httpx.Response, body: bytes, request: httpx.Request
) -> httpx.Response:
    """A non-streaming response carrying an already-decoded body."""
    return httpx.Response(
        status_code=response.status_code,
        headers=headers_for_decoded_body(response.headers, body),
        content=body,
        request=request,
    )


async def retry_transient(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    backoff_s: float,
    retry_on: Iterable[type[BaseException]],
    should_retry: Callable[[T], bool] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Retry a transport-level failure a bounded number of times.

    Deliberately narrow. Only the exception types a caller names are retried,
    and ``should_retry`` exists for a *status* that is transient (a 5xx), never
    for a well-formed answer a caller dislikes: re-asking a provider that
    already told us the truth about a malformed or semantically wrong payload
    just spends the budget twice for the same answer.

    Backoff is linear and bounded because these calls sit inside a run
    deadline; an exponential ladder would spend a user's whole turn waiting.
    """
    if attempts < 1:
        raise ValueError("retry_transient needs at least one attempt")
    retry_on = tuple(retry_on)
    sleeper = sleep or asyncio.sleep
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = await operation()
        except retry_on as exc:  # noqa: B030 - tuple built above
            last_exc = exc
            if attempt == attempts - 1:
                raise
        else:
            if should_retry is None or not should_retry(result) or attempt == attempts - 1:
                return result
        await sleeper(backoff_s * (attempt + 1))
    # Unreachable: the loop either returns or re-raises on its last attempt.
    raise last_exc if last_exc else RuntimeError("retry_transient made no attempt")
