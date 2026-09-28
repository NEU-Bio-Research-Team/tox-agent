"""P2-5: a gzipped PubChem answer must not fail after it arrived intact.

The audit's CCO identity lookup failed with a decompression error. Nothing was
wrong with the response: the provider streamed the body so it could stop at a
byte cap, httpx decoded it on the way through, and the rebuilt response was
handed the original ``Content-Encoding: gzip`` header — so the next read tried
to gunzip plain JSON.

These tests drive a real gzip round trip through the transport, because the
bug is invisible to any fixture that only speaks in already-decoded strings.
"""
from __future__ import annotations

import gzip
import json

import httpx
import pytest

from toxagent.platform.config import CompoundSettings
from toxagent.domain.errors import EvidenceUnavailable
from toxagent.research.providers.pubchem import PubChemCompoundProvider
from toxagent.research.transport import (
    headers_for_decoded_body,
    rebuild_decoded_response,
    retry_transient,
)

pytestmark = pytest.mark.anyio

_PROPERTIES = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 702,
                "MolecularFormula": "C2H6O",
                "MolecularWeight": "46.07",
                "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "Title": "Ethanol",
            }
        ]
    }
}


def _settings(**overrides) -> CompoundSettings:
    return CompoundSettings(retry_backoff_s=0.0, **overrides)


def _gzip_handler(payload: dict, *, status: int = 200):
    body = gzip.compress(json.dumps(payload).encode("utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        if "/synonyms/" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(
            status,
            content=body,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": str(len(body)),
            },
        )

    return handler


# --- the finding ------------------------------------------------------------


async def test_a_gzipped_property_table_resolves() -> None:
    provider = PubChemCompoundProvider(
        _settings(), transport=httpx.MockTransport(_gzip_handler(_PROPERTIES))
    )
    record = await provider.resolve(canonical_smiles="CCO")
    assert record.resolved is True
    assert record.preferred_name == "Ethanol"
    await provider.aclose()


async def test_an_uncompressed_answer_still_resolves() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/synonyms/" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json=_PROPERTIES)

    provider = PubChemCompoundProvider(_settings(), transport=httpx.MockTransport(handler))
    record = await provider.resolve(canonical_smiles="CCO")
    assert record.resolved is True
    await provider.aclose()


# --- the rule the fix encodes ----------------------------------------------


def test_decoded_headers_no_longer_claim_an_encoding() -> None:
    original = httpx.Headers(
        {"content-encoding": "gzip", "content-length": "17", "content-type": "application/json"}
    )
    rebuilt = headers_for_decoded_body(original, b'{"a": 1}')
    assert "content-encoding" not in rebuilt
    assert rebuilt["content-length"] == "8"
    assert rebuilt["content-type"] == "application/json"


def test_a_rebuilt_response_can_be_read_twice() -> None:
    request = httpx.Request("GET", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/x")
    streamed = httpx.Response(200, headers={"content-encoding": "br"}, request=request)
    rebuilt = rebuild_decoded_response(
        response=streamed, body=b'{"ok": true}', request=request
    )
    assert rebuilt.json() == {"ok": True}
    assert rebuilt.text == '{"ok": true}'


# --- bounded retry ----------------------------------------------------------


async def test_a_connect_failure_is_retried_within_its_bound() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/synonyms/" in request.url.path:
            return httpx.Response(404)
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=_PROPERTIES)

    provider = PubChemCompoundProvider(
        _settings(retry_attempts=3), transport=httpx.MockTransport(handler)
    )
    record = await provider.resolve(canonical_smiles="CCO")
    assert record.resolved is True
    assert attempts["n"] == 3
    await provider.aclose()


async def test_retries_run_out_and_the_gap_stays_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = PubChemCompoundProvider(
        _settings(retry_attempts=2), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EvidenceUnavailable):
        await provider.resolve(canonical_smiles="CCO")
    await provider.aclose()


async def test_a_malformed_payload_is_not_retried() -> None:
    """It is an answer, not a transport failure. Asking again costs twice and
    returns the same broken shape."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})

    provider = PubChemCompoundProvider(
        _settings(retry_attempts=3), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EvidenceUnavailable):
        await provider.resolve(canonical_smiles="CCO")
    assert attempts["n"] == 1
    await provider.aclose()


async def test_a_404_is_an_answer_not_a_retryable_failure() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404)

    provider = PubChemCompoundProvider(
        _settings(retry_attempts=3), transport=httpx.MockTransport(handler)
    )
    record = await provider.resolve(canonical_smiles="CCO")
    assert record.resolved is False
    assert attempts["n"] == 1
    await provider.aclose()


async def test_a_5xx_is_retried_then_reported() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503)

    provider = PubChemCompoundProvider(
        _settings(retry_attempts=3), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EvidenceUnavailable):
        await provider.resolve(canonical_smiles="CCO")
    assert attempts["n"] == 3
    await provider.aclose()


async def test_retry_transient_refuses_a_zero_attempt_budget() -> None:
    with pytest.raises(ValueError):
        await retry_transient(
            lambda: _noop(), attempts=0, backoff_s=0, retry_on=(RuntimeError,)
        )


async def _noop() -> None:
    return None
