"""EuropePMC provider contract (plan sections 8.4, 14.4, DEC-03).

Fixture bodies below are trimmed from a real ``resultType=core`` response
captured live against ``https://www.ebi.ac.uk/europepmc/webservices/rest``
(2026-09-05) — the same "snapshot the real shape" discipline used for the
predictor and OpenCode contracts, just inlined here rather than a separate
file since EuropePMC's REST API is a stable public web service, not a pinned
binary.
"""
from __future__ import annotations


import httpx
import pytest

from toxagent.platform.config import ResearchSettings
from toxagent.domain.errors import EvidenceUnavailable, ProviderRateLimited
from toxagent.domain.evidence import SourceIdentifier, SourceType
from toxagent.research.providers.europepmc import (
    EuropePmcProvider,
    full_text_id,
    hit_from_record,
)

pytestmark = pytest.mark.anyio

SETTINGS = ResearchSettings(
    base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
    allowed_hosts=("www.ebi.ac.uk", "europepmc.org"),
    contact_email="toxagent-dev@example.test",
)

CORE_RECORD = {
    "id": "42593910",
    "source": "MED",
    "pmid": "42593910",
    "doi": "10.1021/acs.jmedchem.5c03742",
    "title": "Overcoming hERG Cardiotoxicity via Rational Design to Discover a Safe Antifibrotic Lead.",
    "authorString": "Yang T, Li R, Zhang X, Tang Y.",
    "journalInfo": {
        "issue": "15", "volume": "69",
        "journal": {
            "title": "Journal of medicinal chemistry",
            "medlineAbbreviation": "J Med Chem",
        },
    },
    "pubYear": "2026",
    "abstractText": "Liver fibrosis is a major global health challenge with limited treatment options.",
    "pubTypeList": {"pubType": ["Journal Article"]},
    "firstPublicationDate": "2026-08-01",
}

RESPONSE_BODY = {
    "version": "6.9",
    "hitCount": 1,
    "request": {"queryString": "hERG AND cardiotoxicity", "resultType": "CORE"},
    "resultList": {"result": [CORE_RECORD]},
}


def _mock(handler) -> EuropePmcProvider:
    return EuropePmcProvider(SETTINGS, transport=httpx.MockTransport(handler))


def _json_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=RESPONSE_BODY)


async def test_a_search_hit_is_parsed_into_a_fully_populated_search_hit():
    provider = _mock(_json_ok)
    hits = await provider.search(query="hERG AND cardiotoxicity", source_types=None, date_from=None, limit=10)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.provider_record_id == "MED:42593910"
    assert hit.source_type is SourceType.ARTICLE
    assert hit.title == CORE_RECORD["title"]
    assert hit.authors == ("Yang T", "Li R", "Zhang X", "Tang Y")
    # EuropePMC's authorString always ends with a trailing period.
    assert hit.identifier.pmid == "42593910"
    assert hit.identifier.doi == "10.1021/acs.jmedchem.5c03742"
    assert hit.canonical_url == "https://europepmc.org/article/MED/42593910"
    assert hit.abstract_or_excerpt == CORE_RECORD["abstractText"]
    assert hit.normalized_facts["journal"] == "Journal of medicinal chemistry"
    assert hit.normalized_facts["pub_type"] == "Journal Article"
    assert hit.published_at.isoformat() == "2026-08-01"


async def test_the_request_asks_for_core_result_type_and_forwards_the_contact_email():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=RESPONSE_BODY)

    provider = _mock(handler)
    await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)
    assert captured["params"]["resultType"] == "core"
    assert captured["params"]["format"] == "json"
    assert captured["params"]["pageSize"] == "5"
    assert captured["params"]["email"] == "toxagent-dev@example.test"


async def test_page_size_is_capped_at_the_configured_maximum():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["pageSize"] = request.url.params["pageSize"]
        return httpx.Response(200, json=RESPONSE_BODY)

    provider = _mock(handler)
    await provider.search(query="aspirin", source_types=None, date_from=None, limit=999)
    assert captured["pageSize"] == str(SETTINGS.max_results)


async def test_a_date_from_filter_is_appended_to_the_query():
    import datetime

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = request.url.params["query"]
        return httpx.Response(200, json=RESPONSE_BODY)

    provider = _mock(handler)
    await provider.search(
        query="aspirin", source_types=None, date_from=datetime.date(2020, 1, 1), limit=5
    )
    assert "FIRST_PDATE:[2020-01-01 TO 3000-01-01]" in captured["query"]


async def test_an_empty_query_returns_no_hits_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an empty query must not reach the provider")

    provider = _mock(handler)
    hits = await provider.search(query="   ", source_types=None, date_from=None, limit=5)
    assert hits == []


async def test_a_connection_failure_becomes_evidence_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _mock(handler)
    with pytest.raises(EvidenceUnavailable):
        await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)


async def test_a_timeout_becomes_evidence_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = _mock(handler)
    with pytest.raises(EvidenceUnavailable):
        await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)


async def test_a_429_becomes_provider_rate_limited_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "30"}, json={})

    provider = _mock(handler)
    with pytest.raises(ProviderRateLimited) as excinfo:
        await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)
    assert excinfo.value.retry_after_ms == 30000


async def test_a_5xx_becomes_evidence_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream is down")

    provider = _mock(handler)
    with pytest.raises(EvidenceUnavailable):
        await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)


async def test_a_non_json_response_becomes_evidence_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = _mock(handler)
    with pytest.raises(EvidenceUnavailable):
        await provider.search(query="aspirin", source_types=None, date_from=None, limit=5)


def test_a_base_url_off_its_own_allowlist_fails_at_construction():
    """A per-response host check would be theatre here: httpx does not follow
    redirects by default, so every response resolves to ``base_url``'s own
    host. The check that can actually catch something is a self-consistency
    check at startup — this deployment's own config disagreeing with itself
    (plan section 14.4)."""
    misconfigured = ResearchSettings(
        base_url="https://not-the-allowed-host.example/rest",
        allowed_hosts=("www.ebi.ac.uk", "europepmc.org"),
    )
    with pytest.raises(ValueError, match="misconfigured"):
        EuropePmcProvider(misconfigured)


def test_a_record_missing_a_title_or_id_is_skipped_not_crashed_on():
    assert hit_from_record({"id": "1", "source": "MED"}) is None
    assert hit_from_record({"title": "x", "source": "MED"}) is None
    assert hit_from_record({"title": "x", "id": "1"}) is None


# ------------------------------- K11: bounded fetch


def _client(handler, **overrides):
    """A provider wired to a scripted transport, with settings overridable."""
    settings = ResearchSettings(**{"contact_email": "ops@example.test", **overrides})
    return EuropePmcProvider(settings, transport=httpx.MockTransport(handler))


async def test_a_body_larger_than_the_cap_is_refused_rather_than_read():
    """A timeout bounds how long the provider may take. Nothing bounded how
    much it could send, and `.json()` reads the whole body into memory first —
    so a third party decided how much of a multi-tenant control plane's memory
    to use."""
    async def chunks():
        # No content-length: a chunked response has none, which is exactly why
        # the header check alone is not enough.
        for _ in range(8):
            yield b"x" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks(), headers={"content-type": "application/json"})

    provider = _client(handler, max_response_bytes=1024)
    with pytest.raises(EvidenceUnavailable, match="more than 1024 bytes"):
        await provider.search(query="hERG")
    await provider.aclose()


async def test_an_announced_oversize_is_refused_before_the_body_is_read():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"resultList": {"result": []}},
            headers={"content-type": "application/json", "content-length": "99999999"},
        )

    provider = _client(handler, max_response_bytes=1024)
    with pytest.raises(EvidenceUnavailable, match="announced"):
        await provider.search(query="hERG")
    await provider.aclose()


async def test_a_body_within_the_cap_still_works():
    """The cap must not be the reason nothing ever succeeds."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"resultList": {"result": []}},
            headers={"content-type": "application/json"},
        )

    provider = _client(handler, max_response_bytes=8 * 1024 * 1024)
    assert await provider.search(query="hERG") == []
    await provider.aclose()


async def test_a_non_json_content_type_is_refused_before_parsing():
    """`accept: application/json` is a request header — a wish, not a
    constraint on what comes back. An HTML error page that happens to parse
    as something must not become evidence."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>service unavailable</html>",
                              headers={"content-type": "text/html"})

    provider = _client(handler)
    with pytest.raises(EvidenceUnavailable, match="not JSON"):
        await provider.search(query="hERG")
    await provider.aclose()


async def test_the_client_does_not_follow_a_redirect_off_the_allowed_host():
    """The host allowlist is checked once, at startup, against `base_url`.
    That is only equivalent to checking every response while nothing can move
    the request somewhere else."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://attacker.example.com/x"})

    provider = _client(handler)
    with pytest.raises(EvidenceUnavailable) as excinfo:
        await provider.search(query="hERG")
    # A 302 is not success, so it is reported as a provider failure rather
    # than followed.
    assert "302" in str(excinfo.value)
    await provider.aclose()


async def test_an_oversized_response_counts_as_a_provider_failure():
    """It opens the circuit like any other failure: a provider sending
    unbounded bodies should stop being called, not be retried forever."""
    async def chunks():
        for _ in range(4):
            yield b"x" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks(),
                              headers={"content-type": "application/json"})

    provider = _client(handler, max_response_bytes=64, circuit_failure_threshold=2)
    for _ in range(2):
        with pytest.raises(EvidenceUnavailable):
            await provider.search(query="hERG")
    with pytest.raises(EvidenceUnavailable, match="(?i)circuit|unavailable"):
        await provider.search(query="hERG")
    await provider.aclose()


# ── full text (issue #43) ────────────────────────────────────────────────

FULL_TEXT_XML = """<article>
  <front><article-meta><abstract><p>We measured hERG block.</p></abstract></article-meta></front>
  <body>
    <sec><title>Results</title><p>The IC50 was 0.9 nM.</p></sec>
  </body>
</article>"""


#: The one route Europe PMC serves full text on. ``/{source}/{id}/fullTextXML``
#: answers 404 for every record (checked against the live API 2026-10-21:
#: ``/PMC3257301/fullTextXML`` 200, ``/PMC/PMC3257301/fullTextXML`` and
#: ``/MED/22238641/fullTextXML`` 404), so these tests pin the path exactly.
FULL_TEXT_PATH = "/europepmc/webservices/rest/{}/fullTextXML"
WITH_PMCID = SourceIdentifier(pmid="42593910", pmcid="PMC1234567")


def _xml(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=FULL_TEXT_XML, headers={"content-type": "application/xml"})


async def test_full_text_comes_back_as_located_passages():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _xml(request)

    fetch = await _mock(handler).fetch_full_text(
        provider_record_id="MED:42593910", identifier=WITH_PMCID,
    )

    assert seen == [FULL_TEXT_PATH.format("PMC1234567")]
    assert fetch.retrieved
    assert fetch.full_text_id == "PMC1234567"
    assert fetch.section_titles == ("Results",)
    assert any(p.locator == "sec:1/p:1" and "0.9 nM" in p.text for p in fetch.passages)


async def test_a_pmc_record_is_fetched_under_its_own_id():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _xml(request)

    fetch = await _mock(handler).fetch_full_text(provider_record_id="PMC:PMC7654321")

    assert seen == [FULL_TEXT_PATH.format("PMC7654321")]
    assert fetch.retrieved


async def test_a_record_naming_no_pmcid_is_reported_without_a_request():
    """A MEDLINE record with no deposited text cannot be fetched under its
    PMID; asking anyway is the request that always 404'd."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError(f"no request expected, got {request.url}")

    fetch = await _mock(handler).fetch_full_text(
        provider_record_id="MED:42593910", identifier=SourceIdentifier(pmid="42593910"),
    )

    assert not fetch.retrieved
    assert "PMCID" in (fetch.reason or "")
    assert fetch.full_text_id is None


@pytest.mark.parametrize("pmcid", ["PMC1/../../search", "PMC12?x=1", "pmc", "", "MED:1"])
def test_a_pmcid_that_is_not_one_never_reaches_a_request_path(pmcid):
    """The PMCID is provider content placed into a URL path."""
    assert full_text_id("MED:1", SourceIdentifier(pmcid=pmcid)) is None


def test_a_pmcid_is_normalised_before_it_is_used():
    assert full_text_id("MED:1", SourceIdentifier(pmcid=" pmc42 ")) == "PMC42"
    assert full_text_id("MED:1", SourceIdentifier(pmcid="42")) == "PMC42"
    assert full_text_id("PMC:PMC9", None) == "PMC9"


async def test_a_record_with_no_full_text_is_a_failed_access_not_an_exception():
    """404 here is the provider answering correctly about one document."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    fetch = await _mock(handler).fetch_full_text(provider_record_id="PMC:PMC1")

    assert not fetch.retrieved
    assert "no full text" in (fetch.reason or "")
    assert fetch.full_text_id == "PMC1"
    assert fetch.passages == ()


async def test_an_ordinary_404_does_not_trip_the_circuit_breaker():
    """The breaker exists for a provider that is unwell, not for a paper that
    has no deposited text."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.endswith("fullTextXML"):
            calls += 1
            return httpx.Response(404, text="not found")
        return httpx.Response(200, json=RESPONSE_BODY)

    provider = _mock(handler)
    for index in range(SETTINGS.circuit_failure_threshold + 2):
        assert not (
            await provider.fetch_full_text(provider_record_id=f"PMC:PMC{index + 1}")
        ).retrieved
    assert calls == SETTINGS.circuit_failure_threshold + 2
    # the search path still works afterwards
    hits = await provider.search(query="hERG", source_types=None, date_from=None, limit=1)
    assert len(hits) == 1


async def test_a_non_xml_full_text_answer_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>paywall</html>",
                              headers={"content-type": "text/html"})

    fetch = await _mock(handler).fetch_full_text(
        provider_record_id="MED:42593910", identifier=WITH_PMCID,
    )
    assert not fetch.retrieved
    assert "not XML" in (fetch.reason or "")


async def test_a_provider_outage_during_a_full_text_fetch_is_reported_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    fetch = await _mock(handler).fetch_full_text(
        provider_record_id="MED:42593910", identifier=WITH_PMCID,
    )
    assert not fetch.retrieved
    assert fetch.reason
