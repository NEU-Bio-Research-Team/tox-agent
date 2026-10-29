"""Full text captured once and replayed offline (#43).

The store answers a frozen run's ``fetch_full_text`` without the network: the
text that was captured, the access failure that was captured, or "not held"
for a record it never captured -- never a fetch. Bytes that changed after the
capture are refused. A live provider's text is fetched once per document even
when several sessions ask at once, and an outage is never cached.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tests.support.research import POTENCY_JATS
from toxagent.platform.config import ResearchSettings
from toxagent.research.fulltext import PARSER_VERSION, parse_jats
from toxagent.research.fulltext_store import (
    FrozenFullTextProvider,
    FrozenFullTextStore,
    FullTextStoreError,
    build_store,
    freeze_document,
    write_store,
)
from toxagent.research.interfaces import FullTextCapableProvider, FullTextFetch
from toxagent.research.providers import build_provider
from toxagent.research.providers.europepmc import EuropePmcProvider

pytestmark = pytest.mark.anyio

RECORD = "MED:11111111"


def _retrieved() -> FullTextFetch:
    document = parse_jats(POTENCY_JATS)
    return FullTextFetch(
        retrieved=True, passages=document.passages, section_titles=document.section_titles,
        omitted=dict(document.omitted), full_text_id="PMC1111111",
        provenance={"served_by": "europepmc", "parser_version": PARSER_VERSION,
                    "source_sha256": "a" * 64},
    )


def _documents() -> dict:
    return {
        RECORD: freeze_document(_retrieved(), provider="europepmc",
                                captured_at="2026-10-21T00:00:00+00:00"),
        "MED:22222222": freeze_document(
            FullTextFetch(retrieved=False, reason="the provider holds no full text for PMC2",
                          full_text_id="PMC2"),
            provider="europepmc", captured_at="2026-10-21T00:00:00+00:00",
        ),
    }


def test_a_captured_table_row_is_served_with_its_capture():
    store = FrozenFullTextStore.from_document(build_store(_documents()), name="store")
    fetched = store.fetch(RECORD, served_by="snapshot")
    assert fetched.retrieved and not fetched.transient
    row = next(p for p in fetched.passages if p.locator == "table-wrap:1/row:3")
    assert row.cells == ("compound 7", "0.38")
    assert fetched.provenance["replay"] is True
    assert fetched.provenance["captured_from"] == "europepmc"
    assert fetched.provenance["source_sha256"] == "a" * 64
    assert fetched.provenance["parser_version"] == PARSER_VERSION


def test_a_captured_access_failure_and_an_uncaptured_record_are_definitive():
    store = FrozenFullTextStore.from_document(build_store(_documents()), name="store")
    failed = store.fetch("MED:22222222", served_by="snapshot")
    assert not failed.retrieved and not failed.transient
    assert "no full text" in failed.reason
    absent = store.fetch("MED:99999999", served_by="snapshot")
    assert not absent.retrieved and not absent.transient
    assert "holds no text" in absent.reason


def test_a_transient_failure_is_never_frozen():
    with pytest.raises(FullTextStoreError):
        freeze_document(FullTextFetch(retrieved=False, reason="down", transient=True),
                        provider="europepmc", captured_at="now")


def test_changed_bytes_are_refused(tmp_path):
    path = tmp_path / "store.json"
    pin = write_store(_documents(), path)
    assert len(FrozenFullTextStore.load(path, sha256=pin)) == 2
    data = json.loads(path.read_text())
    data["documents"][RECORD]["passages"][-1]["text"] = "compound 7 | hERG IC50 (uM): 3.8"
    path.write_text(json.dumps(data))
    with pytest.raises(FullTextStoreError):
        FrozenFullTextStore.load(path, sha256=pin)
    with pytest.raises(FullTextStoreError):
        FrozenFullTextStore.from_document(data, name="edited")
    with pytest.raises(FullTextStoreError):
        FrozenFullTextStore.load(path, sha256="")


def _snapshot_fixture(tmp_path, *, full_text=None):
    from toxagent.domain.provenance import content_sha256

    evidence = {
        "search": {"herg": [RECORD]},
        "records": {RECORD: {"provider_record_id": RECORD, "title": "hERG potency series",
                             "identifier": {"pmid": "11111111"},
                             "abstract_or_excerpt": "We screened a series."}},
    }
    if full_text is not None:
        evidence["full_text"] = full_text
    body = {"fixture_version": "eval-fixture-v1", "name": "t", "evidence": evidence}
    body["content_sha256"] = content_sha256(body)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(body))
    return path


async def test_a_frozen_provider_is_text_capable_only_with_a_store(tmp_path):
    plain = build_provider(ResearchSettings(provider="snapshot",
                                            snapshot_path=str(_snapshot_fixture(tmp_path))))
    assert not isinstance(plain, FullTextCapableProvider)

    store_path = tmp_path / "store.json"
    pin = write_store(_documents(), store_path)
    frozen = build_provider(ResearchSettings(
        provider="snapshot", snapshot_path=str(_snapshot_fixture(tmp_path)),
        full_text_path=str(store_path), full_text_sha256=pin,
    ))
    assert isinstance(frozen, FullTextCapableProvider) and frozen.name == "snapshot"
    hits = await frozen.search(query="herg potency", source_types=None, date_from=None, limit=5)
    assert [h.provider_record_id for h in hits] == [RECORD]
    fetched = await frozen.fetch_full_text(provider_record_id=RECORD)
    assert fetched.retrieved and fetched.provenance["store_sha256"] == pin

    inline = build_provider(ResearchSettings(
        provider="snapshot",
        snapshot_path=str(_snapshot_fixture(tmp_path, full_text=build_store(_documents()))),
    ))
    assert isinstance(inline, FullTextCapableProvider)


async def test_an_injected_outage_reaches_the_text_as_transient(tmp_path):
    store_path = tmp_path / "store.json"
    pin = write_store(_documents(), store_path)
    provider = build_provider(ResearchSettings(
        provider="snapshot", snapshot_path=str(_snapshot_fixture(tmp_path)),
        snapshot_fault="unavailable", full_text_path=str(store_path), full_text_sha256=pin,
    ))
    assert isinstance(provider, FrozenFullTextProvider)
    fetched = await provider.fetch_full_text(provider_record_id=RECORD)
    assert fetched.transient and not fetched.retrieved


def test_a_live_provider_takes_no_frozen_store():
    with pytest.raises(ValueError):
        build_provider(ResearchSettings(full_text_path="/tmp/x.json"))


async def test_one_document_is_fetched_once_and_an_outage_is_not_cached():
    calls: list[str] = []
    state = {"down": True}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        await asyncio.sleep(0.01)
        if state["down"]:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text=POTENCY_JATS,
                              headers={"content-type": "application/xml"})

    provider = EuropePmcProvider(ResearchSettings(), transport=httpx.MockTransport(handler))
    try:
        first = await provider.fetch_full_text(provider_record_id="PMC:PMC1111111")
        assert first.transient
        state["down"] = False
        outage_calls = len(calls)
        results = await asyncio.gather(*(
            provider.fetch_full_text(provider_record_id="PMC:PMC1111111") for _ in range(5)
        ))
        assert all(r.retrieved for r in results)
        assert results[0].provenance["source_sha256"]
        assert results[0].provenance["replay"] is False
        before = len(calls)
        again = await provider.fetch_full_text(provider_record_id="PMC:PMC1111111")
        assert again.retrieved and len(calls) == before
    finally:
        await provider.aclose()
    # Five concurrent readers and one later one: a single live fetch.
    assert len(calls) - outage_calls == 1


async def test_a_capture_freezes_definitive_answers_and_skips_outages():
    from evals.investigation.capture_full_text import capture
    from tests.support.research import FullTextStubProvider

    provider = FullTextStubProvider(documents={RECORD: POTENCY_JATS})
    records = {RECORD: {"identifier": {"pmid": "11111111"}},
               "MED:22222222": {"identifier": {}}}
    documents, transient = await capture(records, provider, captured_at="t")
    assert documents[RECORD]["outcome"] == "retrieved"
    assert documents["MED:22222222"]["outcome"] == "access_failed"
    assert transient == []
    provider.outage = True
    documents, transient = await capture(records, provider, captured_at="t")
    assert documents == {} and sorted(transient) == sorted(records)
    store = FrozenFullTextStore.from_document(build_store({RECORD: _documents()[RECORD]}),
                                              name="s")
    assert store.fetch(RECORD, served_by="corpus").retrieved
