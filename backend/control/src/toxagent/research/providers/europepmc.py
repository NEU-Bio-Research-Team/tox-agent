"""The EuropePMC search provider (plan sections 8.4, 14.4, DEC-03).

EuropePMC's REST API is public and free — no credential, no procurement, no
per-call cost — which is exactly why it is this deployment's first evidence
provider: Phase 5 does not have to block on a provider contract or an API key
before a citation can exist at all. Only ``resultType=core`` is used, because
it is the one EuropePMC response shape that includes ``abstractText`` — the
lite shape used for casual browsing omits it. The only second call is
``fetch_full_text`` (issue #43), for a record whose text Europe PMC holds;
anything else a search does not include, ``get_evidence_record`` can never
show either.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from datetime import date, datetime
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx

from ...platform.config import ResearchSettings
from ...domain.errors import EvidenceUnavailable, ProviderRateLimited
from ...domain.evidence import SourceIdentifier, SourceType
from ..circuit_breaker import CircuitBreaker, CircuitOpen
from ..fulltext import PARSER_VERSION, parse_jats
from ..transport import rebuild_decoded_response
from ..interfaces import FullTextFetch, SearchHit
from ..text_access import access_facts

#: EuropePMC's own source codes for a peer-reviewed-journal record space
#: (MED = MEDLINE, PMC = PubMed Central full text); everything else this API
#: can return (preprints, patents, agricultural literature, ...) still maps to
#: ``SourceType.ARTICLE`` here — EuropePMC is a literature search engine, so a
#: hit from it is never ``database``/``regulatory``/``vendor_documentation``.
_ARTICLE_SOURCE = SourceType.ARTICLE


def _authors(record: dict[str, Any]) -> tuple[str, ...]:
    # EuropePMC's authorString always ends the last name with a trailing
    # period ("... Tang Y."); stripped once from the whole string rather than
    # per-name so a genuine initial-with-period in the middle is untouched.
    raw = (record.get("authorString") or "").rstrip(".")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _published_at(record: dict[str, Any]) -> date | None:
    first_pub = record.get("firstPublicationDate")
    if first_pub:
        try:
            return datetime.strptime(first_pub, "%Y-%m-%d").date()
        except ValueError:
            pass
    year = record.get("pubYear")
    if isinstance(year, str) and year.isdigit():
        return date(int(year), 1, 1)
    return None


def _journal_title(record: dict[str, Any]) -> str | None:
    journal_info = record.get("journalInfo") or {}
    journal = journal_info.get("journal") or {}
    return journal.get("title") or None


def _pub_type(record: dict[str, Any]) -> str | None:
    types = (record.get("pubTypeList") or {}).get("pubType") or []
    return ", ".join(types) if types else None


def hit_from_record(record: dict[str, Any]) -> SearchHit | None:
    """``None`` for a record missing what a citation needs — normalization
    never has to guess whether a hit was malformed or genuinely titleless."""
    title = record.get("title")
    provider_record_id = record.get("id")
    source = record.get("source")
    if not title or not provider_record_id or not source:
        return None
    canonical_url = f"https://europepmc.org/article/{source}/{provider_record_id}"
    return SearchHit(
        provider_record_id=f"{source}:{provider_record_id}",
        source_type=_ARTICLE_SOURCE,
        title=title,
        authors=_authors(record),
        published_at=_published_at(record),
        canonical_url=canonical_url,
        identifier=SourceIdentifier(
            doi=record.get("doi"), pmid=record.get("pmid"), pmcid=record.get("pmcid"),
        ),
        abstract_or_excerpt=record.get("abstractText"),
        normalized_facts={
            "journal": _journal_title(record),
            "pub_type": _pub_type(record),
            "pub_year": record.get("pubYear"),
            # Whether the text can be read at all, derived from what the
            # provider declared rather than assumed (research/text_access.py).
            # An abstract-only hit now says so, so a search that found
            # abstracts is never reported as an absence of evidence.
            **access_facts(record, has_abstract=bool(record.get("abstractText"))),
        },
        raw=record,
    )


#: What a PMCID looks like. The value comes from provider content and goes
#: into a request path, so anything else is refused rather than escaped.
_PMCID = re.compile(r"^PMC[0-9]{1,10}$")


def full_text_id(
    provider_record_id: str, identifier: SourceIdentifier | None
) -> str | None:
    """The PMCID Europe PMC holds this record's text under, if it has one.

    A ``PMC:`` record's own id is its PMCID. Any other record (``MED:``,
    ``PPR:`` ...) can still have text deposited in PMC, and then names the
    PMCID in its ``pmcid`` field -- which the stored record keeps as
    ``identifier.pmcid``.
    """
    source, _, own_id = provider_record_id.partition(":")
    candidates = [own_id] if source.upper() == "PMC" else []
    if identifier is not None and identifier.pmcid:
        candidates.append(identifier.pmcid)
    for candidate in candidates:
        normalised = candidate.strip().upper()
        if normalised.isdigit():
            normalised = f"PMC{normalised}"
        if _PMCID.match(normalised):
            return normalised
    return None


def _query_with_filters(query: str, date_from: date | None) -> str:
    if date_from is None:
        return query
    return f"({query}) AND FIRST_PDATE:[{date_from.isoformat()} TO 3000-01-01]"


class EuropePmcProvider:
    """Plan section 8.1: the provider only searches; ``research/policy.py``
    decides what is citable, so this class carries no acceptance logic."""

    name = "europepmc"

    def __init__(
        self, settings: ResearchSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        # A per-response host check would be theatre: httpx does not follow
        # redirects by default, so a request's response always resolves to
        # the host in ``base_url`` — a deployment fact, not something a
        # provider response can vary per call. Catching a misconfigured
        # ``base_url`` here, once, at startup, is the check that can actually
        # fail. ``canonical_url`` (which *does* vary per record, and is
        # provider-supplied content) is still checked per record in
        # ``research/policy.py``.
        base_host = urlparse(settings.base_url).hostname or ""
        if base_host not in settings.allowed_hosts:
            raise ValueError(
                f"research base_url host {base_host!r} is not in its own allowed_hosts "
                f"{settings.allowed_hosts!r} — this deployment is misconfigured"
            )
        self._settings = settings
        #: #43: definitive text answers by PMCID, bounded, so concurrent
        #: sessions reading one paper ask Europe PMC once (single-flight) and
        #: a later session reuses the parse. Never a transient failure, which
        #: says nothing about the document.
        self._text_cache: "OrderedDict[str, FullTextFetch]" = OrderedDict()
        self._text_inflight: dict[str, asyncio.Future] = {}
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.hard_timeout_s, connect=settings.timeout_s),
            transport=transport,
            headers={"accept": "application/json"},
            # Stated rather than inherited. It is httpx's default, and the
            # host allowlist above is checked once at startup against
            # `base_url` — which is only equivalent to checking every response
            # while no redirect can move the request somewhere else.
            follow_redirects=False,
        )
        self._circuit = CircuitBreaker(
            failure_threshold=settings.circuit_failure_threshold,
            reset_after_s=settings.circuit_reset_after_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        query: str,
        source_types: Sequence[str] | None = None,
        date_from: date | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        stripped = query.strip()
        if not stripped:
            return []
        params: dict[str, str] = {
            "query": _query_with_filters(stripped, date_from),
            "format": "json",
            "resultType": "core",
            "pageSize": str(max(1, min(limit, self._settings.max_results))),
        }
        if self._settings.contact_email:
            params["email"] = self._settings.contact_email
        response = await self._request("GET", "/search", params=params)
        body = self._parse_json(response)
        records = ((body.get("resultList") or {}).get("result")) or []
        hits = [hit_from_record(r) for r in records]
        return [h for h in hits if h is not None]

    async def fetch_full_text(
        self, *, provider_record_id: str, identifier: SourceIdentifier | None = None,
    ) -> FullTextFetch:
        """Ask for this record's text, as located passages.

        Europe PMC serves full text at ``/{PMCID}/fullTextXML`` and nowhere
        else; ``/{source}/{id}/fullTextXML`` answers 404 for every record
        (checked against the live API on 2026-10-21), so asking under the
        record's own ``MED:<pmid>`` id reported every paper as having no text.
        The PMCID comes from the record itself (:func:`full_text_id`), and a
        record that names none is reported as such without a request.

        A record with no full text in Europe PMC answers 404. That is an
        ordinary, expected answer about one document, so it is reported as a
        failed access rather than raised -- and it must not count against the
        circuit breaker, which exists for a provider that is unwell.
        """
        source, _, own_id = provider_record_id.partition(":")
        if not source or not own_id:
            return FullTextFetch(
                retrieved=False,
                reason=f"{provider_record_id!r} is not a '<source>:<id>' record id",
            )
        pmcid = full_text_id(provider_record_id, identifier)
        if pmcid is None:
            return FullTextFetch(
                retrieved=False,
                reason=(
                    "the record names no PMCID, and Europe PMC serves full text only "
                    "by PMCID"
                ),
                provenance=self._text_provenance(None),
            )
        cached = self._text_cache.get(pmcid)
        if cached is not None:
            self._text_cache.move_to_end(pmcid)
            return cached
        pending = self._text_inflight.get(pmcid)
        if pending is not None:
            return await asyncio.shield(pending)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._text_inflight[pmcid] = future
        try:
            fetched = await self._fetch_text(pmcid)
        except Exception as exc:
            future.set_exception(exc)
            # Marked retrieved: with no other waiter it is not a lost error.
            future.exception()
            raise
        except BaseException:
            future.cancel()
            raise
        else:
            future.set_result(fetched)
            if not fetched.transient:
                self._text_cache[pmcid] = fetched
                while len(self._text_cache) > self.TEXT_CACHE_SIZE:
                    self._text_cache.popitem(last=False)
            return fetched
        finally:
            self._text_inflight.pop(pmcid, None)

    #: Parsed documents kept per provider instance.
    TEXT_CACHE_SIZE = 64

    @staticmethod
    def _text_provenance(source: bytes | None) -> dict[str, Any]:
        return {
            "served_by": "europepmc", "replay": False, "parser_version": PARSER_VERSION,
            "source_sha256": hashlib.sha256(source).hexdigest() if source is not None else None,
        }

    async def _fetch_text(self, pmcid: str) -> FullTextFetch:
        try:
            # The client asks for JSON by default (the search API), and this
            # route answers that with 406 Not Acceptable: found by the first
            # live run, 2026-10-21, after every mocked test had passed.
            response = await self._request(
                "GET", f"/{pmcid}/fullTextXML", not_found_ok=True,
                headers={"accept": "application/xml"},
            )
        except (EvidenceUnavailable, ProviderRateLimited) as exc:
            return FullTextFetch(
                retrieved=False, reason=str(exc), full_text_id=pmcid, transient=True,
                provenance=self._text_provenance(None),
            )
        if response.status_code == 404:
            return FullTextFetch(
                retrieved=False,
                reason=f"the provider holds no full text for {pmcid}",
                full_text_id=pmcid, provenance=self._text_provenance(None),
            )
        provenance = self._text_provenance(response.content)
        media_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if media_type and media_type not in self._settings.allowed_full_text_content_types:
            return FullTextFetch(
                retrieved=False,
                reason=f"the provider answered {media_type!r}, which is not XML",
                full_text_id=pmcid, provenance=provenance,
            )
        document = parse_jats(response.text)
        if not document.passages:
            return FullTextFetch(
                retrieved=False,
                reason="the provider returned a document with no readable passages",
                full_text_id=pmcid, provenance=provenance,
            )
        return FullTextFetch(
            retrieved=True,
            passages=document.passages,
            section_titles=document.section_titles,
            truncated=document.truncated,
            omitted=dict(document.omitted),
            full_text_id=pmcid,
            provenance=provenance,
        )

    async def _request(
        self, method: str, path: str, *, not_found_ok: bool = False, **kwargs: Any
    ) -> httpx.Response:
        # remaining-plan W3-06: fail fast, without even attempting the
        # network call, once the provider has failed enough consecutive
        # times in a row — never pummel a struggling provider with more
        # requests while it is down, and never make every caller pay a full
        # connect/read timeout to discover that on their own.
        try:
            self._circuit.before_call()
        except CircuitOpen as exc:
            raise EvidenceUnavailable(str(exc)) from exc
        try:
            response = await self._read_bounded(method, path, **kwargs)
        except EvidenceUnavailable:
            self._circuit.record_failure()
            raise
        except httpx.ConnectError as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                f"cannot reach the research provider at {self._settings.base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable("the research provider did not answer within its budget") from exc
        except httpx.HTTPError as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(f"research provider transport failure: {exc}") from exc
        if response.status_code == 429:
            self._circuit.record_failure()
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimited(
                "the research provider rate-limited this request",
                retry_after_ms=int(float(retry_after) * 1000) if retry_after else None,
            )
        # A 404 for one record is the provider answering correctly about a
        # document it does not hold, so the caller interprets it and the
        # breaker counts it as a healthy call.
        if response.status_code == 404 and not_found_ok:
            self._circuit.record_success()
            return response
        if not response.is_success:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                f"research provider returned {response.status_code}",
                status=response.status_code,
            )
        self._circuit.record_success()
        return response

    async def _read_bounded(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Stream the body, stopping at the cap rather than after it.

        A timeout bounds how long a remote host may take. Nothing bounded how
        much it could send, and `response.json()` materialises the whole body
        first — so a provider having a bad day, or anything able to answer as
        one, decided how much of a multi-tenant control plane's memory to use.
        Checking `content-length` alone would not do it: it is optional, and a
        chunked response has none.
        """
        limit = self._settings.max_response_bytes
        request = self._client.build_request(method, path, **kwargs)
        response = await self._client.send(request, stream=True)
        try:
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise EvidenceUnavailable(
                    f"the research provider announced {declared} bytes, over the "
                    f"{limit}-byte limit"
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise EvidenceUnavailable(
                        f"the research provider sent more than {limit} bytes"
                    )
                chunks.append(chunk)
        finally:
            await response.aclose()
        # Rebuilt as a non-streaming response so everything downstream — the
        # status checks, `retry-after`, `.json()` — is unchanged.
        # The headers must describe the bytes actually held. ``aiter_bytes``
        # already decoded any Content-Encoding, so carrying the original
        # header over would make the next ``.json()`` try to decompress plain
        # JSON — the decompression failure the 2026-09-13 audit hit against
        # PubChem (P2-5).
        return rebuild_decoded_response(
            response=response, body=b"".join(chunks), request=request
        )

    def _parse_json(self, response: httpx.Response) -> dict[str, Any]:
        media_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if media_type and media_type not in self._settings.allowed_content_types:
            raise EvidenceUnavailable(
                f"the research provider answered {media_type!r}, which is not JSON"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise EvidenceUnavailable("the research provider returned a non-JSON response") from exc
