"""The EuropePMC search provider (plan sections 8.4, 14.4, DEC-03).

EuropePMC's REST API is public and free — no credential, no procurement, no
per-call cost — which is exactly why it is this deployment's first evidence
provider: Phase 5 does not have to block on a provider contract or an API key
before a citation can exist at all. Only ``resultType=core`` is used, because
it is the one EuropePMC response shape that includes ``abstractText`` — the
lite shape used for casual browsing omits it, and this provider has no second
"detail" call (``research/interfaces.py``): whatever a search does not
include, ``get_evidence_record`` can never show either.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx

from ...config import ResearchSettings
from ...domain.errors import EvidenceUnavailable, ProviderRateLimited
from ...domain.evidence import SourceIdentifier, SourceType
from ..circuit_breaker import CircuitBreaker, CircuitOpen
from ..interfaces import SearchHit

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
        },
        raw=record,
    )


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
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.hard_timeout_s, connect=settings.timeout_s),
            transport=transport,
            headers={"accept": "application/json"},
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

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
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
            response = await self._client.request(method, path, **kwargs)
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
        if not response.is_success:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                f"research provider returned {response.status_code}",
                status=response.status_code,
            )
        self._circuit.record_success()
        return response

    def _parse_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise EvidenceUnavailable("the research provider returned a non-JSON response") from exc
