"""PubChem PUG REST as the compound-information provider (spec section 9).

Chosen for the same reason EuropePMC was chosen for literature: public, free,
no credential and no procurement, so identity resolution can exist before a
commercial database contract does. It resolves *by structure* — the canonical
SMILES the analysis snapshot recorded — so the record describes the molecule
that was actually predicted on, never one that merely shares a name.

The transport discipline is EuropePMC's, deliberately: startup host allowlist,
no redirects, a bounded response body, a content-type check and a circuit
breaker. A provider that can send unbounded bytes into this process is a
provider that decides how much memory a multi-tenant deployment uses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from ...config import CompoundSettings
from ...domain.errors import EvidenceUnavailable, ProviderRateLimited
from ..circuit_breaker import CircuitBreaker, CircuitOpen
from ..transport import rebuild_decoded_response, retry_transient
from ..compound import CompoundProperty, CompoundRecord

#: The PUG REST property table this provider asks for. A closed list: a report
#: may only cite properties the product has agreed to show, exactly as an
#: analysis slice may only expose declared fields.
_PROPERTY_FIELDS: tuple[tuple[str, str, str | None], ...] = (
    ("MolecularFormula", "molecular_formula", None),
    ("MolecularWeight", "molecular_weight", "g/mol"),
    ("XLogP", "xlogp", None),
    ("TPSA", "topological_polar_surface_area", "Å²"),
    ("HBondDonorCount", "hydrogen_bond_donors", None),
    ("HBondAcceptorCount", "hydrogen_bond_acceptors", None),
    ("InChIKey", "inchikey", None),
    ("Title", "title", None),
)

MAX_SYNONYMS = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PubChemCompoundProvider:
    """Resolves identity and bulk properties for one canonical SMILES."""

    name = "pubchem"

    def __init__(
        self, settings: CompoundSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        base_host = urlparse(settings.base_url).hostname or ""
        if base_host not in settings.allowed_hosts:
            raise ValueError(
                f"compound base_url host {base_host!r} is not in its own allowed_hosts "
                f"{settings.allowed_hosts!r} — this deployment is misconfigured"
            )
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.hard_timeout_s, connect=settings.timeout_s),
            transport=transport,
            headers={"accept": "application/json"},
            follow_redirects=False,
        )
        self._circuit = CircuitBreaker(
            failure_threshold=settings.circuit_failure_threshold,
            reset_after_s=settings.circuit_reset_after_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve(self, *, canonical_smiles: str) -> CompoundRecord:
        smiles = canonical_smiles.strip()
        if not smiles:
            return CompoundRecord.unresolved(
                provider=self.name, query_smiles=canonical_smiles,
                retrieved_at=_now(), reason="no canonical SMILES to resolve",
            )
        fields = ",".join(name for name, _, _ in _PROPERTY_FIELDS)
        path = f"/compound/smiles/{quote(smiles, safe='')}/property/{fields}/JSON"
        try:
            response = await self._request("GET", path)
        except EvidenceUnavailable:
            # Distinguished from "PubChem has no such compound": the report
            # gap reason differs, and so does whether a retry could help.
            raise
        if response.status_code == 404:
            return CompoundRecord.unresolved(
                provider=self.name, query_smiles=smiles, retrieved_at=_now(),
                reason="the compound database has no record for this structure",
            )
        body = self._parse_json(response)
        rows = ((body.get("PropertyTable") or {}).get("Properties")) or []
        if not rows:
            return CompoundRecord.unresolved(
                provider=self.name, query_smiles=smiles, retrieved_at=_now(),
                reason="the compound database returned no properties for this structure",
            )
        record = rows[0]
        cid = record.get("CID")
        synonyms = await self._synonyms(cid) if cid else ()
        properties = tuple(
            CompoundProperty(
                name=local, value=record[remote], unit=unit, source_field=remote
            )
            for remote, local, unit in _PROPERTY_FIELDS
            if remote in record and remote not in {"InChIKey", "Title"}
        )
        return CompoundRecord(
            provider=self.name,
            query_smiles=smiles,
            resolved=True,
            retrieved_at=_now(),
            # PubChem's ``Title`` is its preferred depositor-supplied name; the
            # first synonym is not, and is often a registry code.
            preferred_name=record.get("Title"),
            synonyms=synonyms,
            identifiers={
                "pubchem_cid": cid,
                "inchikey": record.get("InChIKey"),
                # PUG REST's property table carries no CAS number, and this
                # provider does not fabricate one from a synonym that merely
                # looks like one.
                "cas": None,
            },
            properties=properties,
            canonical_url=(
                f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None
            ),
            raw=record,
        )

    async def _synonyms(self, cid: Any) -> tuple[str, ...]:
        """Best-effort. A failure here loses a nice-to-have list, and must not
        turn a resolved identity into an unresolved one."""
        try:
            response = await self._request("GET", f"/compound/cid/{cid}/synonyms/JSON")
        except (EvidenceUnavailable, ProviderRateLimited):
            return ()
        if response.status_code != 200:
            return ()
        try:
            body = self._parse_json(response)
        except EvidenceUnavailable:
            return ()
        entries = ((body.get("InformationList") or {}).get("Information")) or []
        if not entries:
            return ()
        names = entries[0].get("Synonym") or []
        return tuple(str(n) for n in names[:MAX_SYNONYMS])

    # --- transport, mirroring the EuropePMC provider ----------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            self._circuit.before_call()
        except CircuitOpen as exc:
            raise EvidenceUnavailable(str(exc)) from exc
        try:
            response = await retry_transient(
                lambda: self._read_bounded(method, path, **kwargs),
                attempts=max(1, self._settings.retry_attempts),
                backoff_s=self._settings.retry_backoff_s,
                # Connect and read failures only. A body that arrived and did
                # not parse is a semantic answer, and retrying it would ask the
                # same question twice to get the same wrong shape back.
                retry_on=(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout),
                should_retry=lambda r: r.status_code >= 500,
            )
        except EvidenceUnavailable:
            self._circuit.record_failure()
            raise
        except httpx.ConnectError as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                f"cannot reach the compound provider at {self._settings.base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                "the compound provider did not answer within its budget"
            ) from exc
        except httpx.HTTPError as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(f"compound provider transport failure: {exc}") from exc
        if response.status_code == 429:
            self._circuit.record_failure()
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimited(
                "the compound provider rate-limited this request",
                retry_after_ms=int(float(retry_after) * 1000) if retry_after else None,
            )
        # 404 is a real answer ("no such compound"), not a provider failure,
        # and must not push the circuit towards open.
        if response.status_code >= 500:
            self._circuit.record_failure()
            raise EvidenceUnavailable(
                f"the compound provider answered {response.status_code}"
            )
        self._circuit.record_success()
        return response

    async def _read_bounded(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        limit = self._settings.max_response_bytes
        request = self._client.build_request(method, path, **kwargs)
        response = await self._client.send(request, stream=True)
        try:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise EvidenceUnavailable(
                        f"the compound provider sent more than {limit} bytes"
                    )
                chunks.append(chunk)
        finally:
            await response.aclose()
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
                f"the compound provider answered {media_type!r}, which is not JSON"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise EvidenceUnavailable(
                "the compound provider returned a non-JSON response"
            ) from exc
