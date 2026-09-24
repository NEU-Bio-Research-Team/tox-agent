"""Pinned, resumable client for the ChEMBL 37 REST API.

The bioactivity data contract (`toxact-chembl37-hq-v1`) pins ChEMBL release 37.
We read through the REST API rather than the SQLite dump because the dump does
not fit the available disk budget, and because the API reports the release it is
serving -- so we can *assert* the pin instead of trusting a filename.

Every request carries the full HQ-Exact filter set server-side, which both cuts
transfer volume roughly in half and keeps the filter definition in exactly one
place. Nothing here interprets or aggregates: this module only fetches and
records provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import requests

LOG = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PINNED_RELEASE = "ChEMBL_37"
PINNED_RELEASE_DATE = "2026-05-01"

#: Potency/affinity types kept as *separate* endpoints. They are never collapsed
#: into a single label -- see the data contract, section 3.3.
ALLOWED_STANDARD_TYPES = ("Ki", "Kd", "IC50", "EC50")

#: Server-side HQ-Exact predicates. `target_type == SINGLE PROTEIN` cannot be
#: expressed on the activity endpoint, so it is enforced by restricting the
#: target panel to single-protein targets at selection time instead.
HQ_EXACT_FILTERS: dict[str, Any] = {
    "standard_relation": "=",
    "standard_units": "nM",
    "pchembl_value__isnull": "false",
    "potential_duplicate": 0,
    "assay_confidence_score": 9,
    "target_organism": "Homo sapiens",
}

#: Fields pulled for each activity row. Anything used by standardization,
#: aggregation, splitting or provenance must be listed here.
ACTIVITY_FIELDS = (
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "target_chembl_id",
    "target_organism",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "assay_variant_mutation",
    "bao_label",
    "standard_type",
    "standard_relation",
    "standard_units",
    "standard_value",
    "pchembl_value",
    "document_chembl_id",
    "document_year",
    "potential_duplicate",
    "data_validity_comment",
)

PAGE_LIMIT = 1000


class ChemblApiError(RuntimeError):
    """Raised when the API cannot satisfy a request after retries."""


class ReleaseMismatchError(RuntimeError):
    """Raised when the live API is not serving the pinned release."""


@dataclass
class RequestStats:
    """Counts requests so a run can report its own cost."""

    requests: int = 0
    retries: int = 0
    bytes_down: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, n_bytes: int, retried: int = 0) -> None:
        with self._lock:
            self.requests += 1
            self.retries += retried
            self.bytes_down += n_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "bytes_down": self.bytes_down,
        }


class ChemblClient:
    """Thread-safe ChEMBL REST reader with retry and release pinning."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        timeout: int = 120,
        max_retries: int = 5,
        backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.stats = RequestStats()
        self._local = threading.local()

    # -- plumbing ---------------------------------------------------------

    @property
    def _session(self) -> requests.Session:
        """One session per thread; `requests.Session` is not thread-safe."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Accept": "application/json"})
            self._local.session = session
        return session

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET one page, retrying on throttling and transient server errors."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}.json"
        retried = 0
        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retried += 1
                time.sleep(self.backoff * (attempt + 1))
                continue

            if response.status_code == 200:
                self.stats.record(n_bytes=len(response.content), retried=retried)
                return response.json()

            # 429 is throttling; 5xx is transient. Both are worth waiting out.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                retried += 1
                wait = self.backoff * (attempt + 1)
                if response.status_code == 429:
                    wait = max(wait, float(response.headers.get("Retry-After", wait)))
                LOG.debug("throttled on %s (%s), waiting %.1fs", endpoint, last_error, wait)
                time.sleep(wait)
                continue

            # 4xx other than 429 means the query itself is wrong: do not retry.
            raise ChemblApiError(
                f"{url} -> HTTP {response.status_code}: {response.text[:300]}"
            )

        raise ChemblApiError(f"{url} failed after {self.max_retries} attempts: {last_error}")

    # -- release pin ------------------------------------------------------

    def assert_release(self, expected: str = PINNED_RELEASE) -> dict[str, Any]:
        """Fail loudly if the API has moved past the pinned release.

        A silent release bump would change the dataset underneath a frozen split
        manifest, so this is an assertion rather than a warning.
        """
        status = self.get("status", {})
        actual = status.get("chembl_db_version")
        if actual != expected:
            raise ReleaseMismatchError(
                f"data contract pins {expected} but the API is serving {actual}. "
                "Either re-pin the contract deliberately or use an archived dump."
            )
        return status

    # -- counting and paging ----------------------------------------------

    def count(self, endpoint: str, params: dict[str, Any]) -> int:
        """Return `total_count` without transferring the rows themselves."""
        probe = dict(params)
        probe.update({"limit": 1, "only": "activity_id" if endpoint == "activity" else None})
        probe = {k: v for k, v in probe.items() if v is not None}
        return int(self.get(endpoint, probe)["page_meta"]["total_count"])

    def iter_pages(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        fields: Sequence[str] | None = None,
        page_limit: int = PAGE_LIMIT,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield successive record pages, following offset pagination.

        ChEMBL's `page_meta.next` is followed by offset rather than by URL so
        that the full parameter set (and therefore the filter definition) stays
        explicit in every request.
        """
        collection = _collection_key(endpoint)
        base = dict(params)
        base["limit"] = page_limit
        if fields:
            base["only"] = ",".join(fields)

        offset = 0
        total: int | None = None
        while True:
            page = dict(base)
            page["offset"] = offset
            payload = self.get(endpoint, page)
            records = payload.get(collection, [])
            if total is None:
                total = int(payload["page_meta"]["total_count"])
            if not records:
                break
            yield records
            offset += len(records)
            if offset >= total:
                break

    def fetch_all(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        fields: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in self.iter_pages(endpoint, params, fields=fields):
            out.extend(page)
        return out


def _collection_key(endpoint: str) -> str:
    """Map an endpoint name to the JSON key holding its records."""
    special = {"activity": "activities", "molecule": "molecules"}
    name = endpoint.strip("/")
    return special.get(name, f"{name}s")


def hq_exact_activity_params(
    *,
    target_chembl_id: str | None = None,
    standard_type: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build activity-endpoint params carrying the full HQ-Exact filter set."""
    params: dict[str, Any] = dict(HQ_EXACT_FILTERS)
    if target_chembl_id is not None:
        params["target_chembl_id"] = target_chembl_id
    if standard_type is None:
        params["standard_type__in"] = ",".join(ALLOWED_STANDARD_TYPES)
    elif isinstance(standard_type, str):
        params["standard_type"] = standard_type
    else:
        params["standard_type__in"] = ",".join(standard_type)
    return params


def filter_definition_sha256() -> str:
    """Content hash of the filter definition, for the dataset manifest.

    If anyone edits the filters or the field list, this hash changes and the
    resulting dataset is a new version -- which is exactly the intent.
    """
    payload = json.dumps(
        {
            "release": PINNED_RELEASE,
            "filters": HQ_EXACT_FILTERS,
            "standard_types": list(ALLOWED_STANDARD_TYPES),
            "fields": list(ACTIVITY_FIELDS),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
