"""A pinned local corpus as the evidence provider, for transfer benchmarks (W7-04).

``snapshot`` serves a fixed list of records per keyword: the fixture decides
what a query finds, so it cannot measure retrieval. A benchmark like SciFact
asks the opposite question — *given the whole corpus, does the product find and
judge the right abstracts?* — and for that the provider must rank, and the
product must write its own queries.

This provider ranks a local corpus with BM25 and nothing else: no network, no
learned model, no per-query configuration. Retrieval quality is therefore a
property of the query the product wrote, not of a re-ranker tuned against the
benchmark.

Deployment option only, never a default::

    TOXAGENT_RESEARCH_PROVIDER=corpus
    TOXAGENT_RESEARCH_CORPUS_PATH=/path/to/corpus.jsonl
    TOXAGENT_RESEARCH_CORPUS_SHA256=<sha256 of that file>

The pin is required. A benchmark corpus is usually licensed material that
cannot live in this repository (SciFact's abstracts are ODC-By), so the file is
built outside it, and without a pin nothing would tie a recorded number to the
corpus that produced it. The effective product records the provider, the corpus
name, its hash and its size, so a run over a local corpus can never be read as
a live literature run.

File format (``research-corpus-v1``), one JSON object per line:

``record_id`` (required, string or int), ``title``, ``sentences`` (list) or
``text`` (string), and optionally ``identifier`` (``doi``/``pmid``/``pmcid``/
``cid``/``other``), ``published_at`` (ISO date), ``source_type``. Unknown keys
are refused rather than ignored: a typo in the builder that silently dropped
the text would look like a retrieval failure.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from ...domain.evidence import SourceIdentifier, SourceType
from ..interfaces import SearchHit

#: Bumped when tokenisation or scoring changes: a recorded run says which.
RANKING_VERSION = "bm25-1"

#: Okapi BM25 with the usual defaults (Robertson & Zaragoza 2009, §3.1).
K1 = 1.2
B = 0.75

#: Enough of an abstract for a model to judge a claim against, bounded so one
#: corpus record cannot dominate a turn's context.
MAX_EXCERPT_CHARS = 4_000

_ALLOWED_KEYS = frozenset(
    {"record_id", "title", "sentences", "text", "identifier", "published_at", "source_type"}
)
_IDENTIFIER_KEYS = frozenset({"doi", "pmid", "pmcid", "cid", "other"})
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text or "").casefold())


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    record_id: str
    title: str
    text: str
    identifier: SourceIdentifier
    published_at: date | None
    source_type: SourceType
    term_frequencies: Counter
    length: int


def _document(entry: dict[str, Any], *, line_number: int) -> CorpusDocument:
    unknown = sorted(set(entry) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"corpus line {line_number}: unknown field(s) {unknown}")
    if "record_id" not in entry:
        raise ValueError(f"corpus line {line_number}: record_id is required")
    record_id = str(entry["record_id"]).strip()
    if not record_id:
        raise ValueError(f"corpus line {line_number}: record_id is empty")
    title = str(entry.get("title") or "").strip()
    if "sentences" in entry and "text" in entry:
        raise ValueError(f"corpus line {line_number}: give sentences or text, not both")
    if "sentences" in entry:
        sentences = entry["sentences"]
        if not isinstance(sentences, list) or any(not isinstance(s, str) for s in sentences):
            raise ValueError(f"corpus line {line_number}: sentences must be a list of strings")
        text = " ".join(s.strip() for s in sentences if s.strip())
    else:
        text = str(entry.get("text") or "").strip()
    # Both are required here rather than left to acceptance policy, which
    # rejects an untitled record silently (``research/policy.py``): a builder
    # that dropped the text would then look like a retrieval failure.
    if not title:
        raise ValueError(f"corpus line {line_number}: record {record_id} has no title")
    if not text:
        raise ValueError(f"corpus line {line_number}: record {record_id} has no text")
    raw_identifier = entry.get("identifier") or {}
    if not isinstance(raw_identifier, dict):
        raise ValueError(f"corpus line {line_number}: identifier must be an object")
    unknown_ids = sorted(set(raw_identifier) - _IDENTIFIER_KEYS)
    if unknown_ids:
        raise ValueError(f"corpus line {line_number}: unknown identifier field(s) {unknown_ids}")
    identifier = SourceIdentifier(
        **{k: str(v) for k, v in raw_identifier.items() if v},
    )
    published = entry.get("published_at")
    tokens = _tokens(f"{title} {text}")
    return CorpusDocument(
        record_id=record_id,
        title=title,
        text=text,
        identifier=identifier,
        published_at=date.fromisoformat(str(published)) if published else None,
        source_type=SourceType(entry.get("source_type") or SourceType.ARTICLE.value),
        term_frequencies=Counter(tokens),
        length=len(tokens),
    )


class CorpusResearchProvider:
    """BM25 over a pinned local corpus. Benchmarks only (module docstring)."""

    name = "corpus"

    def __init__(self, path: Path, *, sha256: str) -> None:
        path = Path(path)
        if not sha256.strip():
            raise ValueError(
                "TOXAGENT_RESEARCH_CORPUS_SHA256 is required for the corpus provider: a "
                "benchmark number is only readable next to the corpus hash that produced it"
            )
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != sha256.strip().lower():
            raise ValueError(
                f"corpus {path} has sha256 {digest}, not the pinned {sha256.strip().lower()}; "
                "refusing to serve a corpus that is not the recorded one"
            )
        documents: list[CorpusDocument] = []
        seen: set[str] = set()
        for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            document = _document(json.loads(line), line_number=number)
            if document.record_id in seen:
                raise ValueError(f"corpus line {number}: duplicate record_id {document.record_id}")
            seen.add(document.record_id)
            documents.append(document)
        if not documents:
            raise ValueError(f"corpus {path} has no records")
        self.corpus_name = path.name
        self.corpus_sha256 = digest
        self.corpus_size = len(documents)
        self._documents = documents
        self._average_length = sum(d.length for d in documents) / len(documents)
        document_frequency: Counter = Counter()
        for document in documents:
            document_frequency.update(document.term_frequencies.keys())
        total = len(documents)
        # Robertson/Sparck Jones IDF with the +0.5 smoothing BM25 uses; floored
        # at zero so a term in almost every document cannot subtract score.
        self._idf = {
            term: max(0.0, math.log(1 + (total - count + 0.5) / (count + 0.5)))
            for term, count in document_frequency.items()
        }

    def _score(self, document: CorpusDocument, query_terms: Sequence[str]) -> float:
        if not document.length:
            return 0.0
        score = 0.0
        normaliser = K1 * (1 - B + B * document.length / self._average_length)
        for term in query_terms:
            frequency = document.term_frequencies.get(term, 0)
            if not frequency:
                continue
            score += self._idf.get(term, 0.0) * frequency * (K1 + 1) / (frequency + normaliser)
        return score

    async def search(
        self,
        *,
        query: str,
        source_types: Sequence[str] | None,
        date_from: date | None,
        limit: int,
    ) -> list[SearchHit]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        candidates = self._documents
        if source_types:
            allowed = set(source_types)
            candidates = [d for d in candidates if d.source_type.value in allowed]
        if date_from:
            candidates = [
                d for d in candidates if d.published_at is None or d.published_at >= date_from
            ]
        scored = [(self._score(d, query_terms), d) for d in candidates]
        # Ranked by score, then by record id: two documents that score the same
        # must come back in the same order on every run, or a rerun of the same
        # benchmark would measure the dictionary order of a dict.
        ranked = sorted(
            (pair for pair in scored if pair[0] > 0),
            key=lambda pair: (-pair[0], pair[1].record_id),
        )[: max(1, limit)]
        return [
            SearchHit(
                provider_record_id=document.record_id,
                source_type=document.source_type,
                title=document.title,
                published_at=document.published_at,
                identifier=document.identifier,
                abstract_or_excerpt=document.text[:MAX_EXCERPT_CHARS] or None,
                normalized_facts={
                    "corpus": self.corpus_name,
                    "corpus_record_id": document.record_id,
                    "retrieval": RANKING_VERSION,
                    "rank": rank,
                    "score": round(score, 4),
                },
                raw={"corpus_record_id": document.record_id, "corpus_sha256": self.corpus_sha256},
            )
            for rank, (score, document) in enumerate(ranked, 1)
        ]

    async def aclose(self) -> None:
        return None
