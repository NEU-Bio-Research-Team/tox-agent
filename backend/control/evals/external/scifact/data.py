"""SciFact release data: fetched at run time, pinned by hash, never committed.

Claims and annotations are CC BY 4.0; the corpus abstracts are from S2ORC under
ODC-By 1.0 (see https://github.com/allenai/scifact, LICENSE.md). The archive is
downloaded to a cache directory and refused if its SHA-256 differs from the one
recorded when this adapter was written, so a silently changed release cannot
produce a number that looks comparable.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
RELEASE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DEFAULT_CACHE = Path.home() / ".cache" / "toxagent-evals" / "scifact"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    id: int
    claim: str
    #: doc_id -> {"label", "rationales"}; empty for NEI claims and on the test split.
    evidence: dict[int, dict[str, Any]]
    cited_doc_ids: tuple[int, ...]


def ensure_release(cache: Path = DEFAULT_CACHE) -> Path:
    data_dir = cache / "data"
    if (data_dir / "corpus.jsonl").exists():
        return data_dir
    import httpx

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "data.tar.gz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != RELEASE_SHA256:
        partial = cache / "data.tar.gz.part"
        last_error: Exception | None = None
        for _ in range(3):
            try:
                # A stalled read is a failed attempt, not a hang: 30 s per read.
                with httpx.stream("GET", RELEASE_URL, follow_redirects=True,
                                  timeout=httpx.Timeout(30.0, connect=15.0)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                partial.replace(archive)
                break
            except httpx.HTTPError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not download the SciFact release: {last_error}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != RELEASE_SHA256:
        archive.unlink()
        raise RuntimeError(f"SciFact release hash {digest} is not the pinned {RELEASE_SHA256}")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(cache, filter="data")
        else:  # pragma: no cover - Python without PEP 706; the hash is already pinned
            tar.extractall(cache)
    return data_dir


def _gold_evidence(raw: dict[str, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    evidence: dict[int, dict[str, Any]] = {}
    for doc_id, rationales in raw.items():
        labels = {r["label"] for r in rationales}
        if len(labels) != 1:  # the release has one label per claim/abstract pair
            raise ValueError(f"abstract {doc_id} carries several labels")
        evidence[int(doc_id)] = {"label": labels.pop(),
                                 "rationales": [list(r["sentences"]) for r in rationales]}
    return evidence


def load_corpus(data_dir: Path) -> dict[int, Document]:
    corpus: dict[int, Document] = {}
    for line in (data_dir / "corpus.jsonl").read_text().splitlines():
        entry = json.loads(line)
        corpus[int(entry["doc_id"])] = Document(int(entry["doc_id"]), entry["title"],
                                                tuple(entry["abstract"]))
    return corpus


def load_claims(data_dir: Path, split: str) -> list[Claim]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    claims = []
    for line in (data_dir / f"claims_{split}.jsonl").read_text().splitlines():
        entry = json.loads(line)
        claims.append(Claim(
            id=int(entry["id"]), claim=entry["claim"],
            evidence=_gold_evidence(entry.get("evidence") or {}),
            cited_doc_ids=tuple(int(d) for d in entry.get("cited_doc_ids") or ()),
        ))
    return sorted(claims, key=lambda c: c.id)


def gold_of(claims: list[Claim]) -> dict[int, dict[int, dict[str, Any]]]:
    return {claim.id: claim.evidence for claim in claims}
