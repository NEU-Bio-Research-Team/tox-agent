"""Deployment-selected research provider (DEC-03).

A separate factory, not a branch inside ``api/app.py``, so adding a second
provider later is a one-line addition here rather than a change to the
composition root's control flow.
"""
from __future__ import annotations

from pathlib import Path

from ...platform.config import CompoundSettings, ResearchSettings
from ..compound import CompoundProvider
from ..interfaces import ResearchProvider
from .europepmc import EuropePmcProvider
from .pubchem import PubChemCompoundProvider


def build_provider(settings: ResearchSettings) -> ResearchProvider | None:
    """``None`` when no provider is configured — the deployment simply does
    not register the evidence tools, rather than registering ones that would
    always fail (plan section 8.1)."""
    if not settings.provider:
        return None
    if settings.provider == "europepmc":
        return EuropePmcProvider(settings)
    if settings.provider == "snapshot":
        # Frozen evidence for benchmarks (research/providers/snapshot.py).
        from .snapshot import SnapshotResearchProvider

        if not settings.snapshot_path:
            raise ValueError("TOXAGENT_RESEARCH_SNAPSHOT_PATH is required for the snapshot provider")
        return SnapshotResearchProvider(
            Path(settings.snapshot_path), fault=settings.snapshot_fault
        )
    if settings.provider == "corpus":
        # A pinned local corpus, ranked (research/providers/corpus.py): the
        # transfer benchmarks where the product must write its own queries.
        from .corpus import CorpusResearchProvider

        if not settings.corpus_path:
            raise ValueError("TOXAGENT_RESEARCH_CORPUS_PATH is required for the corpus provider")
        return CorpusResearchProvider(
            Path(settings.corpus_path), sha256=settings.corpus_sha256
        )
    raise ValueError(f"unknown research provider: {settings.provider!r}")


def build_compound_provider(settings: CompoundSettings) -> CompoundProvider | None:
    """The substance-information provider (report spec section 9).

    Same contract as ``build_provider``: ``None`` when nothing is configured,
    so ``resolve_compound_record`` is simply not registered and a report
    records an identity gap rather than calling a tool that could only fail.
    """
    if not settings.provider:
        return None
    if settings.provider == "pubchem":
        return PubChemCompoundProvider(settings)
    raise ValueError(f"unknown compound provider: {settings.provider!r}")
