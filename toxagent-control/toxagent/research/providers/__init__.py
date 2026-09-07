"""Deployment-selected research provider (DEC-03).

A separate factory, not a branch inside ``api/app.py``, so adding a second
provider later is a one-line addition here rather than a change to the
composition root's control flow.
"""
from __future__ import annotations

from ...config import ResearchSettings
from ..interfaces import ResearchProvider
from .europepmc import EuropePmcProvider


def build_provider(settings: ResearchSettings) -> ResearchProvider | None:
    """``None`` when no provider is configured — the deployment simply does
    not register the evidence tools, rather than registering ones that would
    always fail (plan section 8.1)."""
    if not settings.provider:
        return None
    if settings.provider == "europepmc":
        return EuropePmcProvider(settings)
    raise ValueError(f"unknown research provider: {settings.provider!r}")
