"""Access to the sanitized 2026-09-13 audit captures.

One loader, so a fixture's path is written once and a renamed file fails
loudly in every test that reads it rather than in one.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "audit_2026_09_13"

DUPLICATE_USAGE_SSE = "opencode_duplicate_usage_sse.json"
ETHANOL_HERG_FALSE_MATCHES = "evidence_ethanol_herg_false_matches.json"
REPORT_CONTRADICTION = "report_contradiction_artifact.json"
CCO_ATTRIBUTION = "explanation_cco_herg_attribution.json"
BASELINE_MANIFEST = "baseline_manifest.json"

ALL_FIXTURES = (
    DUPLICATE_USAGE_SSE,
    ETHANOL_HERG_FALSE_MATCHES,
    REPORT_CONTRADICTION,
    CCO_ATTRIBUTION,
    BASELINE_MANIFEST,
)


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = FIXTURE_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            f"audit fixture {name!r} is missing from {FIXTURE_ROOT}; the findings "
            "it reproduces have no regression guard without it"
        )
    return path.read_text(encoding="utf-8")


def load(name: str) -> Any:
    """The fixture's parsed contents. A fresh object each call, so a test that
    mutates what it loaded cannot leak into the next one."""
    return json.loads(_read(name))
