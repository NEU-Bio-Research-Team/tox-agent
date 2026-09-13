"""The instructions for the one LLM boundary in an orchestrated report (PR-12, PR-16).

The report-builder profile composes AGENTS.md, three references and five
skills: 25.9 kB of instructions, most of which tell a model how to gather
context, request explanations, run searches and repair a saved draft. Under the
orchestrator none of that is the model's work — the stage handlers have already
done it by the time anything is dispatched — so shipping those instructions to
the synthesis turn would spend tokens describing tools the turn cannot see.

This profile keeps only what synthesis needs: its own short brief, and the
shared reference on how a finding may be worded. The builder's source-hierarchy
reference is deliberately not shared: it explains the hierarchy in terms of the
builder's read tools (``get_or_create_explanation`` and friends), which this
turn cannot see, so the brief states the hierarchy in terms of the fact bundle
instead. The report-build profile is left untouched, because the old path still
runs whenever ``report_orchestrator_v2`` is off.

Composition and hashing follow ``report_profile``: same separator, same
per-file hashes, recorded in the build's instruction manifest.
"""
from __future__ import annotations

from pathlib import Path

from ..domain.provenance import content_sha256
from .report_profile import _SEPARATOR, ComposedProfile, ProfileUnavailable, _read

PROFILE_NAME = "report_synthesis"

#: Shared with the report-build profile rather than copied, so a wording rule
#: cannot be fixed in one and left stale in the other.
SHARED_REFERENCES: tuple[str, ...] = (
    "report_build/references/wording-and-safety-policy.md",
)


def compose_synthesis_profile(profiles_dir: Path) -> ComposedProfile:
    root = Path(profiles_dir)
    if not (root / PROFILE_NAME).is_dir():
        raise ProfileUnavailable(f"no {PROFILE_NAME} profile directory at {root / PROFILE_NAME}")
    hashes: dict[str, str] = {}
    parts = [_read(root, f"{PROFILE_NAME}/AGENTS.md", hashes)]
    for relative in SHARED_REFERENCES:
        parts.append(f"# Reference: {Path(relative).stem}\n\n{_read(root, relative, hashes)}")
    instructions = _SEPARATOR.join(parts)
    return ComposedProfile(
        profile=PROFILE_NAME,
        instructions=instructions,
        file_hashes=hashes,
        content_sha256=content_sha256(instructions),
        skills=(),
    )
