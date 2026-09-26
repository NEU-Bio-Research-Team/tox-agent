"""Capability profiles from a validated manifest (RETHINK §4.10, W9-09).

Which tools a profile may see used to be a Python literal in
``tools/registry.py``, so even a permission change was a code change. It is now
``agent_profiles/tool_profiles.json``: data, reviewed like code, and refused at
start-up unless it passes the checks below. ``tools.registry.PROFILES`` and
``FLAG_GATED_TOOLS`` are derived from it, so every existing consumer (the
registry's own claim check, capability tokens, the effective product, the
capability matrix, the skill catalog's tool check) reads the manifest.

A manifest cannot create a capability. A tool it names still has to be
registered by the bootstrap from code; a tool the deployment does not register
is simply absent, exactly as before. What the manifest decides is which of the
registered tools a profile exposes.

The checks encode rules the product already relied on, stated once:

* profile and tool names are identifiers, and no profile lists a tool twice;
* a flag-gated tool names a declared rollout flag and appears in some profile;
* **one way to finish.** A profile holds at most one family of finishing tools
  (a grounded answer, a report draft, a report synthesis) — two ways to finish
  a run would mean two validators and two things a transcript could call the
  result;
* a ``read_only`` profile holds no finishing tool and no state-writing tool.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..domain.provenance import content_sha256

SCHEMA_VERSION = "tool-profiles-v1"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "agent_profiles" / "tool_profiles.json"

_PROFILE_NAME = re.compile(r"^[a-z][a-z_]{1,40}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

#: Tools that end a run, by family. A profile may hold one family at most.
FINISHING_FAMILIES: Mapping[str, frozenset[str]] = {
    "grounded_answer": frozenset({"submit_grounded_answer"}),
    "report_draft": frozenset({"submit_report_draft", "submit_saved_report_draft"}),
    "report_synthesis": frozenset({"submit_report_synthesis"}),
}

#: Tools that change product state without finishing a run.
STATE_WRITERS = frozenset({
    "create_analysis_snapshot", "update_scientific_case", "record_decision_plan",
    "save_report_draft", "patch_saved_report_draft", "get_or_create_explanation",
})


class ProfileManifestError(ValueError):
    """The manifest is refused; the message names the rule it breaks."""


@dataclass(frozen=True)
class ProfileManifest:
    profiles: Mapping[str, frozenset[str]]
    descriptions: Mapping[str, str]
    read_only: frozenset[str]
    flag_gated_tools: Mapping[str, str]
    content_sha256: str

    def summary(self) -> dict[str, Any]:
        """What the effective product records about the manifest in force."""
        return {"schema_version": SCHEMA_VERSION, "content_sha256": self.content_sha256}


def parse(document: Mapping[str, Any], *, known_flags: frozenset[str]) -> ProfileManifest:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ProfileManifestError(
            f"schema_version must be {SCHEMA_VERSION!r}; got {document.get('schema_version')!r}"
        )
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ProfileManifestError("profiles must be a non-empty object")

    profiles: dict[str, frozenset[str]] = {}
    descriptions: dict[str, str] = {}
    read_only: set[str] = set()
    for name, spec in raw_profiles.items():
        if not _PROFILE_NAME.match(str(name)):
            raise ProfileManifestError(f"profile name {name!r} is not an identifier")
        if not isinstance(spec, Mapping):
            raise ProfileManifestError(f"profile {name!r} must be an object")
        description = spec.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ProfileManifestError(f"profile {name!r} needs a description of what it is for")
        tools = spec.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ProfileManifestError(f"profile {name!r} must list at least one tool")
        for tool in tools:
            if not isinstance(tool, str) or not _TOOL_NAME.match(tool):
                raise ProfileManifestError(f"profile {name!r} names {tool!r}, not a tool identifier")
        duplicates = sorted({tool for tool in tools if tools.count(tool) > 1})
        if duplicates:
            raise ProfileManifestError(f"profile {name!r} lists {duplicates} more than once")
        toolset = frozenset(tools)
        families = [family for family, members in FINISHING_FAMILIES.items() if toolset & members]
        if len(families) > 1:
            raise ProfileManifestError(
                f"profile {name!r} can finish a run in {len(families)} ways ({families}); "
                "a profile has one way to finish"
            )
        if spec.get("read_only", False) is True:
            writes = sorted(
                toolset & (STATE_WRITERS | frozenset().union(*FINISHING_FAMILIES.values()))
            )
            if writes:
                raise ProfileManifestError(
                    f"read-only profile {name!r} lists tools that write or finish: {writes}"
                )
            read_only.add(name)
        elif spec.get("read_only", False) is not False:
            raise ProfileManifestError(f"profile {name!r}: read_only must be true or false")
        profiles[name] = toolset
        descriptions[name] = description.strip()

    gated = document.get("flag_gated_tools") or {}
    if not isinstance(gated, Mapping):
        raise ProfileManifestError("flag_gated_tools must be an object of tool -> flag")
    listed = frozenset().union(*profiles.values())
    for tool, flag_name in gated.items():
        if tool not in listed:
            raise ProfileManifestError(f"flag-gated tool {tool!r} is in no profile")
        if flag_name not in known_flags:
            raise ProfileManifestError(
                f"flag-gated tool {tool!r} names {flag_name!r}, which is not a declared rollout flag"
            )

    return ProfileManifest(
        profiles=profiles, descriptions=descriptions, read_only=frozenset(read_only),
        flag_gated_tools=dict(gated), content_sha256=content_sha256(dict(document)),
    )


def load(path: Path = DEFAULT_PATH) -> ProfileManifest:
    from ..flags import FLAGS

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileManifestError(f"cannot read the tool profile manifest {path}: {exc}") from None
    return parse(document, known_flags=frozenset(flag.name for flag in FLAGS))
