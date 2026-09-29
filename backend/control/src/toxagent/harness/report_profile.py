"""Compose the report-builder instructions, and hash what was composed.

Spec sections 6 and 7. Skills here are versioned instruction packages, not
runtime permissions — the deployed OpenCode profile denies the runtime ``skill``
tool along with everything else, so nothing is "loaded" by a model at run time.
This module reads the packages off disk, composes them into one system prompt
before dispatch, and records each file's hash.

The hashing is the point. "Which instructions produced this report" has to be
answerable six months later from the runtime manifest, not from whatever
happens to be on a worker's disk today. A changed SKILL.md changes the manifest
hash, and the audit says so.

Composition is deterministic: files are read in a declared order, joined with a
fixed separator, and the ``profile.json`` skill list is the order. Two processes
composing the same directory produce byte-identical output and the same hash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..domain.provenance import content_sha256

PROFILE_NAME = "report_build"

#: Separator between composed parts. Fixed, and part of the hashed content:
#: changing it changes every manifest, which is correct — it changes the prompt.
_SEPARATOR = "\n\n---\n\n"


class ProfileUnavailable(FileNotFoundError):
    """The profile directory is missing or incomplete.

    Raised rather than degraded: dispatching a report build with silently
    missing instructions would produce a report whose manifest claims rules the
    model was never given.
    """


@dataclass(frozen=True)
class ComposedProfile:
    """The instructions for one dispatch, plus what they were made of."""

    profile: str
    instructions: str
    #: relative path -> sha256 of that file's bytes. Recorded in the runtime
    #: manifest so a report can be traced to the exact instruction set.
    file_hashes: Mapping[str, str]
    #: sha256 over the composed text. One value the manifest can compare.
    content_sha256: str
    skills: tuple[str, ...]
    #: Each skill's catalog pin (id, version, content hash), W9-10.
    skill_pins: tuple[Mapping[str, str], ...] = ()

    def manifest(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "instructions_sha256": self.content_sha256,
            "skills": list(self.skills),
            "skill_pins": [dict(pin) for pin in self.skill_pins],
            "files": dict(sorted(self.file_hashes.items())),
        }


def _read(root: Path, relative: str, hashes: dict[str, str]) -> str:
    path = root / relative
    if not path.is_file():
        raise ProfileUnavailable(f"{relative} is missing from the {PROFILE_NAME} profile")
    text = path.read_text(encoding="utf-8")
    hashes[relative] = content_sha256(text)
    return text.strip()


def compose_report_profile(profiles_dir: Path) -> ComposedProfile:
    """Read ``AGENTS.md``, the shared references and every declared skill.

    A skill contributes its ``SKILL.md`` and every file under its
    ``references/``. Reference files are sorted by name so composition does not
    depend on directory iteration order — the difference between a stable hash
    and one that changes per filesystem.
    """
    root = Path(profiles_dir) / PROFILE_NAME
    if not root.is_dir():
        raise ProfileUnavailable(f"no {PROFILE_NAME} profile directory at {root}")

    hashes: dict[str, str] = {}
    parts: list[str] = [_read(root, "AGENTS.md", hashes)]

    manifest_text = _read(root, "profile.json", hashes)
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise ProfileUnavailable(f"{PROFILE_NAME}/profile.json is not valid JSON: {exc}") from exc
    skills = tuple(manifest.get("skills", ()))
    pins = _catalog_pins(Path(profiles_dir), skills)

    shared = root / "references"
    if shared.is_dir():
        for path in sorted(shared.glob("*.md")):
            relative = f"references/{path.name}"
            parts.append(f"# Reference: {path.stem}\n\n{_read(root, relative, hashes)}")

    for skill in skills:
        base = f"skills/{skill}"
        parts.append(_read(root, f"{base}/SKILL.md", hashes))
        skill_refs = root / base / "references"
        if skill_refs.is_dir():
            for path in sorted(skill_refs.glob("*.md")):
                relative = f"{base}/references/{path.name}"
                parts.append(
                    f"# Reference for {skill}: {path.stem}\n\n{_read(root, relative, hashes)}"
                )

    instructions = _SEPARATOR.join(parts)
    return ComposedProfile(
        profile=PROFILE_NAME,
        instructions=instructions,
        file_hashes=hashes,
        content_sha256=content_sha256(instructions),
        skills=skills,
        skill_pins=pins,
    )


def _catalog_pins(profiles_dir: Path, skills: tuple[str, ...]) -> tuple[Mapping[str, str], ...]:
    """Every declared skill must be an active catalog skill for this profile.

    W9-10: the report skills are catalog packages now — validated, versioned
    and hash-pinned like the scientific ones — but they are still composed
    whole into the prompt (the static arm, and the default), byte for byte as
    before. RETHINK §4.11 step 2 moves them to on-demand loading one at a time,
    only after the dynamic arm has shown an effect; this is the step that
    makes that possible without touching what a report run is told today.
    """
    from ..application.investigation.skill_catalog import SkillCatalogError, load_catalog

    try:
        catalog = load_catalog(profiles_dir)
    except SkillCatalogError as exc:
        raise ProfileUnavailable(f"the skill catalog does not load: {exc}") from exc
    pins = []
    for name in skills:
        skill = catalog.get(name)
        if skill is None or skill.status != "active" or PROFILE_NAME not in skill.allowed_profiles:
            raise ProfileUnavailable(
                f"{PROFILE_NAME}/profile.json declares {name!r}, which is not an active catalog "
                f"skill allowed for {PROFILE_NAME}"
            )
        pins.append(skill.pin())
    return tuple(pins)
