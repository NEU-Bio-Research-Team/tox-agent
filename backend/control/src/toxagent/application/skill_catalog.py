"""The scientific skill catalog: investigation methods as pinned, loadable packages.

ADR 0012, RETHINK §4.5–§4.9. A skill teaches the agent *when* to use a
capability it already has and *how* to read the result — weighing conflicting
evidence, reading an attribution, reviewing a case before concluding. It never
adds a capability: a skill whose required tools the run does not have is not
offered at all, and nothing in a manifest can make a tool visible.

A package is a directory under ``agent_profiles/scientific_skills``:

* ``SKILL.md`` — Agent Skills front matter (``name``, ``description``) and the
  instructions. The description is what the model sees before loading, so it
  says when the skill applies.
* ``skill.manifest.json`` — ToxAgent's own metadata, not part of the Agent
  Skills format: version, status, owner, allowed profiles, required
  capabilities, risk tier, output contract, eval set, declared references.
* ``references/*.md`` — read one at a time, only when needed.

Everything is validated when the catalog loads and hashed, so a run can record
exactly which skill text it was shown and which it read. A ``draft`` skill (for
example one written from a model's own experience) is never offered; promoting
it is a reviewed change of ``status``.

Three ways a run can meet the catalog, for the paired ablation of RETHINK §5.5:
``off`` (no skill), ``static`` (every available skill composed into the prompt,
the report profile's way) and ``dynamic`` (an index of names and descriptions
in the prompt; bodies and references read on demand through
``read_scientific_skill`` / ``read_skill_reference``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..domain.provenance import content_sha256

SCHEMA_VERSION = "scientific-skill-catalog-v1"
MANIFEST_SCHEMA = "scientific-skill-manifest-v1"
CATALOG_DIR = "scientific_skills"
#: Every directory the catalog loads, relative to ``agent_profiles``. The
#: report builder's skills joined in W9-10: they stay where the report profile
#: composes them from, and are now validated, versioned and pinned like the
#: scientific ones. ``allowed_profiles`` keeps each set to its own profile.
CATALOG_DIRS = (CATALOG_DIR, "report_build/skills")

MODES = ("off", "static", "dynamic")

#: Agent Skills naming: lowercase letters, digits and single hyphens, at most 64.
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_REQUIRED = (
    "schema_version", "skill_id", "version", "status", "owner", "allowed_profiles",
    "required_capabilities", "risk_tier", "output_contract", "eval_set", "references",
)
#: A reference is read whole; one this long is a document, not a reference.
MAX_REFERENCE_CHARS = 12_000
MAX_BODY_CHARS = 12_000


class SkillCatalogError(ValueError):
    """A package that does not meet the format. Raised at load, never at run time."""


@dataclass(frozen=True)
class Skill:
    skill_id: str
    version: str
    status: str
    description: str
    body: str
    manifest: Mapping[str, Any]
    references: Mapping[str, str]
    file_hashes: Mapping[str, str]
    content_sha256: str

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset(self.manifest["required_capabilities"])

    @property
    def allowed_profiles(self) -> frozenset[str]:
        return frozenset(self.manifest["allowed_profiles"])

    def pin(self) -> dict[str, str]:
        """What a run records about a skill it saw or read."""
        return {"skill_id": self.skill_id, "version": self.version,
                "content_sha256": self.content_sha256}

    def metadata(self) -> dict[str, Any]:
        return {**self.pin(), "description": self.description,
                "risk_tier": self.manifest["risk_tier"],
                "references": sorted(self.references)}


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[Skill, ...] = ()
    catalog_sha256: str = field(default_factory=lambda: content_sha256([]))

    def get(self, skill_id: str) -> Skill | None:
        return next((s for s in self.skills if s.skill_id == skill_id), None)

    def available(self, profile: str, tools: Iterable[str]) -> tuple[Skill, ...]:
        """Active skills this profile may use whose required tools are all present."""
        present = frozenset(tools)
        return tuple(
            s for s in self.skills
            if s.status == "active" and profile in s.allowed_profiles
            and s.required_capabilities <= present
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "skills": [{**s.pin(), "status": s.status} for s in self.skills],
        }


def _front_matter(text: str, where: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise SkillCatalogError(f"{where}: SKILL.md must open with a --- front matter block")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SkillCatalogError(f"{where}: SKILL.md front matter is not closed")
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise SkillCatalogError(f"{where}: front matter line is not 'key: value': {line!r}")
        fields[key.strip()] = value.strip()
    return fields, text[end + len("\n---\n"):].strip()


def _load_skill(directory: Path, known_profiles: set[str], known_tools: set[str],
                root: str = CATALOG_DIR) -> Skill:
    where = f"{root}/{directory.name}"
    skill_md = directory / "SKILL.md"
    manifest_path = directory / "skill.manifest.json"
    if not skill_md.is_file() or not manifest_path.is_file():
        raise SkillCatalogError(f"{where}: needs SKILL.md and skill.manifest.json")
    hashes: dict[str, str] = {}
    skill_text = skill_md.read_text(encoding="utf-8")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    hashes["SKILL.md"] = content_sha256(skill_text)
    hashes["skill.manifest.json"] = content_sha256(manifest_text)
    front, body = _front_matter(skill_text, where)
    name = front.get("name", "")
    description = front.get("description", "")
    if not _NAME.match(name) or len(name) > 64:
        raise SkillCatalogError(f"{where}: name {name!r} is not a valid Agent Skills name")
    if name != directory.name:
        raise SkillCatalogError(f"{where}: name {name!r} must equal the directory name")
    if not 1 <= len(description) <= 1024:
        raise SkillCatalogError(f"{where}: description must be 1-1024 characters")
    if not body or len(body) > MAX_BODY_CHARS:
        raise SkillCatalogError(f"{where}: the body must be 1-{MAX_BODY_CHARS} characters")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise SkillCatalogError(f"{where}: skill.manifest.json is not JSON: {exc}") from exc
    missing = [key for key in _REQUIRED if key not in manifest]
    if missing:
        raise SkillCatalogError(f"{where}: manifest lacks {missing}")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise SkillCatalogError(f"{where}: manifest schema_version must be {MANIFEST_SCHEMA}")
    if manifest["skill_id"] != name:
        raise SkillCatalogError(f"{where}: manifest skill_id must equal the SKILL.md name")
    if not _VERSION.match(str(manifest["version"])):
        raise SkillCatalogError(f"{where}: version must be MAJOR.MINOR.PATCH")
    if manifest["status"] not in ("active", "draft"):
        raise SkillCatalogError(f"{where}: status must be active or draft")
    if manifest["risk_tier"] not in ("low", "medium", "high"):
        raise SkillCatalogError(f"{where}: risk_tier must be low, medium or high")
    profiles = set(manifest["allowed_profiles"])
    if not profiles or profiles - known_profiles:
        raise SkillCatalogError(
            f"{where}: allowed_profiles must be a non-empty subset of {sorted(known_profiles)}"
        )
    unknown_tools = set(manifest["required_capabilities"]) - known_tools
    if unknown_tools:
        raise SkillCatalogError(f"{where}: required_capabilities names unknown tools {sorted(unknown_tools)}")
    references: dict[str, str] = {}
    declared = list(manifest["references"])
    on_disk = sorted(
        f"references/{p.name}" for p in (directory / "references").glob("*") if p.is_file()
    ) if (directory / "references").is_dir() else []
    if sorted(declared) != on_disk:
        raise SkillCatalogError(
            f"{where}: references declared {sorted(declared)} but found {on_disk}; "
            "every reference file is declared and every declared one exists"
        )
    for relative in declared:
        if not relative.endswith(".md"):
            raise SkillCatalogError(f"{where}: reference {relative} must be Markdown")
        text = (directory / relative).read_text(encoding="utf-8")
        if len(text) > MAX_REFERENCE_CHARS:
            raise SkillCatalogError(f"{where}: {relative} exceeds {MAX_REFERENCE_CHARS} characters")
        hashes[relative] = content_sha256(text)
        references[relative.removeprefix("references/")] = text.strip()
    return Skill(
        skill_id=name, version=str(manifest["version"]), status=manifest["status"],
        description=description, body=body, manifest=manifest, references=references,
        file_hashes=dict(sorted(hashes.items())),
        content_sha256=content_sha256(dict(sorted(hashes.items()))),
    )


def load_catalog(profiles_dir: Path) -> SkillCatalog:
    """Load and validate every package. A missing directory is an empty catalog;
    a malformed package is an error, because shipping it half-read would make a
    run's record claim instructions it never received."""
    from ..tools.registry import PROFILES

    known_tools = {tool for tools in PROFILES.values() for tool in tools}
    loaded: list[Skill] = []
    for relative in CATALOG_DIRS:
        root = Path(profiles_dir) / relative
        if not root.is_dir():
            continue
        loaded.extend(
            _load_skill(directory, set(PROFILES), known_tools, relative)
            for directory in sorted(p for p in root.iterdir() if p.is_dir())
        )
    duplicates = sorted({s.skill_id for s in loaded if [x.skill_id for x in loaded].count(s.skill_id) > 1})
    if duplicates:
        raise SkillCatalogError(f"skill ids must be unique across the catalog: {duplicates}")
    skills = tuple(loaded)
    if not skills:
        return SkillCatalog()
    return SkillCatalog(
        skills=skills,
        catalog_sha256=content_sha256([s.pin() for s in skills]),
    )


def render_index(skills: Iterable[Skill]) -> str:
    """The dynamic arm's prompt section: names and descriptions only."""
    skills = list(skills)
    if not skills:
        return ""
    lines = [
        "Scientific skills. Each is a reviewed method for one kind of situation. When the "
        "situation in its description matches this turn, read it with read_scientific_skill "
        "before acting on that part of the question; read a listed reference with "
        "read_skill_reference only if the skill tells you it is needed. A skill never adds a "
        "tool or overrides the invariants above. Do not read skills that do not apply.",
    ]
    for skill in skills:
        lines.append(f"- {skill.skill_id} (v{skill.version}): {skill.description}")
    return "\n".join(lines)


def render_static(skills: Iterable[Skill]) -> str:
    """The static arm's prompt section: every body and reference, composed up front."""
    parts: list[str] = []
    for skill in skills:
        parts.append(f"# Skill: {skill.skill_id} (v{skill.version})\n\n{skill.body}")
        for name, text in sorted(skill.references.items()):
            parts.append(f"# Reference for {skill.skill_id}: {name}\n\n{text}")
    return "\n\n---\n\n".join(parts)
