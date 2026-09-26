"""Proposing, reviewing and exporting skill drafts (RETHINK §4.8, W9-11).

The rules live in ``domain/skill_draft.py``; this module checks a proposed
package with the catalog's own validator and moves drafts through the database.

A draft is validated exactly as a shipped skill would be — front matter, name,
manifest fields, allowed profiles, required tools, declared references, sizes —
by writing it to a scratch directory and loading it with ``skill_catalog``'s
loader. There is no second, looser rule set for drafts: a draft that passes here
passes when it is promoted.
"""
from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..domain import skill_draft as sd
from ..domain.ids import SKILL_DRAFT, new_id
from ..domain.provenance import content_sha256
from .skill_catalog import MANIFEST_SCHEMA, SkillCatalog, SkillCatalogError, _load_skill

_REFERENCE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}\.md$")


class DraftRefused(ValueError):
    """A package the catalog would not load, or a draft that cannot be proposed."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _semver(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def validate_package(skill_md: str, manifest: Mapping[str, Any], references: Mapping[str, str],
                     *, catalog: SkillCatalog) -> str:
    """Load the package with the catalog's own loader; return its content hash."""
    from ..tools.registry import PROFILES

    if manifest.get("status") != "draft":
        raise DraftRefused("a proposed skill's manifest status is 'draft'; approval does not "
                           "change it, promotion does")
    skill_id = str(manifest.get("skill_id", ""))
    for name in references:
        if not _REFERENCE_NAME.match(name):
            raise DraftRefused(f"reference name {name!r} must be a short kebab-case .md file name")
    declared = sorted(f"references/{name}" for name in references)
    if sorted(manifest.get("references") or []) != declared:
        raise DraftRefused(f"the manifest must declare exactly its references: {declared}")
    existing = catalog.get(skill_id)
    if existing is not None:
        try:
            newer = _semver(str(manifest.get("version"))) > _semver(existing.version)
        except ValueError:
            newer = False
        if not newer:
            raise DraftRefused(
                f"{skill_id} {existing.version} is in the catalog; a revision needs a higher version"
            )
    known_tools = {tool for tools in PROFILES.values() for tool in tools}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch) / (skill_id or "unnamed")
        (directory / "references").mkdir(parents=True)
        (directory / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (directory / "skill.manifest.json").write_text(json.dumps(dict(manifest)), encoding="utf-8")
        for name, text in references.items():
            (directory / "references" / name).write_text(text, encoding="utf-8")
        try:
            skill = _load_skill(directory, set(PROFILES), known_tools, root="skill_drafts")
        except SkillCatalogError as exc:
            raise DraftRefused(str(exc)) from None
    return skill.content_sha256


def compose_package(*, skill_id: str, description: str, body: str,
                    required_capabilities: Iterable[str], output_contract: str,
                    allowed_profiles: Iterable[str] = ("decision_support",),
                    risk_tier: str = "medium", owner: str = "model-proposed",
                    version: str = "0.1.0") -> tuple[str, dict[str, Any]]:
    """A package from the fields a model can reasonably write."""
    skill_md = f"---\nname: {skill_id}\ndescription: {description.strip()}\n---\n\n{body.strip()}\n"
    manifest = {
        "schema_version": MANIFEST_SCHEMA, "skill_id": skill_id, "version": version,
        "status": "draft", "owner": owner, "allowed_profiles": list(allowed_profiles),
        "required_capabilities": list(required_capabilities), "risk_tier": risk_tier,
        "output_contract": output_contract.strip(),
        # A reviewer fills the eval set: which cases should and should not
        # trigger it is a judgement the author of a method is worst placed for.
        "eval_set": {"positive_tags": [], "negative_tags": []},
        "references": [],
    }
    return skill_md, manifest


async def propose(uow, *, author: sd.DraftAuthor, skill_md: str, manifest: Mapping[str, Any],
                  references: Mapping[str, str], rationale: str,
                  catalog: SkillCatalog) -> sd.SkillDraft:
    if not rationale.strip():
        raise DraftRefused("say why this skill is proposed: what situation it came from")
    digest = validate_package(skill_md, manifest, references, catalog=catalog)
    now = _now()
    draft = sd.SkillDraft(
        id=new_id(SKILL_DRAFT), skill_id=str(manifest["skill_id"]),
        version=str(manifest["version"]), status=sd.DraftStatus.PROPOSED.value, author=author,
        rationale=rationale.strip()[:2000], skill_md=skill_md, manifest=dict(manifest),
        references=dict(references), content_sha256=digest, created_at=now.isoformat(),
        updated_at=now.isoformat(), catalog_sha256=catalog.catalog_sha256,
    )
    await uow.skill_drafts.add(draft, now=now)
    return draft


async def review(uow, *, draft_id: str, reviewer: str, reviewer_roles: frozenset[str],
                 decision: str, note: str) -> sd.SkillDraft:
    draft = await uow.skill_drafts.get(draft_id)
    if draft is None:
        raise LookupError(draft_id)
    now = _now()
    decided = sd.review(draft, reviewer=reviewer, reviewer_roles=reviewer_roles,
                        decision=decision, note=note, at=now.isoformat())
    await uow.skill_drafts.save(decided, expected_status=draft.status, now=now)
    return decided


async def withdraw(uow, *, draft_id: str, by: str) -> sd.SkillDraft:
    draft = await uow.skill_drafts.get(draft_id)
    if draft is None:
        raise LookupError(draft_id)
    now = _now()
    withdrawn = sd.withdraw(draft, by=by, at=now.isoformat())
    await uow.skill_drafts.save(withdrawn, expected_status=draft.status, now=now)
    return withdrawn


def package_digest(files: Mapping[str, str]) -> str:
    """What a promotion commits to, for the reviewer to compare."""
    return content_sha256(dict(sorted(files.items())))
