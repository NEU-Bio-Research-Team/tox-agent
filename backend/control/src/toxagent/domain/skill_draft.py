"""A proposed skill, waiting for an expert (RETHINK §4.8, W9-11).

"A skill the model distils from its own experience can be kept as a draft for
an expert to review; it never activates in production by itself." This is that
draft: a complete skill package (``SKILL.md``, ``skill.manifest.json``,
references) plus who proposed it, from which run, and why.

A draft lives in the database, never in the catalog a run is offered. Its life:

* ``proposed`` — written by a model (``propose_skill_draft``) or a person
  (the API). The package already passes the catalog's own validator, with its
  manifest ``status`` held at ``draft``.
* ``approved`` / ``rejected`` — decided by a reviewer with the ``expert`` role
  who is not its author, with a note. Approval does **not** reach the catalog:
  the catalog is hash-pinned files shipped with the release, so an approved
  draft is exported as a package and lands through a reviewed change (see
  ``scripts/promote_skill_draft.py``), where its status becomes ``active``.
* ``withdrawn`` — its author took it back before review.

Every transition is pure and returns a new draft; nothing is overwritten
silently, and the reviewer, decision, note and time are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class DraftStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


#: The role a reviewer needs. The same role the product already gives a
#: subject-matter expert; review is what an expert is for.
REVIEWER_ROLE = "expert"


class InvalidDraftTransition(ValueError):
    """A refused review or withdrawal; the message names why."""


@dataclass(frozen=True, slots=True)
class DraftAuthor:
    #: ``model`` or ``user``.
    actor: str
    subject_id: str
    session_id: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"actor": self.actor, "subject_id": self.subject_id,
                "session_id": self.session_id, "run_id": self.run_id}


@dataclass(frozen=True, slots=True)
class DraftReview:
    reviewer: str
    decision: str
    note: str
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"reviewer": self.reviewer, "decision": self.decision, "note": self.note,
                "at": self.at}


@dataclass(frozen=True, slots=True)
class SkillDraft:
    id: str
    skill_id: str
    version: str
    status: str
    author: DraftAuthor
    rationale: str
    #: ``SKILL.md`` text, the manifest object, and reference name -> text.
    skill_md: str
    manifest: Mapping[str, Any]
    references: Mapping[str, str]
    content_sha256: str
    created_at: str
    updated_at: str
    review: DraftReview | None = None
    #: Where the draft's author stood when they wrote it — the catalog pins it
    #: could have seen — so a reviewer knows what it adds to.
    catalog_sha256: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.id, "skill_id": self.skill_id, "version": self.version,
            "status": self.status, "author": self.author.to_dict(), "rationale": self.rationale,
            "skill_md": self.skill_md, "manifest": dict(self.manifest),
            "references": dict(self.references), "content_sha256": self.content_sha256,
            "catalog_sha256": self.catalog_sha256,
            "review": self.review.to_dict() if self.review else None,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillDraft":
        review = data.get("review")
        return cls(
            id=data["draft_id"], skill_id=data["skill_id"], version=data["version"],
            status=data["status"], author=DraftAuthor(**data["author"]),
            rationale=data.get("rationale", ""), skill_md=data["skill_md"],
            manifest=dict(data["manifest"]), references=dict(data.get("references") or {}),
            content_sha256=data["content_sha256"], created_at=data["created_at"],
            updated_at=data["updated_at"], review=DraftReview(**review) if review else None,
            catalog_sha256=data.get("catalog_sha256", ""),
        )

    def package(self) -> dict[str, str]:
        """The package as files, for promotion — with the manifest made active.

        Only an approved draft has a package: an unreviewed one must never be
        one file copy away from being offered to a run.
        """
        if self.status != DraftStatus.APPROVED.value:
            raise InvalidDraftTransition(f"only an approved draft has a package; this one is {self.status}")
        import json

        manifest = {**self.manifest, "status": "active"}
        files = {
            "SKILL.md": self.skill_md,
            "skill.manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        }
        for name, text in sorted(self.references.items()):
            files[f"references/{name}"] = text
        return files


def review(draft: SkillDraft, *, reviewer: str, reviewer_roles: frozenset[str], decision: str,
           note: str, at: str) -> SkillDraft:
    if REVIEWER_ROLE not in reviewer_roles:
        raise InvalidDraftTransition(f"reviewing a skill draft needs the {REVIEWER_ROLE!r} role")
    if draft.status != DraftStatus.PROPOSED.value:
        raise InvalidDraftTransition(f"this draft is {draft.status}; only a proposed draft is reviewed")
    if reviewer == draft.author.subject_id:
        raise InvalidDraftTransition("a draft is reviewed by someone other than its author")
    if decision not in ("approve", "reject"):
        raise InvalidDraftTransition("decision must be approve or reject")
    if not note.strip():
        raise InvalidDraftTransition("a review states its reason")
    status = DraftStatus.APPROVED if decision == "approve" else DraftStatus.REJECTED
    return replace(
        draft, status=status.value, updated_at=at,
        review=DraftReview(reviewer=reviewer, decision=decision, note=note.strip(), at=at),
    )


def withdraw(draft: SkillDraft, *, by: str, at: str) -> SkillDraft:
    if draft.status != DraftStatus.PROPOSED.value:
        raise InvalidDraftTransition(f"this draft is {draft.status}; only a proposed draft is withdrawn")
    if by != draft.author.subject_id:
        raise InvalidDraftTransition("only its author withdraws a draft")
    return replace(draft, status=DraftStatus.WITHDRAWN.value, updated_at=at)
