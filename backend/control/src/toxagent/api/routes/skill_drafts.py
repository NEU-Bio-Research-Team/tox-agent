"""Skill drafts: propose, review, withdraw, export."""
from __future__ import annotations

from fastapi import Depends, Query, Request

from ...application.policy import Actor
from ...domain.errors import (
    InvalidRequest,
    NotFound,
)
from ..responses import SkillDraft, SkillDraftList, SkillDraftPackage
from ..schemas import (
    SkillDraftRequest,
    SkillDraftReviewRequest,
)
from ._common import _services, actor, router


def _drafts_enabled() -> None:
    from ...platform.flags import is_enabled

    if not is_enabled("skill_drafts_v1"):
        raise NotFound("skill drafts are not enabled on this deployment")


async def _visible_draft(request: Request, principal: Actor, draft_id: str):
    """An expert sees every draft; anyone else sees only their own."""
    from ...domain.skill_draft import REVIEWER_ROLE

    _drafts_enabled()
    async with _services(request).database.unit_of_work() as uow:
        draft = await uow.skill_drafts.get(draft_id)
    if draft is None or (
        not principal.has_role(REVIEWER_ROLE) and draft.author.subject_id != principal.subject_id
    ):
        raise NotFound("no such skill draft", draft_id=draft_id)
    return draft


@router.post("/skill-drafts", status_code=201, responses={201: {"model": SkillDraft}})
async def propose_skill_draft(
    request: Request, body: SkillDraftRequest, principal: Actor = Depends(actor)
):
    """Propose a skill package for expert review. It is validated exactly as a
    shipped skill would be, and never offered to a run."""
    from ...application.investigation import skill_drafts
    from ...domain.skill_draft import DraftAuthor

    _drafts_enabled()
    services = _services(request)
    async with services.database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.propose(
                uow, author=DraftAuthor(actor="user", subject_id=principal.subject_id),
                skill_md=body.skill_md, manifest=body.manifest, references=body.references,
                rationale=body.rationale, catalog=services.skill_catalog,
            )
        except skill_drafts.DraftRefused as exc:
            raise InvalidRequest(str(exc)) from None
        await uow.commit()
    return draft.to_dict()


@router.get("/skill-drafts", responses={200: {"model": SkillDraftList}})
async def list_skill_drafts(
    request: Request, status: str | None = Query(default=None, max_length=16),
    principal: Actor = Depends(actor),
):
    from ...domain.skill_draft import REVIEWER_ROLE

    _drafts_enabled()
    async with _services(request).database.unit_of_work() as uow:
        drafts = await uow.skill_drafts.list(
            status=status,
            author_subject=None if principal.has_role(REVIEWER_ROLE) else principal.subject_id,
        )
    return {"drafts": [
        {key: value for key, value in d.to_dict().items() if key not in ("skill_md", "references")}
        for d in drafts
    ]}


@router.get("/skill-drafts/{draft_id}", responses={200: {"model": SkillDraft}})
async def get_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    return (await _visible_draft(request, principal, draft_id)).to_dict()


@router.post("/skill-drafts/{draft_id}:review", responses={200: {"model": SkillDraft}})
async def review_skill_draft(
    request: Request, draft_id: str, body: SkillDraftReviewRequest,
    principal: Actor = Depends(actor),
):
    """Approve or reject a proposed draft. Needs the expert role; an author
    does not review their own draft. Approval does not reach the catalog."""
    from ...application.investigation import skill_drafts
    from ...domain.errors import Forbidden
    from ...domain.skill_draft import InvalidDraftTransition

    await _visible_draft(request, principal, draft_id)
    async with _services(request).database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.review(
                uow, draft_id=draft_id, reviewer=principal.subject_id,
                reviewer_roles=principal.roles, decision=body.decision, note=body.note,
            )
        except InvalidDraftTransition as exc:
            raise Forbidden(str(exc), draft_id=draft_id) from None
        await uow.commit()
    return draft.to_dict()


@router.post("/skill-drafts/{draft_id}:withdraw", responses={200: {"model": SkillDraft}})
async def withdraw_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    from ...application.investigation import skill_drafts
    from ...domain.errors import Forbidden
    from ...domain.skill_draft import InvalidDraftTransition

    await _visible_draft(request, principal, draft_id)
    async with _services(request).database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.withdraw(uow, draft_id=draft_id, by=principal.subject_id)
        except InvalidDraftTransition as exc:
            raise Forbidden(str(exc), draft_id=draft_id) from None
        await uow.commit()
    return draft.to_dict()


@router.get("/skill-drafts/{draft_id}/package", responses={200: {"model": SkillDraftPackage}})
async def export_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    """An approved draft as files, with the manifest made active — the input to
    ``scripts/promote_skill_draft.py`` and a reviewed change to the catalog."""
    from ...application.investigation.skill_drafts import package_digest
    from ...domain.errors import Conflict
    from ...domain.skill_draft import InvalidDraftTransition

    draft = await _visible_draft(request, principal, draft_id)
    try:
        files = draft.package()
    except InvalidDraftTransition as exc:
        raise Conflict(str(exc), draft_id=draft_id) from None
    return {"draft_id": draft.id, "skill_id": draft.skill_id, "version": draft.version,
            "review": draft.review.to_dict() if draft.review else None,
            "files": files, "package_sha256": package_digest(files)}
