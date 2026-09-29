"""Saved report drafts: save, check, patch by version, submit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from pydantic import ValidationError

from ....domain.errors import Conflict
from ....domain.events import EventType
from ....validation.report.draft_wire import ReportDraftCandidate
from ....validation.report.validator import (
    validate_report_draft,
)
from ._common import (
    WORKING_DRAFT_KEY,
    WORKING_DRAFT_SHA_KEY,
    WORKING_DRAFT_VERSION_KEY,
    DraftCheckpointOutcome,
    SubmitReportOutcome,
    _apply_draft_patch,
    _draft_sha256,
    _now,
)


class DraftCheckpointMixin:
    """The saved-draft (checkpoint) path of a report draft submission."""

    async def save_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        draft: ReportDraftCandidate,
    ) -> DraftCheckpointOutcome:
        """Persist one full candidate once, and validate the stored version.

        Subsequent repair turns operate on JSON-pointer patches. This keeps a
        long report out of repeated model output and leaves a resumable product
        checkpoint if the runtime or provider disappears between validation
        and final submission.
        """
        document = draft.model_dump(mode="json")
        async with self._db.unit_of_work() as uow:
            build, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
            version = int(build.stage_state.get(WORKING_DRAFT_VERSION_KEY, 0)) + 1
            digest = _draft_sha256(document)
            state = dict(build.stage_state)
            state.update({
                WORKING_DRAFT_KEY: document,
                WORKING_DRAFT_VERSION_KEY: version,
                WORKING_DRAFT_SHA_KEY: digest,
            })
            await uow.reports.save_build(replace(build, stage_state=state, updated_at=_now()))
            uow.emit(
                session_id=session_id, type=EventType.REPORT_DRAFT_SAVED,
                entity_type="report_build", entity_id=build.id, run_id=run_id,
                payload={
                    "draft_version": version,
                    "content_sha256": digest,
                    "violation_count": len(result.violations),
                    "violation_codes": sorted({v.code for v in result.violations}),
                },
            )
            await uow.commit()
        return DraftCheckpointOutcome(
            draft=draft, version=version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def check_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
    ) -> DraftCheckpointOutcome:
        async with self._db.unit_of_work() as uow:
            build, snapshot, draft, version, digest = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
        return DraftCheckpointOutcome(
            draft=draft, version=version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def patch_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
        expected_version: int,
        operations: list[Mapping[str, Any]],
    ) -> DraftCheckpointOutcome:
        async with self._db.unit_of_work() as uow:
            build, snapshot, _, version, _ = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
            if version != expected_version:
                raise Conflict(
                    "the saved report draft changed; re-read it before patching",
                    expected_version=expected_version, current_version=version,
                )
            document = deepcopy(build.stage_state[WORKING_DRAFT_KEY])
            _apply_draft_patch(document, operations)
            try:
                draft = ReportDraftCandidate.model_validate(document)
            except ValidationError as exc:
                raise Conflict(
                    "the patch would make the saved report draft structurally invalid",
                    errors=exc.errors(include_url=False, include_context=False)[:10],
                ) from exc
            if draft.report_build_id != report_build_id:
                raise Conflict("a patch cannot move a draft to another report build")

            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
            next_version = version + 1
            canonical = draft.model_dump(mode="json")
            digest = _draft_sha256(canonical)
            state = dict(build.stage_state)
            state.update({
                WORKING_DRAFT_KEY: canonical,
                WORKING_DRAFT_VERSION_KEY: next_version,
                WORKING_DRAFT_SHA_KEY: digest,
            })
            await uow.reports.save_build(replace(build, stage_state=state, updated_at=_now()))
            uow.emit(
                session_id=session_id, type=EventType.REPORT_DRAFT_PATCHED,
                entity_type="report_build", entity_id=build.id, run_id=run_id,
                payload={
                    "draft_version": next_version,
                    "content_sha256": digest,
                    "operation_count": len(operations),
                    "violation_count": len(result.violations),
                    "violation_codes": sorted({v.code for v in result.violations}),
                },
            )
            await uow.commit()
        return DraftCheckpointOutcome(
            draft=draft, version=next_version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def execute_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
        expected_version: int,
        language: str = "en",
    ) -> SubmitReportOutcome:
        async with self._db.unit_of_work() as uow:
            _, _, draft, version, _ = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
        if version != expected_version:
            raise Conflict(
                "the saved report draft changed; check the current version before submitting",
                expected_version=expected_version, current_version=version,
            )
        return await self.execute(
            session_id=session_id, run_id=run_id, owner_id=owner_id,
            draft=draft, language=language,
        )

    async def _load_checkpoint(self, uow, *, session_id: str, report_build_id: str):
        build = await uow.reports.get_build(report_build_id, session_id=session_id)
        if build is None:
            raise Conflict("no such report build in this session", build_id=report_build_id)
        raw = build.stage_state.get(WORKING_DRAFT_KEY)
        if raw is None:
            raise Conflict(
                "this report build has no saved working draft; call save_report_draft first",
                build_id=report_build_id,
            )
        try:
            draft = ReportDraftCandidate.model_validate(raw)
        except ValidationError as exc:  # defensive: old/corrupt checkpoint
            raise Conflict(
                "the saved working draft cannot be reconstructed",
                errors=exc.errors(include_url=False, include_context=False)[:10],
            ) from exc
        admitted, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
        version = int(build.stage_state.get(WORKING_DRAFT_VERSION_KEY, 1))
        digest = str(build.stage_state.get(WORKING_DRAFT_SHA_KEY) or _draft_sha256(raw))
        return admitted, snapshot, draft, version, digest
