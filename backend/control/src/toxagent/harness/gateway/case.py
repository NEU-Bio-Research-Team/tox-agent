"""The gateway's scientific-case side: skills, claim review, decision state."""
from __future__ import annotations

from datetime import timedelta
from typing import Awaitable, Callable

from ...application.investigation import decision_state_service, scientific_case_service
from ...application.runs.scheduler import RunContext
from ...domain import decision_state
from ...domain.provenance import content_sha256
from ...platform.flags import is_enabled
from ._common import _no_commit, _now, log


class ScientificCaseMixin:
    """Scientific case, decision state and claim review for a gateway turn."""

    def _skills_for(self, profile: str) -> tuple[str, tuple]:
        """The run's skill arm and the skills it offers (RETHINK §5.5).

        ``dynamic`` (flag scientific_skills_v1): an index in the prompt, bodies
        read on demand. ``static`` (TOXAGENT_SCIENTIFIC_SKILLS_STATIC): every
        offered skill composed into the prompt. ``off`` otherwise, and for
        every profile but decision_support. Offered means active, allowed for
        the profile, and with every required tool visible to this run.
        """
        if profile != "decision_support":
            return "off", ()
        if is_enabled("scientific_skills_v1"):
            mode = "dynamic"
        elif getattr(self._settings, "scientific_skills_static", False):
            mode = "static"
        else:
            return "off", ()
        offered = self._skill_catalog.available(
            profile, [tool.name for tool in self._registry.visible_for(profile)]
        )
        return (mode, offered) if offered else ("off", ())

    #: The reviewer's turn gets at most this long, and is skipped when the run
    #: has less than the floor left: the answer is already accepted, and the
    #: review must never be what makes the run miss its deadline (W9-12).
    CLAIM_REVIEW_BUDGET_S = 120

    CLAIM_REVIEW_FLOOR_S = 45

    def _with_claim_review(self, commit: Callable[[RunContext], Awaitable[None]]):
        """``commit``, preceded by the independent claim review when its flag is on."""
        if not is_enabled("claim_reviewer_v1"):
            return commit

        async def reviewed(context: RunContext) -> None:
            await self._review_claims(context)
            await commit(context)

        return reviewed

    async def _review_claims(self, context: RunContext) -> None:
        """One reviewer turn over the accepted answer (RETHINK §4.4 step 5).

        Bookkeeping: whatever happens, the answer stands and the run proceeds;
        the state records ``completed``, ``skipped`` (and why) or ``failed``.
        """
        import json

        from ...application.investigation import claim_review

        def record(review: dict) -> Awaitable:
            return decision_state_service.advance(
                self._db, context.run_id,
                lambda state: decision_state.record_claim_review(state, review),
            )

        try:
            async with self._db.unit_of_work() as uow:
                answer = await uow.answers.get_for_run(context.run_id)
                run = await uow.runs.get(context.run_id)
                bundle = (
                    await claim_review.review_bundle(uow, session_id=context.session_id, answer=answer)
                    if answer is not None else None
                )
            if answer is None or answer.is_fallback:
                await record({"status": "skipped", "reason": "no model-written answer to review"})
                return
            if not bundle["claims"]:
                await record({"status": "skipped", "reason": "the answer makes no reviewable claim",
                              "answer_id": answer.id})
                return
            now = _now()
            remaining = (run.deadline_at - now).total_seconds() if run is not None else 0
            if remaining < self.CLAIM_REVIEW_FLOOR_S:
                await record({"status": "skipped", "reason": "not enough run budget left",
                              "remaining_s": int(remaining), "answer_id": answer.id})
                return
            deadline = min(run.deadline_at, now + timedelta(seconds=self.CLAIM_REVIEW_BUDGET_S))
            prompt = (
                claim_review.REVIEW_INSTRUCTIONS
                + "\n\nThe claims under review, each with the sources it cites:\n```json\n"
                + json.dumps(bundle, sort_keys=True, ensure_ascii=False, default=str)
                + "\n```"
            )

            async def has_review(_context: RunContext) -> bool:
                async with self._db.unit_of_work() as uow:
                    state = await uow.decision_states.get(context.run_id)
                return bool(state and state.claim_review.get("status") == "completed")

            from dataclasses import replace

            await self._dispatch(
                replace(context, text="Review the claims listed in your instructions."),
                system_prompt=prompt, profile="claim_review", deadline=deadline,
                instructions_hash=content_sha256(claim_review.REVIEW_INSTRUCTIONS),
                has_product=has_review, commit=_no_commit,
            )
            if not await has_review(context):
                await record({"status": "failed", "reason": "the reviewer submitted no verdicts",
                              "answer_id": answer.id})
        except Exception as exc:  # noqa: BLE001 - the answer is accepted; review is an observer
            log.exception("claim review failed", extra={"run_id": context.run_id})
            try:
                await record({"status": "failed", "reason": type(exc).__name__})
            except Exception:  # noqa: BLE001
                pass

    async def _commit_with_case(self, context: RunContext) -> None:
        """Finish the case and store the dossier, then complete the run.

        In this order so that a client that sees the run complete can read its
        dossier: the other order leaves a window in which the case has no stop
        reason. The stop reason is the one the run's state will be finalized
        with once it completes (``decision_state.finalize`` is pure, so this is
        a preview of it, not a second source); the state itself is still
        finalized after the run, exactly as without a case.
        """
        stop_reason = None
        try:
            async with self._db.unit_of_work() as uow:
                state = await uow.decision_states.get(context.run_id)
            if state is not None:
                stop_reason = decision_state.finalize(state, run_status="completed").stop_reason
        except Exception:  # noqa: BLE001 - bookkeeping must not stop the run
            log.exception("could not preview the stop reason", extra={"run_id": context.run_id})
        await scientific_case_service.finish_run(
            self._db, session_id=context.session_id, run_id=context.run_id,
            stop_reason=stop_reason,
        )
        await self._commit_product_and_complete(context)

    async def _begin_scientific_case(self, context: RunContext) -> None:
        """Open or continue the session's case for this subject (ADR 0012).

        Bookkeeping: a failure is logged and the run proceeds without a case,
        exactly as it would with the flag off.
        """
        try:
            async with self._db.unit_of_work() as uow:
                session = await uow.sessions.get_unscoped(context.session_id)
                analysis_id = context.analysis_id or (
                    session.active_analysis_id if session else None
                )
                snapshot = (
                    await uow.analyses.get(analysis_id, session_id=context.session_id)
                    if analysis_id else None
                )
                await scientific_case_service.open_or_continue(
                    uow, session_id=context.session_id, analysis_id=analysis_id,
                    run_id=context.run_id, goal=context.text or "",
                    subject_refs=[f"analysis:{analysis_id}"] if analysis_id else [],
                    extra_updates=scientific_case_service.analysis_uncertainties(
                        snapshot, run_id=context.run_id
                    ),
                    requester=session.owner_id if session else "",
                )
                await uow.commit()
        except Exception:  # noqa: BLE001 - bookkeeping must not stop the run
            log.exception("could not open the scientific case", extra={"run_id": context.run_id})

    async def _begin_decision_state(self, context: RunContext) -> None:
        """Open the run's DecisionSupportStateV1 (TAB-Suite Wave 2)."""
        from ...application.runs.budget import effective_run_budget
        from ...platform.config import PolicySettings

        try:
            async with self._db.unit_of_work() as uow:
                session = await uow.sessions.get_unscoped(context.session_id)
                subjects = []
                analysis_id = context.analysis_id or (session.active_analysis_id if session else None)
                if analysis_id:
                    subjects.append(f"analysis:{analysis_id}")
                    report = await uow.reports.get_latest_artifact_for_analysis(
                        analysis_id, session_id=context.session_id
                    )
                    if report is not None:
                        subjects.append(f"report:{report.get('report_id') or report.get('id')}")
                await decision_state_service.begin(
                    uow, session_id=context.session_id, run_id=context.run_id,
                    goal=context.text or "", subject_refs=subjects,
                    budget_snapshot=effective_run_budget(
                        context.intent.value, self._policy or PolicySettings(), self._settings
                    ).to_dict(),
                )
                await uow.commit()
        except Exception:  # noqa: BLE001 - bookkeeping must not stop the run
            log.exception("could not open decision-support state", extra={"run_id": context.run_id})
