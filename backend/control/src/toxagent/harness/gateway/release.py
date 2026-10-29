"""The release gate: review before an answer becomes the run's (#47).

Under ``release_gate_v1`` an accepted answer is a candidate. Before the run
commits, the independent reviewer judges it; a blocking verdict gets one
repair turn, the repaired candidate is validated as any submission is and
reviewed again, and only then is an answer released: ``answer.accepted`` is
emitted, the answer route serves it and the assistant message cites it.

Every way this can end is a recorded outcome (domain/answer_release.py). The
policy, locked here so a run cannot drift between two readings of it:

* every claim under review judged ``supported`` -> ``reviewed``;
* a blocking verdict, repaired, then all supported -> ``repaired_and_reviewed``;
* still blocking after the one repair, or the repair gave nothing new ->
  ``released_with_unresolved_review``, with the blocking claims named. The
  answer is released, because withholding it would hide what was found, and
  it is never presented as reviewed;
* nothing blocking but some claims ``cannot_judge`` or without a verdict ->
  ``review_incomplete``-style ``ungated``, with the count -- not a pass;
* the reviewer failed, timed out or had no budget -> ``ungated``, with why;
* the deterministic fallback -> ``fallback``.

The reviewer and the repair turn run in the same run, so their usage is the
run's usage, and the repair's tool calls count against the run's budget.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from typing import Any

from ...application.investigation import (
    claim_review,
    decision_state_service,
    release_gate,
    scientific_case_service,
)
from ...application.runs.scheduler import RunContext
from ...domain import decision_state
from ...domain.answer_release import AnswerRelease, ReleaseOutcome
from ...domain.run_result import ReleasePolicy
from ...domain.events import EventType
from ...domain.provenance import content_sha256
from ...platform.flags import is_enabled
from ._common import _no_commit, _now, log

#: A repair is one more turn of the answering agent; it is not started with
#: less than this left on the run's clock.
REPAIR_FLOOR_S = 60

REPAIR_INSTRUCTIONS = """\
An independent reviewer judged your answer below against the sources each \
claim cites, and did not accept some of its claims. Repair it once:
- for each claim listed under "blocking", do what its remedy says: correct it \
to what its sources state, narrow it to the scope they carry, or remove it. \
Do not make it sound more certain;
- keep every other claim as it is, with the same local_ref, so the repair can \
be compared claim by claim;
- keep the limitations the answer owes.
Then call submit_grounded_answer once with the whole repaired answer. You may \
read a source again if you need to; do not start a new investigation."""


def release_gate_on() -> bool:
    return (
        is_enabled("release_gate_v1") and is_enabled("claim_reviewer_v1")
        and is_enabled("answer_draft_v2")
    )


def declared_policy() -> ReleasePolicy:
    """The release policy this deployment declares (TOXAGENT_RELEASE_POLICY).
    A strict policy needs the reviewer and the v2 draft; refusing to run
    without them is better than a strict label on an unreviewed answer."""
    from ...platform.config import release_policy

    policy = ReleasePolicy(release_policy())
    if policy is ReleasePolicy.STRICT_REVIEWED_V1 and not (
        is_enabled("claim_reviewer_v1") and is_enabled("answer_draft_v2")
    ):
        raise RuntimeError(
            "TOXAGENT_RELEASE_POLICY=strict-reviewed-v1 needs claim_reviewer_v1 and "
            "answer_draft_v2 on"
        )
    if policy is ReleasePolicy.HARD_ONLY_V1 and not is_enabled("answer_draft_v2"):
        raise RuntimeError("TOXAGENT_RELEASE_POLICY=hard-only-v1 needs answer_draft_v2 on")
    return policy


class ReleaseGateMixin:
    """Review, at most one repair, and the release record, for a gated run."""

    def _with_publication(self, commit, *, system_prompt: str,
                          repair_profile: str = "decision_support", prompt_for=None):
        """#47 under a declared policy: the server decides what is published
        and records a run result, then the run commits. strict-reviewed-v1
        reviews (and repairs once) first; hard-only-v1 publishes the
        hard-validated candidate without a review, by design."""
        policy = declared_policy()
        if policy is ReleasePolicy.LEGACY:
            raise RuntimeError("_with_publication is for a declared policy")

        async def published(context: RunContext) -> None:
            try:
                if policy is ReleasePolicy.HARD_ONLY_V1:
                    candidate = await self._latest_answer(context)
                    if candidate is not None:
                        await self._release(
                            context, candidate,
                            outcome=(ReleaseOutcome.FALLBACK if candidate.is_fallback
                                     else ReleaseOutcome.UNGATED),
                            ungated_reason=None if candidate.is_fallback
                            else "hard-only-v1: no semantic review, by design",
                        )
                else:
                    await self._release_through_review(
                        context,
                        system_prompt=prompt_for(context) if prompt_for else system_prompt,
                        repair_profile=repair_profile,
                    )
            except Exception:  # noqa: BLE001 - a result is recorded, whatever the gate did
                log.exception("the release gate failed", extra={"run_id": context.run_id})
                await self._release_latest(
                    context, outcome=ReleaseOutcome.UNGATED,
                    ungated_reason="the release gate failed; no usable review",
                )
            await commit(context)

        return published

    def _with_release_gate(self, commit, *, system_prompt: str,
                           repair_profile: str = "decision_support", prompt_for=None):
        """``prompt_for(context)`` gives the repair turn's base prompt when it
        is only known once the run has started (arm B's synthesis material)."""
        async def gated(context: RunContext) -> None:
            try:
                await self._release_through_review(
                    context,
                    system_prompt=prompt_for(context) if prompt_for else system_prompt,
                    repair_profile=repair_profile,
                )
            except Exception:  # noqa: BLE001 - an answer is released, whatever the gate did
                log.exception("the release gate failed", extra={"run_id": context.run_id})
                await self._release_latest(
                    context, outcome=ReleaseOutcome.UNGATED,
                    ungated_reason="the release gate failed; released without a usable review",
                )
            await commit(context)

        return gated

    async def _latest_answer(self, context: RunContext):
        async with self._db.unit_of_work() as uow:
            return await uow.answers.get_for_run(context.run_id)

    async def _release_through_review(
        self, context: RunContext, *, system_prompt: str,
        repair_profile: str = "decision_support",
    ) -> None:
        async with self._db.unit_of_work() as uow:
            if await uow.answer_releases.get_for_run(context.run_id) is not None:
                return
        candidate = await self._latest_answer(context)
        if candidate is None:
            return  # the commit raises the protocol error it always has
        if candidate.is_fallback:
            await self._release(context, candidate, outcome=ReleaseOutcome.FALLBACK)
            return

        first, decision = await self._review_and_decide(context, candidate)
        reviews = [first]
        settled = self._settled(first, decision)
        if settled is not None:
            outcome, reason, unresolved = settled
            await self._release(context, candidate, outcome=outcome, reviews=reviews,
                                ungated_reason=reason, unresolved=unresolved)
            return

        # A blocking verdict: one repair.
        repaired = await self._repair(context, candidate, decision, system_prompt=system_prompt,
                                      profile=repair_profile)
        if repaired is None or repaired.id == candidate.id:
            await self._release(
                context, candidate, outcome=ReleaseOutcome.RELEASED_WITH_UNRESOLVED_REVIEW,
                reviews=reviews, unresolved=decision.blocking_claim_ids,
                ungated_reason=(
                    "the repair turn submitted no new answer" if repaired is not None
                    else "no run budget left for a repair"
                ),
            )
            return
        async with self._db.unit_of_work() as uow:
            before = await uow.evidence_relations.list_for_run(
                context.run_id, answer_id=candidate.id
            )
            after = await uow.evidence_relations.list_for_run(
                context.run_id, answer_id=repaired.id
            )
        effect = release_gate.repair_effect(
            candidate, repaired, relations_before=len(before), relations_after=len(after),
        ).to_dict()
        if repaired.is_fallback:
            await self._release(context, repaired, outcome=ReleaseOutcome.FALLBACK,
                                reviews=reviews, repair=effect, replaced=candidate.id)
            return

        second, decision = await self._review_and_decide(context, repaired, phase="review_final")
        reviews.append(second)
        settled = self._settled(second, decision)
        if settled is None:
            outcome, reason, unresolved = (
                ReleaseOutcome.RELEASED_WITH_UNRESOLVED_REVIEW, None, decision.blocking_claim_ids,
            )
        elif settled[0] is ReleaseOutcome.REVIEWED:
            outcome, reason, unresolved = ReleaseOutcome.REPAIRED_AND_REVIEWED, None, ()
        else:
            outcome, reason, unresolved = settled
        await self._release(context, repaired, outcome=outcome, reviews=reviews, repair=effect,
                            replaced=candidate.id, ungated_reason=reason, unresolved=unresolved)

    @staticmethod
    def _settled(review: dict, decision) -> tuple[ReleaseOutcome, str | None, tuple] | None:
        """The outcome when this round ends the gate; ``None`` when it blocks."""
        if review.get("status") != "completed":
            return ReleaseOutcome.UNGATED, (
                f"the review did not complete ({review.get('status')}: "
                f"{review.get('reason', 'no reason given')})"
            ), ()
        if decision.ungated_reason:
            return ReleaseOutcome.UNGATED, decision.ungated_reason, ()
        if not decision.release:
            return None
        unjudged = decision.counts.get("cannot_judge", 0) + decision.counts.get("no_verdict", 0)
        if unjudged:
            return ReleaseOutcome.UNGATED, (
                f"nothing blocked, but {unjudged} claim(s) were not judged "
                "(cannot_judge or no verdict); a partial review is not a pass"
            ), ()
        return ReleaseOutcome.REVIEWED, None, ()

    async def _review_and_decide(self, context: RunContext, answer, *,
                                 phase: str = "review_initial"):
        """One reviewer turn over ``answer``, and what it decides."""
        review = await self._review_round(context, answer, phase=phase)
        decision = release_gate.release_decision(
            review.get("reviews") or (),
            reviewed_claim_ids=[
                c.claim_id for c in answer.claims
                if getattr(c.kind, "value", c.kind) in claim_review.REVIEWED_KINDS
            ],
        )
        review = {
            **{k: v for k, v in review.items() if k != "reviews"},
            "answer_id": answer.id,
            "verdicts": [
                {k: item.get(k) for k in ("claim_id", "verdict", "reason")}
                for item in review.get("reviews") or ()
            ],
            "counts": dict(decision.counts),
            "blocking_claim_ids": list(decision.blocking_claim_ids),
        }
        return review, decision

    async def _review_round(self, context: RunContext, answer, *,
                            phase: str = "review_initial") -> dict[str, Any]:
        """Dispatch the reviewer over ``answer`` and return what it recorded:
        ``completed`` with its reviews, or ``skipped``/``failed`` with why."""
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            bundle = await claim_review.review_bundle(
                uow, session_id=context.session_id, answer=answer
            )
        if not bundle["claims"]:
            return {"status": "skipped", "reason": "the answer makes no reviewable claim"}
        now = _now()
        remaining = (run.deadline_at - now).total_seconds() if run is not None else 0
        if remaining < self.CLAIM_REVIEW_FLOOR_S:
            return {"status": "skipped", "reason": "not enough run budget left",
                    "remaining_s": int(remaining)}
        deadline = min(run.deadline_at, now + timedelta(seconds=self.CLAIM_REVIEW_BUDGET_S))
        prompt = (
            claim_review.REVIEW_INSTRUCTIONS
            + "\n\nThe claims under review, each with the sources it cites:\n```json\n"
            + json.dumps(bundle, sort_keys=True, ensure_ascii=False, default=str)
            + "\n```"
        )

        async def reviewed(_context: RunContext) -> bool:
            async with self._db.unit_of_work() as uow:
                state = await uow.decision_states.get(context.run_id)
            review = state.claim_review if state else {}
            return review.get("status") == "completed" and review.get("answer_id") == answer.id

        try:
            await self._dispatch(
                replace(context, text="Review the claims listed in your instructions."),
                system_prompt=prompt, profile="claim_review", deadline=deadline,
                instructions_hash=content_sha256(claim_review.REVIEW_INSTRUCTIONS),
                has_product=reviewed, commit=_no_commit, phase=phase,
            )
        except Exception as exc:  # noqa: BLE001 - a failed review is an outcome, not a crash
            log.exception("claim review turn failed", extra={"run_id": context.run_id})
            return {"status": "failed", "reason": type(exc).__name__}
        async with self._db.unit_of_work() as uow:
            state = await uow.decision_states.get(context.run_id)
        review = dict(state.claim_review) if state else {}
        if review.get("status") == "completed" and review.get("answer_id") == answer.id:
            return review
        return {"status": "failed", "reason": "the reviewer submitted no verdicts"}

    async def _repair(self, context: RunContext, candidate, decision, *, system_prompt: str,
                      profile: str = "decision_support"):
        """One more turn of the answering agent, told what the review blocked.
        Returns the run's latest answer afterwards (the candidate itself if
        nothing new was accepted), or ``None`` with no budget for the turn."""
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
        if run is None or (run.deadline_at - _now()).total_seconds() < REPAIR_FLOOR_S:
            return None
        await decision_state_service.advance(
            self._db, context.run_id,
            lambda state: decision_state.request_repair(state, answer_id=candidate.id),
        )
        by_id = {c.claim_id: c for c in candidate.claims}
        blocking = [
            {"local_ref": getattr(by_id.get(cid), "local_ref", None) or cid,
             "claim": getattr(by_id.get(cid), "text", ""), "review": violation.message}
            for cid, violation in zip(decision.blocking_claim_ids, decision.violations)
        ]
        answer_view = {
            "answer_markdown": candidate.answer_markdown,
            "claims": [
                {k: v for k, v in claim.to_dict().items()
                 if k in ("local_ref", "kind", "text", "observation_id", "field_path",
                          "transform", "citation_ids", "evidence_spans")}
                for claim in candidate.claims
            ],
            "limitations": [limitation.to_dict() for limitation in candidate.limitations],
        }
        prompt = (
            system_prompt + "\n\n" + REPAIR_INSTRUCTIONS
            + "\n\nYour answer:\n```json\n"
            + json.dumps(answer_view, sort_keys=True, ensure_ascii=False, default=str)
            + "\n```\n\nBlocking:\n```json\n"
            + json.dumps(blocking, sort_keys=True, ensure_ascii=False)
            + "\n```"
        )

        async def repaired(_context: RunContext) -> bool:
            latest = await self._latest_answer(context)
            return latest is not None and latest.id != candidate.id

        try:
            await self._dispatch(
                replace(context, text="Repair your answer as your instructions describe."),
                system_prompt=prompt, profile=profile, deadline=run.deadline_at,
                instructions_hash=content_sha256(REPAIR_INSTRUCTIONS),
                has_product=repaired, commit=_no_commit, phase="semantic_repair",
            )
        except Exception:  # noqa: BLE001 - a failed repair leaves the candidate
            log.exception("repair turn failed", extra={"run_id": context.run_id})
        return await self._latest_answer(context) or candidate

    async def _release_latest(self, context: RunContext, *, outcome, ungated_reason=None) -> None:
        latest = await self._latest_answer(context)
        if latest is None:
            return
        async with self._db.unit_of_work() as uow:
            if await uow.answer_releases.get_for_run(context.run_id) is not None:
                return
        await self._release(context, latest, outcome=outcome, ungated_reason=ungated_reason)

    async def _release(
        self, context: RunContext, answer, *, outcome: ReleaseOutcome,
        reviews: list[dict] | tuple = (), repair: dict | None = None,
        replaced: str | None = None, ungated_reason: str | None = None,
        unresolved: tuple = (),
    ) -> None:
        policy = declared_policy()
        if policy is not ReleasePolicy.LEGACY:
            await self._record_publication(
                context, answer, policy=policy, outcome=outcome, reviews=reviews,
                repair=repair, replaced=replaced, ungated_reason=ungated_reason,
                unresolved=unresolved,
            )
            return
        release = AnswerRelease(
            answer_id=answer.id, run_id=context.run_id, session_id=context.session_id,
            outcome=outcome, released_at=_now(), reviews=tuple(reviews), repair=repair,
            unresolved_claim_ids=tuple(unresolved), ungated_reason=ungated_reason,
            replaced_answer_id=replaced,
        )
        async with self._db.unit_of_work() as uow:
            await uow.answer_releases.add(release)
            if not answer.is_fallback:
                # The fallback was announced when SubmitAnswer built it.
                uow.emit(
                    session_id=context.session_id, type=EventType.ANSWER_ACCEPTED,
                    entity_type="answer", entity_id=answer.id, run_id=context.run_id,
                    payload={"is_fallback": False,
                             "candidate_generation": answer.candidate_generation,
                             "release_outcome": outcome.value},
                )
            await uow.commit()
        if not answer.is_fallback and is_enabled("scientific_case_v1"):
            await self._record_released_in_case(context, answer)

    async def _record_publication(
        self, context: RunContext, answer, *, policy: ReleasePolicy, outcome: ReleaseOutcome,
        reviews, repair, replaced, ungated_reason, unresolved,
    ) -> None:
        """The one writer of a run's publication (#47). Decides by the pure
        policy, then in one unit of work records the run result and -- only
        for a published answer -- the compatible release row and
        answer.accepted. Idempotent on run_id: a retried commit rereads."""
        from ...domain.run_result import (
            Disposition,
            ExecutionOutcome,
            RunResultV1,
            decide_publication,
            knowledge_outcome_of,
            review_status_of,
        )

        status = review_status_of(outcome.value, ungated_reason=ungated_reason)
        decision = decide_publication(
            policy, review_status=status, has_candidate=answer is not None,
            is_fallback=bool(answer is not None and answer.is_fallback),
        )
        async with self._db.unit_of_work() as uow:
            if await uow.run_results.get(context.run_id) is not None:
                return
            state = await uow.decision_states.get(context.run_id)
            preview = (
                decision_state.finalize(state, run_status="completed") if state is not None
                else None
            )
            stop = preview.stop_reason if preview is not None else None
            public = (
                answer.id if decision.disposition is Disposition.PUBLISH
                or (answer is not None and answer.is_fallback) else None
            )
            result = RunResultV1(
                run_id=context.run_id, session_id=context.session_id,
                disposition=decision.disposition, review_status=decision.review_status,
                execution_outcome=(
                    ExecutionOutcome.BUDGET_EXHAUSTED if stop == "budget_exhausted"
                    else ExecutionOutcome.COMPLETED
                ),
                knowledge_outcome=knowledge_outcome_of(
                    stop, published=decision.disposition is Disposition.PUBLISH,
                ),
                release_policy=policy, created_at=_now(),
                public_answer_id=public,
                candidate_answer_id=answer.id if answer is not None else None,
                detail={
                    "reason": decision.reason,
                    "release_outcome": outcome.value,
                    "reviews": [dict(r) for r in reviews],
                    "repair": dict(repair) if repair else None,
                    "unresolved_claim_ids": list(unresolved),
                    "ungated_reason": ungated_reason,
                    "replaced_answer_id": replaced,
                    "confirmed_facts": confirmed_facts(answer),
                    "needed_inputs": list(
                        (preview.declared_stop if preview is not None else {})
                        .get("needed_inputs") or ()
                    ),
                    "stop_reason": stop,
                },
            )
            stored = await uow.run_results.add_if_absent(result)
            published = stored.disposition is Disposition.PUBLISH
            if published:
                await uow.answer_releases.add(AnswerRelease(
                    answer_id=answer.id, run_id=context.run_id, session_id=context.session_id,
                    outcome=outcome,
                    released_at=_now(), reviews=tuple(reviews), repair=repair,
                    unresolved_claim_ids=tuple(unresolved),
                    ungated_reason=ungated_reason, replaced_answer_id=replaced,
                ))
                if not answer.is_fallback:
                    uow.emit(
                        session_id=context.session_id, type=EventType.ANSWER_ACCEPTED,
                        entity_type="answer", entity_id=answer.id, run_id=context.run_id,
                        payload={"is_fallback": False,
                                 "candidate_generation": answer.candidate_generation,
                                 "release_outcome": outcome.value,
                                 "release_policy": policy.value},
                    )
            uow.emit(
                session_id=context.session_id, type=EventType.RUN_RESULT_RECORDED,
                entity_type="run", entity_id=context.run_id, run_id=context.run_id,
                payload={"disposition": stored.disposition.value,
                         "review_status": stored.review_status.value,
                         "public_answer_id": stored.public_answer_id,
                         "release_policy": policy.value},
            )
            await uow.commit()
        if published and not answer.is_fallback and is_enabled("scientific_case_v1"):
            await self._record_released_in_case(context, answer, promote=True)

    async def _record_terminal_result(self, context: RunContext, *, policy: ReleasePolicy,
                                      run_status: str, failure: BaseException | None) -> None:
        """The result of a run that did not commit: withheld, with the typed
        execution outcome. Never raises -- the run's own failure stands."""
        from ...domain.run_result import (
            Disposition,
            KnowledgeOutcome,
            ReviewStatus,
            RunResultV1,
            execution_outcome_of,
        )

        try:
            async with self._db.unit_of_work() as uow:
                if await uow.run_results.get(context.run_id) is not None:
                    return
                candidate = await uow.answers.get_for_run(context.run_id)
                code = getattr(failure, "code", None) if failure is not None else None
                result = RunResultV1(
                    run_id=context.run_id, session_id=context.session_id,
                    disposition=Disposition.WITHHOLD,
                    review_status=ReviewStatus.NOT_COMPLETED,
                    execution_outcome=execution_outcome_of(code, run_status=run_status),
                    knowledge_outcome=KnowledgeOutcome.NONE,
                    release_policy=policy, created_at=_now(),
                    candidate_answer_id=candidate.id if candidate is not None else None,
                    detail={
                        "reason": "the run ended before an answer could be published",
                        "failure_code": code or (
                            "cancelled" if run_status == "cancelled" else type(failure).__name__
                            if failure is not None else None
                        ),
                    },
                )
                stored = await uow.run_results.add_if_absent(result)
                uow.emit(
                    session_id=context.session_id, type=EventType.RUN_RESULT_RECORDED,
                    entity_type="run", entity_id=context.run_id, run_id=context.run_id,
                    payload={"disposition": stored.disposition.value,
                             "review_status": stored.review_status.value,
                             "public_answer_id": None, "release_policy": policy.value},
                )
                await uow.commit()
        except Exception:  # noqa: BLE001 - the run's own outcome is what matters
            log.exception("could not record the run result", extra={"run_id": context.run_id})

    async def _record_released_in_case(self, context: RunContext, answer, *,
                                       promote: bool = False) -> None:
        """The released answer's relations and citations join the case ledger
        -- what submit_grounded_answer does at once without the gate, done here
        so a candidate a repair replaced never enters the case."""
        try:
            async with self._db.unit_of_work() as uow:
                drafts = await uow.answer_drafts.list_for_run(context.run_id)
                cited = await scientific_case_service.cited_sources(
                    uow, session_id=context.session_id, answer=answer,
                )
            payload = next(
                (d["payload"] for d in drafts if d.get("answer_id") == answer.id), {}
            )
            ledger = scientific_case_service.answer_ledger_updates(
                payload.get("evidence_relations") or (), cited, run_id=context.run_id,
            )

            def updates_for(case):
                updates = list(ledger(case))
                if promote:
                    # #47: the run's proposed conclusion becomes the case's
                    # trusted one only now, with the published answer.
                    updates.append(scientific_case_service.update(
                        "promote_conclusion", actor="server", run_id=context.run_id,
                    ))
                return updates

            await scientific_case_service.advance(
                self._db, session_id=context.session_id, run_id=context.run_id,
                updates_for=updates_for,
            )
        except Exception:  # noqa: BLE001 - bookkeeping must not fail the release
            log.exception("could not record the released answer in the case",
                          extra={"run_id": context.run_id})


#: Claim kinds whose value the server read from an artifact itself: a
#: withheld answer may still show these, because no review is needed to
#: trust a value the server resolved.
_SERVER_RESOLVED_KINDS = frozenset({"numeric", "classification", "comparison"})


def confirmed_facts(answer) -> list[dict[str, Any]]:
    """The facts of a candidate that stand without its review: values the
    server resolved from an observation. Never its prose."""
    if answer is None:
        return []
    facts = []
    for claim in answer.claims:
        kind = getattr(claim.kind, "value", claim.kind)
        if kind in _SERVER_RESOLVED_KINDS and claim.observation_id and claim.rendered_value:
            facts.append({"text": claim.text, "value": claim.rendered_value,
                          "observation_id": claim.observation_id,
                          "field_path": claim.field_path})
    return facts
