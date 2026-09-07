"""submit_grounded_answer's workflow (plan sections 8.4, 9.5).

Correction policy in one sentence: a candidate is validated, an invalid one
returns typed violations for a single correction attempt, and a second invalid
attempt ends the run with a server-authored fallback — never a third try, and
never an evaluator agent grading the model's own work (plan section 9.5).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Mapping

from ..config import PolicySettings
from ..domain.answer import GroundedAnswer
from ..domain.errors import AnswerValidationFailed, Conflict, SessionNotFound, Violation
from ..domain.evidence import EvidenceRecord
from ..domain.events import EventType
from ..domain.observation import Observation
from ..validation.answer_validator import AnswerValidationResult, validate_candidate
from ..validation.fallback import build_fallback_answer
from ..validation.wire import GroundedAnswerCandidate

SUBMIT_TOOL_NAME = "submit_grounded_answer"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SubmitOutcome:
    answer: GroundedAnswer
    is_fallback: bool


class SubmitAnswer:
    def __init__(self, database, settings: PolicySettings) -> None:
        self._db = database
        self._settings = settings

    async def execute(
        self, *, session_id: str, run_id: str, candidate: GroundedAnswerCandidate, language: str = "en",
    ) -> SubmitOutcome:
        async with self._db.unit_of_work() as uow:
            session = await uow.sessions.get_unscoped(session_id)
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)

            if await uow.answers.get_for_run(run_id) is not None:
                raise Conflict(
                    "this run has already committed an answer; it cannot be overwritten",
                    run_id=run_id,
                )

            generation = await self._attempt_number(uow, run_id)
            observations_by_id = await self._resolve_observations(uow, session_id, candidate)
            evidence_by_id = await self._resolve_evidence(uow, session_id, candidate)
            read_evidence_ids = await self._resolve_read_evidence_ids(uow, run_id)

            result = validate_candidate(
                candidate,
                session_id=session_id, run_id=run_id, candidate_generation=generation,
                observations_by_id=observations_by_id, evidence_by_id=evidence_by_id,
                language=language, now=_now(), read_evidence_ids=read_evidence_ids,
            )
            result = await self._reject_claim_id_collisions(uow, candidate, result)

            if result.ok:
                await uow.answers.add(result.answer)
                uow.emit(
                    session_id=session_id, type=EventType.ANSWER_ACCEPTED,
                    entity_type="answer", entity_id=result.answer.id, run_id=run_id,
                    payload={"is_fallback": False, "candidate_generation": generation},
                )
                await uow.commit()
                return SubmitOutcome(result.answer, is_fallback=False)

            if generation >= self._settings.max_answer_candidates_per_run:
                fallback = await self._build_fallback(uow, session, session_id, run_id, generation, language)
                await uow.answers.add(fallback)
                uow.emit(
                    session_id=session_id, type=EventType.ANSWER_REJECTED, entity_type="answer",
                    entity_id=fallback.id, run_id=run_id,
                    payload={
                        "candidate_generation": generation,
                        "violations": [v.to_dict() for v in result.violations],
                    },
                )
                uow.emit(
                    session_id=session_id, type=EventType.ANSWER_ACCEPTED, entity_type="answer",
                    entity_id=fallback.id, run_id=run_id,
                    payload={"is_fallback": True, "candidate_generation": generation},
                )
                await uow.commit()
                return SubmitOutcome(fallback, is_fallback=True)

            uow.emit(
                session_id=session_id, type=EventType.ANSWER_REJECTED, entity_type="run",
                entity_id=run_id, run_id=run_id,
                payload={
                    "candidate_generation": generation,
                    "violations": [v.to_dict() for v in result.violations],
                },
            )
            await uow.commit()

        attempts_remaining = self._settings.max_answer_candidates_per_run - generation
        raise AnswerValidationFailed(
            f"candidate {generation} did not pass validation ({len(result.violations)} "
            f"violation(s), listed in details.violations). Correct exactly those and call "
            f"submit_grounded_answer again — {attempts_remaining} attempt(s) remain before this "
            "run ends with a deterministic fallback answer instead of yours.",
            violations=list(result.violations),
            candidate_generation=generation,
            attempts_remaining=attempts_remaining,
        )

    # --- helpers -------------------------------------------------------

    async def _attempt_number(self, uow, run_id: str) -> int:
        """1-indexed. Counts finished prior calls to this tool for this run, so
        the cap survives a retried tool call rather than resetting on replay."""
        calls = await uow.tool_calls.list_for_run(run_id)
        prior = [
            c for c in calls
            if c["tool_name"] == SUBMIT_TOOL_NAME and c["status"] in ("completed", "error")
        ]
        return len(prior) + 1

    async def _resolve_observations(
        self, uow, session_id: str, candidate: GroundedAnswerCandidate
    ) -> Mapping[str, Observation]:
        ids = {c.observation_id for c in candidate.claims if c.observation_id}
        resolved: dict[str, Observation] = {}
        for observation_id in ids:
            observation = await uow.observations.get(observation_id, session_id=session_id)
            if observation is not None:
                resolved[observation_id] = observation
        return resolved

    async def _resolve_evidence(
        self, uow, session_id: str, candidate: GroundedAnswerCandidate
    ) -> Mapping[str, EvidenceRecord]:
        ids = {e for c in candidate.claims for e in c.citation_ids}
        resolved: dict[str, EvidenceRecord] = {}
        for evidence_id in ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is not None:
                resolved[evidence_id] = record
        return resolved

    async def _resolve_read_evidence_ids(self, uow, run_id: str) -> frozenset[str]:
        """W3-07: which evidence ids this run actually read via
        get_evidence_record, not merely saw in a search result — the source
        of truth validate_citations' read_evidence_ids check is built from."""
        calls = await uow.tool_calls.list_for_run(run_id)
        read: set[str] = set()
        for call in calls:
            if call["tool_name"] == "get_evidence_record" and call["status"] == "completed":
                read.update(call.get("observation_ids") or ())
        return frozenset(read)

    async def _reject_claim_id_collisions(
        self, uow, candidate: GroundedAnswerCandidate, result: AnswerValidationResult
    ) -> AnswerValidationResult:
        """A model is told to "make one up" for claim_id (tools/definitions/answer.py) —
        it has no reason to know that id must also be unique against every
        other answer this deployment has ever stored, not just within its own
        candidate (a duplicate within one candidate is already caught by
        validate_candidate's own check). A live sweep produced exactly this
        collision (2026-09-05): a low-entropy self-chosen id from one task's
        answer reused in an unrelated one raised an unhandled
        `sqlite3.IntegrityError` on insert, turning a correctable mistake into
        a hard run failure instead of a normal one-more-try violation.
        """
        collisions = [
            claim.claim_id for claim in candidate.claims
            if await uow.answers.claim_id_exists(claim.claim_id)
        ]
        if not collisions:
            return result
        extra = [
            Violation(
                "claim_id_not_unique",
                f"claim_id {claim_id!r} is already used by a different, unrelated answer; "
                "choose a fresh 32-hex-character id instead of reusing one",
                path=f"claims[{claim_id}].claim_id",
            )
            for claim_id in collisions
        ]
        return replace(result, violations=(*result.violations, *extra), answer=None)

    async def _build_fallback(
        self, uow, session, session_id: str, run_id: str, generation: int, language: str
    ) -> GroundedAnswer:
        observations: list[Observation] = []
        if session.active_analysis_id:
            observations = list(await uow.observations.list_for_analysis(session.active_analysis_id))
        answer = build_fallback_answer(
            session_id=session_id, run_id=run_id, observations=observations,
            language=language, now=_now(),
        )
        # candidate_generation belongs to the run's attempt sequence, same as a
        # model-authored candidate would have used.
        from dataclasses import replace

        return replace(answer, candidate_generation=generation)
