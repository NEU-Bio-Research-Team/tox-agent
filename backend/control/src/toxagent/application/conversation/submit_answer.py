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

from ...platform.config import PolicySettings
from ...domain.answer import GroundedAnswer
from ...domain.errors import AnswerValidationFailed, Conflict, SessionNotFound, Violation
from ...domain.evidence import EvidenceRecord
from ...domain.events import EventType
from ...domain.evidence_relation import (
    Applicability,
    Directness,
    EvidenceRelationAssessment,
    EvidenceScope,
    RelationLabel,
    SourceClass,
    SourceRef,
    Strength,
    new_proposition_id,
    scope_conflicts,
)
from ...domain.development_posture import DevelopmentPosture, DevelopmentScope, PostureValue
from ...domain.observation import Observation
from ...platform.flags import is_enabled
from ...research.measurement import is_measured_value
from ..investigation import evidence_text
from ...validation.answer.validator import AnswerValidationResult, validate_candidate
from ...validation.fallback import build_fallback_answer
from ...validation.prohibited_claims import validate_posture_wording
from ...validation.answer.candidate_wire import GroundedAnswerCandidate
from ...validation.answer.claim_resolver import resolve_draft
from ...validation.answer.coverage import passage_values
from ...validation.answer.draft_wire import (
    DevelopmentPostureInputV2,
    EvidenceRelationInputV2,
    GroundedAnswerDraftV2,
)

SUBMIT_TOOL_NAME = "submit_grounded_answer"

#: A posture that commits a direction with some confidence. P1-03: such a
#: posture may not rest on the model's own synthesis alone.
_COMMITTING_POSTURES = frozenset({"proceed", "deprioritize"})
_COMMITTING_CONFIDENCE = frozenset({"moderate", "strong"})


def _synthesis_only_posture(posture, relations) -> list[Violation]:
    bearing = [r for r in relations if r.relation in ("supports", "contradicts")]
    if (
        posture.value.value in _COMMITTING_POSTURES
        and posture.confidence_band in _COMMITTING_CONFIDENCE
        and bearing
        and all(r.source_class == "agent_synthesis" for r in bearing)
    ):
        return [
            Violation(
                "posture_rests_on_synthesis_only",
                f"a {posture.confidence_band} {posture.value.value} posture needs at least one "
                "supporting or contradicting relation from a direct source (predictor, "
                "explanation, external evidence or report), not only agent_synthesis; lower the "
                "confidence or cite the sources directly",
                path="development_posture",
            )
        ]
    return []


async def _record_answer_in_state(uow, run_id: str, *, relations, is_fallback: bool,
                                  generation: int) -> None:
    """Resolve the run's DecisionSupportStateV1 from the committed answer.

    Inside the answer's own unit of work so the two cannot disagree; a lost
    race on the state is logged and skipped rather than failing the answer.
    """
    from ...domain import decision_state as ds
    from ..investigation import decision_state_service

    payload = decision_state_service.relations_payload(relations)
    try:
        await decision_state_service.advance_in(
            uow, run_id,
            lambda state: ds.apply_answer(
                state, relations=payload, is_fallback=is_fallback,
                candidate_generation=generation,
            ),
        )
    except Conflict:
        decision_state_service.log.warning(
            "decision state changed under an answer commit; resolution skipped",
            extra={"run_id": run_id},
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SubmitOutcome:
    answer: GroundedAnswer
    is_fallback: bool
    #: W6/plan section 12.1: carried separately from ``answer`` so a renderer
    #: can present it as its own labeled block (value/scope/confidence/
    #: conditions) rather than inline prose that could be read as folded into
    #: the answer's ordinary claims.
    development_posture: DevelopmentPosture | None = None


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
            # #47: the draft exactly as the model sent it, before resolution
            # replaces ``candidate`` with the internal form.
            raw_draft = (candidate.schema_version, candidate.model_dump(mode="json"))
            observations_by_id = await self._resolve_observations(uow, session_id, candidate)
            evidence_by_id = await self._resolve_evidence(uow, session_id, candidate)
            read_evidence_ids = await self._resolve_read_evidence_ids(uow, run_id)
            # Captured before v2 resolution reassigns ``candidate`` below: v1
            # (GroundedAnswerCandidate) carries no evidence_relations field at
            # all, so this is empty on that path.
            proposed_relations = (
                candidate.evidence_relations
                if isinstance(candidate, GroundedAnswerDraftV2) else []
            )
            proposed_posture = (
                candidate.development_posture
                if isinstance(candidate, GroundedAnswerDraftV2) else None
            )

            issued_ids: Mapping[str, str] = {}
            claimed_values: tuple[float, ...] = ()
            span_violations: list[Violation] = []
            if isinstance(candidate, GroundedAnswerDraftV2):
                # The server issues the identifiers and reads the values before
                # anything is validated, so every check below runs against
                # facts the model could not have got wrong (P1-4).
                resolved = resolve_draft(
                    candidate,
                    observations_by_id=observations_by_id,
                    language=language,
                )
                if resolved.candidate is None:
                    return await self._reject(
                        uow, session, session_id, run_id, generation,
                        list(resolved.violations), language, raw_draft=raw_draft,
                    )
                candidate = resolved.candidate
                issued_ids = resolved.issued_ids
                claimed_values = resolved.claimed_values
                span_values, span_violations = await self._resolve_evidence_spans(
                    uow, session_id=session_id, run_id=run_id, candidate=candidate,
                )
                claimed_values = (*claimed_values, *span_values)

            result = validate_candidate(
                candidate,
                session_id=session_id, run_id=run_id, candidate_generation=generation,
                observations_by_id=observations_by_id, evidence_by_id=evidence_by_id,
                language=language, now=_now(), read_evidence_ids=read_evidence_ids,
                claimed_values=claimed_values,
                unread_article_ids=await self._unread_articles(
                    uow, session_id=session_id, evidence_by_id=evidence_by_id,
                ),
            )
            result = await self._reject_claim_id_collisions(uow, candidate, result)
            if span_violations:
                result = replace(
                    result, violations=(*result.violations, *span_violations), answer=None
                )

            resolved_relations: list[EvidenceRelationAssessment] = []
            if result.ok and proposed_relations:
                resolved_relations, relation_violations = await self._resolve_evidence_relations(
                    uow, session_id=session_id, run_id=run_id, proposed=proposed_relations,
                    read_evidence_ids=read_evidence_ids,
                )
                if relation_violations:
                    result = replace(
                        result, violations=(*result.violations, *relation_violations), answer=None
                    )

            posture: DevelopmentPosture | None = None
            if result.ok and proposed_posture is not None:
                posture, posture_violations = self._build_development_posture(
                    proposed_posture, issued_ids=issued_ids,
                )
                if posture is not None:
                    posture_violations = posture_violations + validate_posture_wording(
                        candidate.answer_markdown, posture
                    ) + _synthesis_only_posture(posture, proposed_relations)
                if posture_violations:
                    posture = None
                    result = replace(
                        result, violations=(*result.violations, *posture_violations), answer=None
                    )

            if result.ok:
                await uow.answers.add(result.answer)
                await self._record_draft(
                    uow, run_id=run_id, session_id=session_id, generation=generation,
                    raw_draft=raw_draft, outcome="accepted", answer_id=result.answer.id,
                    violations=(),
                )
                for assessment in resolved_relations:
                    await uow.evidence_relations.add(assessment)
                if posture is not None:
                    await uow.development_postures.add(
                        posture, answer_id=result.answer.id, session_id=session_id,
                        run_id=run_id, now=_now(),
                    )
                uow.emit(
                    session_id=session_id, type=EventType.ANSWER_ACCEPTED,
                    entity_type="answer", entity_id=result.answer.id, run_id=run_id,
                    payload={"is_fallback": False, "candidate_generation": generation},
                )
                await _record_answer_in_state(
                    uow, run_id, relations=proposed_relations, is_fallback=False,
                    generation=generation,
                )
                await uow.commit()
                return SubmitOutcome(
                    result.answer, is_fallback=False, development_posture=posture
                )

            await self._record_draft(
                uow, run_id=run_id, session_id=session_id, generation=generation,
                raw_draft=raw_draft, outcome="rejected", answer_id=None,
                violations=result.violations,
            )
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
                await _record_answer_in_state(
                    uow, run_id, relations=(), is_fallback=True, generation=generation,
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

    async def _record_draft(
        self, uow, *, run_id: str, session_id: str, generation: int, raw_draft,
        outcome: str, answer_id: str | None, violations,
    ) -> None:
        """#47: keep the draft as sent, beside what the server made of it, so
        the raw draft, the accepted candidate and the released answer can each
        be graded. A retried tool call that replays a generation is already
        recorded and is not written twice."""
        schema_version, payload = raw_draft
        existing = {d["generation"] for d in await uow.answer_drafts.list_for_run(run_id)}
        if generation in existing:
            return
        await uow.answer_drafts.add(
            run_id=run_id, session_id=session_id, generation=generation,
            schema_version=schema_version, payload=payload, outcome=outcome,
            answer_id=answer_id, violations=[v.to_dict() for v in violations], now=_now(),
        )

    async def _reject(
        self, uow, session, session_id: str, run_id: str, generation: int,
        violations: list, language: str, *, raw_draft=None,
    ) -> SubmitOutcome:
        """The shared rejection path: one more attempt, or the fallback.

        Resolution failures take it too. A draft naming a field path that does
        not exist is exactly as correctable as a claim whose number was wrong,
        and giving it a different shape of error would spend a correction
        attempt on a message the model cannot act on.
        """
        if raw_draft is not None:
            await self._record_draft(
                uow, run_id=run_id, session_id=session_id, generation=generation,
                raw_draft=raw_draft, outcome="rejected", answer_id=None, violations=violations,
            )
        if generation >= self._settings.max_answer_candidates_per_run:
            fallback = await self._build_fallback(
                uow, session, session_id, run_id, generation, language
            )
            await uow.answers.add(fallback)
            uow.emit(
                session_id=session_id, type=EventType.ANSWER_REJECTED, entity_type="answer",
                entity_id=fallback.id, run_id=run_id,
                payload={
                    "candidate_generation": generation,
                    "violations": [v.to_dict() for v in violations],
                },
            )
            uow.emit(
                session_id=session_id, type=EventType.ANSWER_ACCEPTED, entity_type="answer",
                entity_id=fallback.id, run_id=run_id,
                payload={"is_fallback": True, "candidate_generation": generation},
            )
            await _record_answer_in_state(
                uow, run_id, relations=(), is_fallback=True, generation=generation,
            )
            await uow.commit()
            return SubmitOutcome(fallback, is_fallback=True)

        uow.emit(
            session_id=session_id, type=EventType.ANSWER_REJECTED, entity_type="run",
            entity_id=run_id, run_id=run_id,
            payload={
                "candidate_generation": generation,
                "violations": [v.to_dict() for v in violations],
            },
        )
        await uow.commit()
        attempts_remaining = self._settings.max_answer_candidates_per_run - generation
        raise AnswerValidationFailed(
            f"candidate {generation} did not pass validation ({len(violations)} "
            f"violation(s), listed in details.violations). Correct exactly those and call "
            f"submit_grounded_answer again — {attempts_remaining} attempt(s) remain before this "
            "run ends with a deterministic fallback answer instead of yours.",
            violations=list(violations),
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
            # get_chembl_activities (W9-13) returns each record whole, so a
            # record it returned has been read, not merely listed; a record
            # whose full text was read (#43) has been read too.
            if call["tool_name"] in (
                "get_evidence_record", "get_chembl_activities", "read_evidence_text",
            ) and call["status"] == "completed":
                read.update(call.get("observation_ids") or ())
        return frozenset(read)

    async def _unread_articles(
        self, uow, *, session_id: str, evidence_by_id: Mapping[str, EvidenceRecord],
    ) -> frozenset[str]:
        """#43: cited articles whose full text was not read. Only where it can
        be read at all (evidence_full_text_v1); otherwise the default
        evidence_scope_limited wording already says no full text was read."""
        if not is_enabled("evidence_full_text_v1"):
            return frozenset()
        unread: set[str] = set()
        for evidence_id, record in evidence_by_id.items():
            full_text = await uow.evidence_text.get(evidence_id, session_id=session_id)
            if evidence_text.text_not_read(record, full_text):
                unread.add(evidence_id)
        return frozenset(unread)

    async def _resolve_evidence_spans(
        self, uow, *, session_id: str, run_id: str, candidate: GroundedAnswerCandidate,
    ) -> tuple[tuple[float, ...], list[Violation]]:
        """#43: the passages claims quote, checked against the text the server
        holds. A span is accepted when this run read that record's text with
        read_evidence_text and the locator names a passage of it; the numbers
        in an accepted passage are then values the prose may quote."""
        spans = [
            (index, claim, evidence_id, locator)
            for index, claim in enumerate(candidate.claims)
            for evidence_id, locator in claim.evidence_spans
        ]
        if not spans:
            return (), []
        text_read: set[str] = set()
        for call in await uow.tool_calls.list_for_run(run_id):
            if call["tool_name"] == "read_evidence_text" and call["status"] == "completed":
                text_read.update(call.get("observation_ids") or ())
        values: list[float] = []
        violations: list[Violation] = []
        texts: dict[str, object] = {}
        for index, claim, evidence_id, locator in spans:
            path = f"claims[{index}].evidence_spans"
            if evidence_id not in texts:
                texts[evidence_id] = await uow.evidence_text.get(
                    evidence_id, session_id=session_id
                )
            full_text = texts[evidence_id]
            if evidence_id not in text_read or full_text is None or not full_text.was_read:
                violations.append(
                    Violation(
                        "evidence_span_not_read",
                        f"claim {claim.claim_id} quotes {evidence_id} at {locator!r}, but this "
                        "run did not read that record's text with read_evidence_text; read it "
                        "first, or cite the record without a span",
                        path=path, actual=locator,
                    )
                )
                continue
            passage = full_text.passage(locator)
            if passage is None:
                violations.append(
                    Violation(
                        "evidence_span_locator_unknown",
                        f"{evidence_id} has no passage at {locator!r}; use a locator "
                        "read_evidence_text returned",
                        path=path, actual=locator,
                    )
                )
                continue
            values.extend(passage_values(passage.text))
        return tuple(values), violations

    async def _resolve_evidence_relations(
        self, uow, *, session_id: str, run_id: str, proposed: list[EvidenceRelationInputV2],
        read_evidence_ids: frozenset[str] = frozenset(),
    ) -> tuple[list[EvidenceRelationAssessment], list[Violation]]:
        """W5-06: the model proposes, the server independently confirms
        ``source_id`` resolves to a real, session-owned artifact of the
        claimed kind before anything is persisted — an unresolved source is
        rejected outright, consuming the candidate the same way any other
        validation violation does (domain/evidence_relation.py's own
        docstring), not silently dropped or partially trusted.

        Relations that share identical ``proposition`` text within one draft
        share one server-minted ``proposition_id``, since the wire type
        deliberately does not let the model mint that id itself.
        """
        proposition_ids: dict[str, str] = {}
        assessments: list[EvidenceRelationAssessment] = []
        violations: list[Violation] = []
        # Sources that are measured values, whatever scope the model declared
        # for them: a contradiction involving one is held to the measurement
        # bar, so leaving the assay out of the scope cannot lower it.
        measured: set[str] = set()
        now = _now()
        for index, item in enumerate(proposed):
            if item.source_class == "agent_synthesis":
                # P1-03: a synthesis is a transformation. Every input must be a
                # real, session-owned artifact — and evidence must have been
                # read in this run, the same bar a citation has to clear.
                unresolved = [
                    ref for ref in item.input_source_refs
                    if not await self._input_ref_resolves(
                        uow, session_id=session_id, ref=ref, read_evidence_ids=read_evidence_ids,
                    )
                ]
                if unresolved:
                    violations.append(
                        Violation(
                            "agent_synthesis_lineage_unresolved",
                            f"input_source_refs {unresolved} do not resolve to session-owned "
                            "artifacts this run read; a synthesis must derive from real sources",
                            path=f"evidence_relations[{index}].input_source_refs",
                        )
                    )
                    continue
            resolved = await self._source_exists(
                uow, session_id=session_id, source_class=item.source_class, source_id=item.source_id,
            )
            if not resolved:
                violations.append(
                    Violation(
                        "evidence_relation_source_not_found",
                        f"source_id {item.source_id!r} does not resolve to a "
                        f"session-owned {item.source_class} this run can cite",
                        path=f"evidence_relations[{index}].source_id",
                    )
                )
                continue
            if item.source_class in ("external_experimental", "external_regulatory"):
                record = await uow.evidence.get(item.source_id, session_id=session_id)
                if record is not None and is_measured_value(record.normalized_facts):
                    measured.add(item.source_id)
            proposition_id = proposition_ids.setdefault(item.proposition, new_proposition_id())
            assessments.append(
                EvidenceRelationAssessment.create(
                    session_id=session_id,
                    run_id=run_id,
                    proposition_id=proposition_id,
                    source_ref=SourceRef(
                        source_class=SourceClass(item.source_class), source_id=item.source_id,
                    ),
                    relation=RelationLabel(item.relation),
                    directness=Directness(item.directness),
                    applicability=Applicability(item.applicability),
                    strength=Strength(item.strength),
                    reason_codes=tuple(item.reason_codes),
                    scope=EvidenceScope(
                        endpoint=item.endpoint, species=item.species, dose=item.dose,
                        use_context=item.use_context,
                        ligand_id=item.ligand_id, target_id=item.target_id,
                        target_variant=item.target_variant,
                        target_sequence_sha256=item.target_sequence_sha256,
                        assay_type=item.assay_type, readout=item.readout,
                        unit=item.unit, comparator=item.comparator,
                        experimental_context=item.experimental_context,
                    ),
                    input_refs=tuple(item.input_source_refs),
                    now=now,
                )
            )
        # A `contradicts` claims two sources cannot both hold, which requires
        # them to have measured the same thing. Where the scopes differ, the
        # model is told which axis differs and asked for `contextual` instead;
        # where they say too little to tell, it is told what to state. A false
        # contradiction is on the critical-error list, so either is a
        # violation rather than a silent downgrade.
        for conflict in scope_conflicts(assessments, measured_source_ids=frozenset(measured)):
            position = next(
                (
                    index
                    for index, assessment in enumerate(assessments)
                    if assessment.id == conflict.contradicting_relation_id
                ),
                0,
            )
            if conflict.reason == "scope_differs":
                violations.append(
                    Violation(
                        "evidence_relation_contradiction_out_of_scope",
                        f"{conflict.describe()}; state it as 'contextual' and name the "
                        "differing scope, or correct the scope if they do match",
                        path=f"evidence_relations[{position}].relation",
                    )
                )
            else:
                violations.append(
                    Violation(
                        "evidence_relation_contradiction_scope_not_assessed",
                        f"{conflict.describe()}, as each source states them; if a source "
                        "does not say, leave that field out and state the relation as "
                        "'contextual' instead -- do not fill a scope in to make it match",
                        path=f"evidence_relations[{position}].relation",
                    )
                )
        if violations:
            return [], violations
        return assessments, []

    def _build_development_posture(
        self, proposed: DevelopmentPostureInputV2, *, issued_ids: Mapping[str, str],
    ) -> tuple[DevelopmentPosture | None, list[Violation]]:
        """Local refs are already confirmed to exist within the draft by
        ``GroundedAnswerDraftV2``'s own validator; ``issued_ids`` maps them to
        the real, server-minted claim ids the domain type requires
        (W6 — the same "the model never mints an id" principle
        ``resolve_draft`` already applies to claims themselves)."""
        try:
            return DevelopmentPosture(
                value=PostureValue(proposed.value),
                scope=DevelopmentScope(proposed.scope),
                confidence_band=proposed.confidence_band,
                basis_claim_ids=tuple(issued_ids[ref] for ref in proposed.basis_local_refs),
                contrary_claim_ids=tuple(issued_ids[ref] for ref in proposed.contrary_local_refs),
                rationale=proposed.rationale,
                conditions=tuple(proposed.conditions),
                recommended_next_steps=tuple(proposed.recommended_next_steps),
            ), []
        except ValueError as exc:
            return None, [
                Violation("development_posture_invalid", str(exc), path="development_posture")
            ]

    async def _source_exists(
        self, uow, *, session_id: str, source_class: str, source_id: str,
    ) -> bool:
        if source_class in ("predictor_fact", "explanation_fact"):
            return await uow.observations.get(source_id, session_id=session_id) is not None
        if source_class in ("external_experimental", "external_regulatory"):
            return await uow.evidence.get(source_id, session_id=session_id) is not None
        if source_class == "report_fact":
            return await uow.reports.get_artifact(source_id, session_id=session_id) is not None
        # agent_synthesis: not a stored artifact itself. Its provenance is its
        # input_source_refs, resolved one by one in _resolve_evidence_relations
        # before this is reached (P1-03).
        return source_class == "agent_synthesis"

    async def _input_ref_resolves(
        self, uow, *, session_id: str, ref: str, read_evidence_ids: frozenset[str],
    ) -> bool:
        kind, _, identifier = ref.partition(":")
        if kind == "observation":
            return await uow.observations.get(identifier, session_id=session_id) is not None
        if kind == "evidence":
            return identifier in read_evidence_ids and (
                await uow.evidence.get(identifier, session_id=session_id) is not None
            )
        if kind == "report":
            return await uow.reports.get_artifact(identifier, session_id=session_id) is not None
        return False

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
