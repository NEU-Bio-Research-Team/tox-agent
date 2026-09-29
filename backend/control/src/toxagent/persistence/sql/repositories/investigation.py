"""Decision state, scientific cases, development postures, skill drafts, and the superseded kernel's store (ADR 0011)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import and_, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.errors import Conflict
from ....domain.investigation import (
    CaseState,
    ConflictStatus,
    Coverage,
    EvidenceConflict,
    EvidenceGap,
    GapSeverity,
    GoalType,
    InvestigationPlan,
    InvestigationStep,
    StepStatus,
)
from ....domain.observation import Observation
from ....superseded.kernel import KernelTransition
from ...schema import (
    case_revisions,
    cases,
    decision_support_states,
    development_postures,
    investigation_plans,
    investigation_steps,
    kernel_transitions,
    observations,
    scientific_case_dossiers,
    scientific_case_events,
    scientific_cases,
    skill_drafts,
)
from .. import mapping as m


class SqlInvestigationStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def save_case(self, case: CaseState, *, expected_revision: int | None) -> None:
        state = case.to_dict()
        row = {
            "id": case.id, "session_id": case.session_id, "goal": case.goal.value,
            "subject": dict(case.subject), "active_analysis_id": case.active_analysis_id,
            "active_plan_id": case.plan_id, "state": state, "revision": case.revision,
            "revision_reason": case.revision_reason, "created_at": case.created_at,
            "updated_at": case.updated_at,
        }
        if expected_revision is None:
            await self._conn.execute(insert(cases).values(row))
        else:
            values = dict(row)
            values.pop("id")
            values.pop("session_id")
            values.pop("created_at")
            result = await self._conn.execute(
                update(cases).where(and_(cases.c.id == case.id, cases.c.revision == expected_revision)).values(**values)
            )
            if result.rowcount == 0:
                raise Conflict("case changed underneath this write", case_id=case.id)
        await self._conn.execute(insert(case_revisions).values(
            case_id=case.id, revision=case.revision, reason=case.revision_reason,
            state=state, created_at=case.updated_at,
        ))

    async def get_case(self, case_id: str, *, session_id: str) -> CaseState | None:
        row = (await self._conn.execute(select(cases).where(and_(
            cases.c.id == case_id, cases.c.session_id == session_id,
        )))).mappings().first()
        if row is None:
            return None
        state = row["state"]
        conflicts = tuple(EvidenceConflict(
            item["id"], item["proposition"], tuple(item["claim_ids"]),
            ConflictStatus(item["status"]), item.get("resolution_reason"),
        ) for item in state.get("conflicts", ()))
        gaps = tuple(EvidenceGap(
            item["id"], item["question"], GapSeverity(item["severity"]), item["reason"],
        ) for item in state.get("gaps", ()))
        coverage = state.get("coverage", {})
        return CaseState(
            id=row["id"], session_id=row["session_id"], subject=row["subject"],
            goal=GoalType(row["goal"]), active_analysis_id=row["active_analysis_id"],
            questions=tuple(state.get("questions", ())),
            hypothesis_refs=tuple(state.get("hypothesis_refs", ())),
            claim_refs=tuple(state.get("claim_refs", ())), conflicts=conflicts, gaps=gaps,
            plan_id=row["active_plan_id"], coverage=Coverage(
                tuple(coverage.get("required_questions", ())),
                tuple(coverage.get("answered_questions", ())),
                tuple(coverage.get("unresolved_conflict_ids", ())),
                tuple(coverage.get("blocking_gap_ids", ())),
            ), action_refs=tuple(state.get("action_refs", ())), revision=row["revision"],
            revision_reason=row["revision_reason"], created_at=m.utc(row["created_at"]),
            updated_at=m.utc(row["updated_at"]),
        )

    async def save_plan(self, plan: InvestigationPlan) -> None:
        await self._conn.execute(insert(investigation_plans).values(
            id=plan.id, case_id=plan.case_id, revision=plan.revision,
            reason=plan.reason, created_at=plan.created_at,
        ))
        await self._conn.execute(insert(investigation_steps), [
            {
                "id": step.id, "plan_id": plan.id, "position": position,
                "question": step.question, "capability": step.capability,
                "input_refs": list(step.input_refs), "expected_output": step.expected_output,
                "success_condition": step.success_condition, "case_revision": step.case_revision,
                "status": step.status.value, "output_refs": list(step.output_refs),
                "failure_reason": step.failure_reason,
            }
            for position, step in enumerate(plan.steps)
        ])

    async def save_step(self, plan_id: str, step: InvestigationStep) -> None:
        result = await self._conn.execute(update(investigation_steps).where(and_(
            investigation_steps.c.id == step.id, investigation_steps.c.plan_id == plan_id,
        )).values(status=step.status.value, output_refs=list(step.output_refs),
                  failure_reason=step.failure_reason))
        if result.rowcount == 0:
            raise Conflict("investigation step does not exist", step_id=step.id)

    async def get_plan(self, plan_id: str) -> InvestigationPlan | None:
        """The committed plan and its steps, in order, with their outcomes.

        I31: the kernel could only ever plan afresh, so a control plane that
        restarted mid-investigation started over — a new plan, a new model
        turn, and every completed step's work paid for again. Reading the plan
        back is what makes resuming possible.
        """
        row = (
            await self._conn.execute(
                select(investigation_plans).where(investigation_plans.c.id == plan_id)
            )
        ).mappings().first()
        if row is None:
            return None
        step_rows = (
            await self._conn.execute(
                select(investigation_steps)
                .where(investigation_steps.c.plan_id == plan_id)
                .order_by(investigation_steps.c.position)
            )
        ).mappings().all()
        steps = tuple(
            InvestigationStep(
                step["id"], step["question"], step["capability"],
                tuple(step["input_refs"] or ()), step["expected_output"],
                step["success_condition"], step["case_revision"],
                StepStatus(step["status"]), tuple(step["output_refs"] or ()),
                step["failure_reason"],
            )
            for step in step_rows
        )
        return InvestigationPlan(
            id=row["id"], case_id=row["case_id"], revision=row["revision"],
            steps=steps, reason=row["reason"], created_at=m.utc(row["created_at"]),
        )

    async def load_observations(self, ids: Sequence[str]) -> tuple[Observation, ...]:
        """Observations a completed step already produced, in the given order.

        Committed observations are immutable and product-owned, so a resumed
        investigation reads them rather than calling the predictor or the
        evidence provider again for answers it already has.
        """
        if not ids:
            return ()
        rows = (
            await self._conn.execute(
                select(observations).where(observations.c.id.in_(list(ids)))
            )
        ).mappings().all()
        by_id = {row["id"]: m.row_to_observation(row) for row in rows}
        return tuple(by_id[i] for i in ids if i in by_id)

    async def append_transition(self, case_id: str, transition: KernelTransition) -> None:
        await self._conn.execute(insert(kernel_transitions).values(
            case_id=case_id, state=transition.state.value, detail=transition.detail,
            occurred_at=transition.occurred_at,
        ))


class SqlDecisionStateStore:
    """DecisionSupportStateV1 rows (domain/decision_state.py), one per run.

    Writes are compare-and-set on ``revision``: a writer that read revision N
    may only store revision N+1. Tool results for one run can finish
    concurrently, and a lost update here would drop a counted search or an
    available artifact without anyone noticing.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get(self, run_id: str):
        from ....domain.decision_state import DecisionSupportStateV1

        row = (
            await self._conn.execute(
                select(decision_support_states).where(decision_support_states.c.run_id == run_id)
            )
        ).mappings().first()
        return DecisionSupportStateV1.from_dict(row["state"]) if row is not None else None

    async def latest_for_session(self, session_id: str, *, exclude_run_id: str | None = None):
        from ....domain.decision_state import DecisionSupportStateV1

        query = (
            select(decision_support_states)
            .where(decision_support_states.c.session_id == session_id)
            .where(decision_support_states.c.stop_reason.is_not(None))
            .order_by(decision_support_states.c.updated_at.desc())
            .limit(1)
        )
        if exclude_run_id is not None:
            query = query.where(decision_support_states.c.run_id != exclude_run_id)
        row = (await self._conn.execute(query)).mappings().first()
        return DecisionSupportStateV1.from_dict(row["state"]) if row is not None else None

    async def list_for_session(self, session_id: str, *, limit: int = 20):
        from ....domain.decision_state import DecisionSupportStateV1

        rows = (
            await self._conn.execute(
                select(decision_support_states)
                .where(decision_support_states.c.session_id == session_id)
                .order_by(decision_support_states.c.updated_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [DecisionSupportStateV1.from_dict(r["state"]) for r in rows]

    async def put(self, state, *, expected_revision: int | None, now: datetime) -> None:
        values = {
            "revision": state.revision,
            "stop_reason": state.stop_reason,
            "state": state.to_dict(),
            "updated_at": now,
        }
        if expected_revision is None:
            await self._conn.execute(
                insert(decision_support_states).values(
                    run_id=state.run_id, session_id=state.session_id, created_at=now, **values,
                )
            )
            return
        result = await self._conn.execute(
            update(decision_support_states)
            .where(decision_support_states.c.run_id == state.run_id)
            .where(decision_support_states.c.revision == expected_revision)
            .where(decision_support_states.c.stop_reason.is_(None))
            .values(**values)
        )
        if result.rowcount != 1:
            raise Conflict(
                "the decision-support state changed or is final; re-read and retry",
                run_id=state.run_id, expected_revision=expected_revision,
            )


class SqlScientificCaseStore:
    """ScientificCaseV1 snapshots, their append-only log, and run dossiers.

    ``append`` is the only write path for a case: it inserts the updates as
    event rows and moves the snapshot from ``expected_revision`` to the new
    revision in the same unit of work. A writer that read an older revision
    loses with ``Conflict`` and must re-read — a lost update here would drop a
    hypothesis or an evidence entry without anyone noticing, and the event
    table's ``(case_id, revision)`` key makes the loss impossible even if the
    snapshot check were bypassed.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    @staticmethod
    def _case(row):
        from ....domain.scientific_case import ScientificCaseV1

        return ScientificCaseV1.from_dict(row["state"]) if row is not None else None

    async def get(self, case_id: str, *, session_id: str):
        row = (
            await self._conn.execute(
                select(scientific_cases)
                .where(scientific_cases.c.id == case_id)
                .where(scientific_cases.c.session_id == session_id)
            )
        ).mappings().first()
        return self._case(row)

    async def open_for_subject(self, session_id: str, subject_key: str):
        row = (
            await self._conn.execute(
                select(scientific_cases)
                .where(scientific_cases.c.session_id == session_id)
                .where(scientific_cases.c.subject_key == subject_key)
                .where(scientific_cases.c.status == "open")
                .order_by(scientific_cases.c.updated_at.desc())
                .limit(1)
            )
        ).mappings().first()
        return self._case(row)

    async def list_for_session(self, session_id: str, *, limit: int = 20):
        rows = (
            await self._conn.execute(
                select(scientific_cases)
                .where(scientific_cases.c.session_id == session_id)
                .order_by(scientific_cases.c.updated_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [self._case(row) for row in rows]

    async def append(self, case, updates, *, expected_revision: int | None, now: datetime) -> None:
        """Store ``case`` as the fold of ``updates`` over ``expected_revision``."""
        if not updates:
            return
        first = case.revision - len(updates) + 1
        if first != (expected_revision or 0) + 1:
            raise Conflict(
                "the updates do not continue the case from its expected revision",
                case_id=case.id, expected_revision=expected_revision, revision=case.revision,
            )
        values = {
            "status": case.status, "revision": case.revision, "state": case.to_dict(),
            "updated_at": now,
        }
        if expected_revision is None:
            await self._conn.execute(
                insert(scientific_cases).values(
                    id=case.id, session_id=case.session_id, subject_key=case.subject_key,
                    created_at=now, **values,
                )
            )
        else:
            result = await self._conn.execute(
                update(scientific_cases)
                .where(scientific_cases.c.id == case.id)
                .where(scientific_cases.c.revision == expected_revision)
                .values(**values)
            )
            if result.rowcount != 1:
                raise Conflict(
                    "the scientific case changed; re-read and retry",
                    case_id=case.id, expected_revision=expected_revision,
                )
        await self._conn.execute(
            insert(scientific_case_events),
            [
                {
                    "case_id": case.id, "revision": first + index, "op": item.op,
                    "actor": item.actor, "run_id": item.run_id,
                    "payload": {"payload": dict(item.payload), "at": item.at},
                    "created_at": now,
                }
                for index, item in enumerate(updates)
            ],
        )

    async def events(self, case_id: str, *, session_id: str):
        from ....domain.scientific_case import CaseUpdate

        rows = (
            await self._conn.execute(
                select(scientific_case_events)
                .join(scientific_cases, scientific_cases.c.id == scientific_case_events.c.case_id)
                .where(scientific_case_events.c.case_id == case_id)
                .where(scientific_cases.c.session_id == session_id)
                .order_by(scientific_case_events.c.revision)
            )
        ).mappings().all()
        return [
            CaseUpdate(
                op=row["op"], payload=dict(row["payload"].get("payload") or {}),
                actor=row["actor"], at=row["payload"].get("at", ""), run_id=row["run_id"],
                revision=row["revision"],
            )
            for row in rows
        ]

    async def case_id_for_run(self, run_id: str, *, session_id: str) -> str | None:
        """The case a run was attached to (its ``attach_run`` event)."""
        row = (
            await self._conn.execute(
                select(scientific_case_events.c.case_id)
                .join(scientific_cases, scientific_cases.c.id == scientific_case_events.c.case_id)
                .where(scientific_case_events.c.run_id == run_id)
                .where(scientific_case_events.c.op == "attach_run")
                .where(scientific_cases.c.session_id == session_id)
                .limit(1)
            )
        ).first()
        return row[0] if row is not None else None

    async def put_dossier(self, dossier: Mapping[str, Any], *, now: datetime) -> None:
        await self._conn.execute(
            insert(scientific_case_dossiers).values(
                run_id=dossier["run_id"], case_id=dossier["case_id"],
                case_revision=dossier["case_revision"], dossier=dict(dossier), created_at=now,
            )
        )

    async def get_dossier(self, run_id: str, *, session_id: str):
        row = (
            await self._conn.execute(
                select(scientific_case_dossiers.c.dossier)
                .join(scientific_cases, scientific_cases.c.id == scientific_case_dossiers.c.case_id)
                .where(scientific_case_dossiers.c.run_id == run_id)
                .where(scientific_cases.c.session_id == session_id)
            )
        ).first()
        return dict(row[0]) if row is not None else None

    async def latest_dossier(self, case_id: str, *, session_id: str):
        row = (
            await self._conn.execute(
                select(scientific_case_dossiers.c.dossier)
                .join(scientific_cases, scientific_cases.c.id == scientific_case_dossiers.c.case_id)
                .where(scientific_case_dossiers.c.case_id == case_id)
                .where(scientific_cases.c.session_id == session_id)
                .order_by(scientific_case_dossiers.c.created_at.desc(),
                          scientific_case_dossiers.c.case_revision.desc())
                .limit(1)
            )
        ).first()
        return dict(row[0]) if row is not None else None


class SqlDevelopmentPostureStore:
    """One row per answer that carried a development posture (ADS plan
    section 10.1/10.2, W6). Append-only, 1:1 with the answer it belongs to —
    ``answer_id`` is the primary key, not a separately minted id."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(
        self, posture, *, answer_id: str, session_id: str, run_id: str, now: datetime,
    ) -> None:
        await self._conn.execute(
            insert(development_postures).values(
                m.development_posture_to_row(
                    posture, answer_id=answer_id, session_id=session_id, run_id=run_id,
                    created_at=now,
                )
            )
        )

    async def get_for_answer(self, answer_id: str):
        row = (
            await self._conn.execute(
                select(development_postures).where(development_postures.c.answer_id == answer_id)
            )
        ).mappings().first()
        return m.row_to_development_posture(row) if row is not None else None

    async def list_for_run(self, run_id: str):
        rows = (
            await self._conn.execute(
                select(development_postures)
                .where(development_postures.c.run_id == run_id)
                .order_by(development_postures.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_development_posture(r) for r in rows]


class SqlSkillDraftStore:
    """Skill drafts awaiting review (W9-11). A save names the status it expects
    to replace, so two reviewers deciding at once cannot both win."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    @staticmethod
    def _draft(row):
        from ....domain.skill_draft import SkillDraft

        return SkillDraft.from_dict(row["draft"]) if row is not None else None

    async def add(self, draft, *, now: datetime) -> None:
        await self._conn.execute(insert(skill_drafts).values(
            id=draft.id, skill_id=draft.skill_id, status=draft.status,
            author_subject=draft.author.subject_id, draft=draft.to_dict(),
            created_at=now, updated_at=now,
        ))

    async def get(self, draft_id: str):
        row = (await self._conn.execute(
            select(skill_drafts).where(skill_drafts.c.id == draft_id)
        )).mappings().first()
        return self._draft(row)

    async def list(self, *, status: str | None = None, author_subject: str | None = None,
                   limit: int = 100):
        query = select(skill_drafts).order_by(skill_drafts.c.created_at.desc()).limit(limit)
        if status is not None:
            query = query.where(skill_drafts.c.status == status)
        if author_subject is not None:
            query = query.where(skill_drafts.c.author_subject == author_subject)
        rows = (await self._conn.execute(query)).mappings().all()
        return [self._draft(row) for row in rows]

    async def save(self, draft, *, expected_status: str, now: datetime) -> None:
        result = await self._conn.execute(
            update(skill_drafts)
            .where(skill_drafts.c.id == draft.id)
            .where(skill_drafts.c.status == expected_status)
            .values(status=draft.status, draft=draft.to_dict(), updated_at=now)
        )
        if result.rowcount != 1:
            raise Conflict("the skill draft changed; re-read it", draft_id=draft.id)
