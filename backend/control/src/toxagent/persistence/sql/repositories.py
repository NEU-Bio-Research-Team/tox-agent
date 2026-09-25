"""SQLAlchemy Core repositories.

Each takes the connection of the enclosing unit of work, so everything a
workflow touches — including the events it emits — lands in one transaction.
None of these classes opens a transaction of its own.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import and_, delete, func, insert, literal, or_, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ...domain.analysis import AnalysisSnapshot
from ...domain.answer import Claim, GroundedAnswer
from ...domain.attachment import Attachment
from ...domain.errors import Conflict
from ...domain.evidence import EvidenceRecord, EvidenceStatus
from ...domain.evidence_relation import EvidenceRelationAssessment
from ...domain.message import Message
from ...domain.observation import Observation
from ...domain.run import Run
from ...domain.runtime import RuntimeBinding
from ...domain.usage import RuntimeUsageEvent
from ...domain.session import Session
from ...connections.model import (
    ConnectionCapabilities, ConnectionStatus, ModelConnection,
)
from ...domain.runtime import AuthMode
from ...agent.kernel import KernelTransition
from ...domain.investigation import (
    CaseState, ConflictStatus, Coverage, EvidenceConflict, EvidenceGap, GapSeverity,
    GoalType, InvestigationPlan, InvestigationStep, StepStatus,
)
from ..schema import (
    analysis_snapshots,
    answers,
    attachments,
    capability_tokens,
    case_revisions,
    cases,
    claim_sources,
    claims,
    concurrency_slots,
    decision_support_states,
    development_postures,
    evidence_records,
    evidence_relation_assessments,
    explanation_checkpoints,
    report_artifacts,
    report_builds,
    report_claim_links,
    report_evidence_links,
    report_figures,
    report_renderings,
    investigation_plans,
    investigation_steps,
    kernel_transitions,
    model_connections,
    message_parts,
    messages,
    observations,
    runs,
    run_configuration_snapshots,
    run_jobs,
    scientific_case_dossiers,
    scientific_case_events,
    scientific_cases,
    session_settings,
    runtime_bindings,
    runtime_usage_events,
    sessions,
    tool_calls,
)


class SqlSessionSettingsStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get(self, session_id: str) -> dict[str, Any]:
        row = (await self._conn.execute(select(session_settings).where(
            session_settings.c.session_id == session_id
        ))).mappings().first()
        if row is None:
            return {"ai_profile_id": None, "predictor_bindings": {}}
        return {"ai_profile_id": row["ai_profile_id"], "predictor_bindings": dict(row["predictor_bindings"] or {})}

    async def put(self, session_id: str, *, ai_profile_id: str | None,
                  predictor_bindings: dict[str, str], now: datetime) -> None:
        values = {"session_id": session_id, "ai_profile_id": ai_profile_id,
                  "predictor_bindings": dict(predictor_bindings), "updated_at": now}
        existing = await self._conn.execute(select(session_settings.c.session_id).where(
            session_settings.c.session_id == session_id
        ))
        if existing.scalar() is None:
            await self._conn.execute(insert(session_settings).values(**values))
        else:
            await self._conn.execute(update(session_settings).where(
                session_settings.c.session_id == session_id
            ).values(**{k: v for k, v in values.items() if k != "session_id"}))


class SqlRunConfigurationSnapshotStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, run_id: str, *, ai_profile_id: str | None,
                  predictor_bindings: dict[str, str], now: datetime,
                  intent_decision: dict[str, Any] | None = None,
                  effective_budget: dict[str, Any] | None = None) -> None:
        await self._conn.execute(insert(run_configuration_snapshots).values(
            run_id=run_id, ai_profile_id=ai_profile_id,
            predictor_bindings=dict(predictor_bindings),
            intent_decision=dict(intent_decision) if intent_decision else None,
            effective_budget=dict(effective_budget) if effective_budget else None,
            created_at=now,
        ))

    async def get(self, run_id: str) -> dict[str, Any] | None:
        row = (await self._conn.execute(select(run_configuration_snapshots).where(
            run_configuration_snapshots.c.run_id == run_id
        ))).mappings().first()
        if row is None:
            return None
        return {
            "ai_profile_id": row["ai_profile_id"],
            "predictor_bindings": dict(row["predictor_bindings"] or {}),
            "intent_decision": dict(row["intent_decision"] or {}),
            "effective_budget": dict(row["effective_budget"]) if row.get("effective_budget") else None,
            "created_at": row["created_at"],
        }
from . import mapping as m


class SqlSessionStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, session: Session, *, client_session_id: str | None = None) -> None:
        await self._conn.execute(insert(sessions).values(m.session_to_row(session, client_session_id)))

    async def get(self, session_id: str, *, owner_id: str) -> Session | None:
        row = (
            await self._conn.execute(
                select(sessions).where(
                    and_(sessions.c.id == session_id, sessions.c.owner_id == owner_id)
                )
            )
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def get_unscoped(self, session_id: str) -> Session | None:
        row = (
            await self._conn.execute(select(sessions).where(sessions.c.id == session_id))
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def get_for_admission(
        self, session_id: str, *, owner_id: str, lock_timeout_ms: int
    ) -> Session | None:
        """Read an owned session while serializing admissions for it.

        Every message admission holds this parent row through its
        message/run/outbox commit. A second control-plane process can therefore
        re-check idempotency and the active-run cap only after the first has
        made its decision durable. ``NO KEY UPDATE`` is enough because an
        admission never changes the session primary key, and it stays
        compatible with foreign-key checks while child rows are inserted.
        """
        query = select(sessions).where(
            and_(sessions.c.id == session_id, sessions.c.owner_id == owner_id)
        )
        if self._conn.dialect.name == "postgresql":
            # ``set_config(..., true)`` is parameterised and transaction-local
            # (unlike interpolating a ``SET LOCAL`` command), so a pooled
            # connection cannot leak this admission timeout into another
            # request once the UoW commits or rolls back.
            await self._conn.execute(
                select(func.set_config("lock_timeout", f"{max(1, lock_timeout_ms)}ms", True))
            )
            query = query.with_for_update(key_share=True)
        row = (await self._conn.execute(query)).mappings().first()
        return m.row_to_session(row) if row else None

    async def find_by_client_id(self, owner_id: str, client_session_id: str) -> Session | None:
        row = (
            await self._conn.execute(
                select(sessions).where(
                    and_(
                        sessions.c.owner_id == owner_id,
                        sessions.c.client_session_id == client_session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def update(self, session: Session, *, expected_version: int) -> None:
        row = m.session_to_row(session)
        row.pop("client_session_id")
        row.pop("id")
        # event_sequence belongs to the unit of work's allocator, not to a
        # caller holding a stale copy of the aggregate.
        row.pop("event_sequence")
        result = await self._conn.execute(
            update(sessions)
            .where(and_(sessions.c.id == session.id, sessions.c.version == expected_version))
            .values(**row)
        )
        if result.rowcount == 0:
            raise Conflict(
                "session changed underneath this write",
                session_id=session.id,
                expected_version=expected_version,
            )

    async def list_for_owner(self, owner_id: str, *, limit: int, offset: int) -> Sequence[Session]:
        rows = (
            await self._conn.execute(
                select(sessions)
                .where(sessions.c.owner_id == owner_id)
                .order_by(sessions.c.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).mappings().all()
        return [m.row_to_session(r) for r in rows]


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


class SqlModelConnectionStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, connection: ModelConnection) -> None:
        await self._conn.execute(insert(model_connections).values(
            id=connection.id, owner_id=connection.owner_id, provider_id=connection.provider_id,
            model_id=connection.model_id, display_name=connection.display_name, base_url=connection.base_url,
            auth_mode=connection.auth_mode.value, credential_ref=connection.credential_ref,
            capabilities={
                "streaming": connection.capabilities.streaming,
                "tool_calls": connection.capabilities.tool_calls,
                "structured_output": connection.capabilities.structured_output,
                "context_size": connection.capabilities.context_size,
            }, status=connection.status.value, created_at=connection.created_at,
            updated_at=connection.updated_at,
        ))

    async def get(self, connection_id: str, *, owner_id: str) -> ModelConnection | None:
        row = (await self._conn.execute(select(model_connections).where(and_(
            model_connections.c.id == connection_id,
            model_connections.c.owner_id == owner_id,
        )))).mappings().first()
        if row is None:
            return None
        caps = row["capabilities"] or {}
        return ModelConnection(
            id=row["id"], owner_id=row["owner_id"], provider_id=row["provider_id"],
            model_id=row["model_id"], auth_mode=AuthMode(row["auth_mode"]),
            credential_ref=row["credential_ref"], base_url=row["base_url"],
            capabilities=ConnectionCapabilities(
                streaming=bool(caps.get("streaming")), tool_calls=bool(caps.get("tool_calls")),
                structured_output=bool(caps.get("structured_output")),
                context_size=caps.get("context_size"),
            ), status=ConnectionStatus(row["status"]), created_at=m.utc(row["created_at"]),
            updated_at=m.utc(row["updated_at"]), display_name=row.get("display_name") or "",
        )

    async def list(self, *, owner_id: str) -> Sequence[ModelConnection]:
        ids = (await self._conn.execute(select(model_connections.c.id).where(
            model_connections.c.owner_id == owner_id
        ).order_by(model_connections.c.created_at))).scalars().all()
        items = [await self.get(item, owner_id=owner_id) for item in ids]
        return [item for item in items if item is not None]

    async def update_probe(self, connection: ModelConnection) -> None:
        result = await self._conn.execute(update(model_connections).where(and_(
            model_connections.c.id == connection.id,
            model_connections.c.owner_id == connection.owner_id,
        )).values(
            capabilities={
                "streaming": connection.capabilities.streaming,
                "tool_calls": connection.capabilities.tool_calls,
                "structured_output": connection.capabilities.structured_output,
                "context_size": connection.capabilities.context_size,
            }, status=connection.status.value, updated_at=connection.updated_at,
        ))
        if result.rowcount == 0:
            raise Conflict("model connection does not exist", connection_id=connection.id)

    async def delete(self, connection_id: str, *, owner_id: str) -> bool:
        result = await self._conn.execute(delete(model_connections).where(and_(
            model_connections.c.id == connection_id,
            model_connections.c.owner_id == owner_id,
        )))
        return result.rowcount > 0


class SqlMessageStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, message: Message) -> None:
        await self._conn.execute(insert(messages).values(m.message_to_row(message)))
        if message.parts:
            await self._conn.execute(
                insert(message_parts), [m.part_to_row(p) for p in message.parts]
            )

    async def get(self, message_id: str) -> Message | None:
        row = (
            await self._conn.execute(select(messages).where(messages.c.id == message_id))
        ).mappings().first()
        if row is None:
            return None
        parts = (
            await self._conn.execute(
                select(message_parts).where(message_parts.c.message_id == message_id)
            )
        ).mappings().all()
        return m.row_to_message(row, list(parts))

    async def find_by_client_id(self, session_id: str, client_message_id: str) -> Message | None:
        row = (
            await self._conn.execute(
                select(messages).where(
                    and_(
                        messages.c.session_id == session_id,
                        messages.c.client_message_id == client_message_id,
                    )
                )
            )
        ).mappings().first()
        return await self.get(row["id"]) if row else None

    async def list_for_session(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> Sequence[Message]:
        rows = (
            await self._conn.execute(
                select(messages)
                .where(
                    and_(messages.c.session_id == session_id, messages.c.sequence > after_sequence)
                )
                .order_by(messages.c.sequence)
                .limit(limit)
            )
        ).mappings().all()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        part_rows = (
            await self._conn.execute(
                select(message_parts).where(message_parts.c.message_id.in_(ids))
            )
        ).mappings().all()
        by_message: dict[str, list] = {i: [] for i in ids}
        for p in part_rows:
            by_message[p["message_id"]].append(p)
        return [m.row_to_message(r, by_message[r["id"]]) for r in rows]

    async def latest_previews(
        self, session_ids: Sequence[str], *, max_length: int = 160
    ) -> dict[str, str]:
        """The newest message's text, per session, in two queries for the page.

        I20: the session list used to read the first 50 messages of each
        session ordered by ascending sequence and take the last of them, which
        is the newest message only while a session has 50 or fewer. Past that
        the preview froze on message 50 — and it cost one message query plus
        one parts query per session in the page.
        """
        if not session_ids:
            return {}
        newest = (
            select(messages.c.session_id, func.max(messages.c.sequence).label("sequence"))
            .where(messages.c.session_id.in_(session_ids))
            .group_by(messages.c.session_id)
            .subquery()
        )
        rows = (
            await self._conn.execute(
                select(messages.c.id, messages.c.session_id).join(
                    newest,
                    and_(
                        messages.c.session_id == newest.c.session_id,
                        messages.c.sequence == newest.c.sequence,
                    ),
                )
            )
        ).mappings().all()
        if not rows:
            return {}
        session_by_message = {row["id"]: row["session_id"] for row in rows}
        part_rows = (
            await self._conn.execute(
                select(message_parts)
                .where(and_(
                    message_parts.c.message_id.in_(list(session_by_message)),
                    message_parts.c.type == "text",
                ))
                .order_by(message_parts.c.index)
            )
        ).mappings().all()
        previews: dict[str, str] = {}
        for part in part_rows:
            session_id = session_by_message[part["message_id"]]
            if session_id in previews:
                continue  # first text part of that message, matching the old read
            text = str((part["content"] or {}).get("text", "")).strip()
            if text:
                previews[session_id] = text if len(text) <= max_length else f"{text[:max_length]}…"
        return previews

    async def next_sequence(self, session_id: str) -> int:
        current = (
            await self._conn.execute(
                select(func.max(messages.c.sequence)).where(messages.c.session_id == session_id)
            )
        ).scalar()
        return (current or 0) + 1

    async def append_part(
        self, message_id: str, index: int, part_type: str, content: dict[str, Any]
    ) -> str:
        from ...domain.ids import PART, new_id

        part_id = new_id(PART)
        await self._conn.execute(
            insert(message_parts).values(
                id=part_id, message_id=message_id, index=index,
                type=part_type, content=content, version=1,
            )
        )
        return part_id


class SqlRunStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, run: Run) -> None:
        await self._conn.execute(insert(runs).values(m.run_to_row(run)))

    async def get(self, run_id: str) -> Run | None:
        row = (await self._conn.execute(select(runs).where(runs.c.id == run_id))).mappings().first()
        return m.row_to_run(row) if row else None

    async def update(self, run: Run, *, expected_version: int) -> None:
        row = m.run_to_row(run)
        row.pop("id")
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run.id, runs.c.version == expected_version))
            .values(**row)
        )
        if result.rowcount == 0:
            raise Conflict("run changed underneath this write", run_id=run.id)

    async def list_for_session(self, session_id: str, *, limit: int = 50) -> Sequence[Run]:
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.session_id == session_id)
                .order_by(runs.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def count_by_session(self, session_ids: Sequence[str]) -> dict[str, int]:
        """How many runs each session actually has.

        I20: the session list reported `len(runs)` from a page capped at ten,
        so any session past its tenth run reported ten forever. A count is a
        count; if a cap were wanted the field would have to say so.
        """
        if not session_ids:
            return {}
        rows = (
            await self._conn.execute(
                select(runs.c.session_id, func.count().label("n"))
                .where(runs.c.session_id.in_(session_ids))
                .group_by(runs.c.session_id)
            )
        ).mappings().all()
        return {row["session_id"]: int(row["n"]) for row in rows}

    async def active_by_session(self, session_ids: Sequence[str]) -> dict[str, Run]:
        """The non-terminal run of each session, where there is one.

        At most one per session by the admission cap, so a newest-first read
        with no per-session limit is a single query over an indexed column.
        """
        if not session_ids:
            return {}
        rows = (
            await self._conn.execute(
                select(runs)
                .where(and_(
                    runs.c.session_id.in_(session_ids),
                    runs.c.status.in_(("queued", "running", "validating")),
                ))
                .order_by(runs.c.created_at.desc())
            )
        ).mappings().all()
        active: dict[str, Run] = {}
        for row in rows:
            active.setdefault(row["session_id"], m.row_to_run(row))
        return active

    async def list_non_terminal(self, *, limit: int = 1000) -> Sequence[Run]:
        """Every run still ``queued``/``running``/``validating``, across every
        session — used only by startup reconciliation, where "unowned by any
        in-process task" and "not terminal" are the same set (plan section
        6.5's cancellation policy assumes exactly one process owns a run)."""
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.status.in_(("queued", "running", "validating")))
                .order_by(runs.c.created_at)
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def request_cancel(self, run_id: str) -> bool:
        """Flag a cancellation request. Whether it is honoured, and how, is the
        gateway's business; this only records that it was asked for."""
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run_id, runs.c.status.in_(("queued", "running", "validating"))))
            .values(cancel_requested=True)
        )
        return result.rowcount > 0

    async def cancel_requested(self, run_id: str) -> bool:
        return bool(
            (
                await self._conn.execute(
                    select(runs.c.cancel_requested).where(runs.c.id == run_id)
                )
            ).scalar()
        )


class SqlRunJobStore:
    """The durable execution record for a run, and who owns executing it.

    Ownership is a lease, not a lock: a worker that stops renewing loses the
    run to whoever claims it next, which is the only way a `kill -9` can be
    told from a worker that is merely busy (I17/I18). Every state change is a
    single conditional UPDATE — the condition *is* the concurrency control, so
    two workers racing produce one winner and one rowcount of zero rather than
    two owners.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def enqueue(
        self,
        run_id: str,
        envelope: dict[str, Any],
        *,
        worker_id: str | None,
        lease_expires_at: datetime,
        now: datetime,
        queue_name: str | None = None,
        priority: int = 0,
    ) -> int:
        """Record the run's execution input, claimed by this worker — or by nobody.

        Written in the transaction that creates the run, so there is no window
        in which an accepted run exists with no way to execute it. Returns the
        fencing epoch the caller must present for every later write.

        ``worker_id=None`` is the external-worker path (WS08): the job is
        written unowned at epoch 0 with a lease that has already expired, which
        is exactly what makes it claimable by ``claim``. Nothing about a job
        waiting for its first worker differs from one whose worker died.
        """
        owned = worker_id is not None
        await self._conn.execute(
            insert(run_jobs).values(
                run_id=run_id, envelope=envelope, worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                lease_epoch=1 if owned else 0, attempts=1 if owned else 0,
                queue_name=queue_name, priority=priority,
                created_at=now, updated_at=now,
            )
        )
        return 1 if owned else 0

    async def renew(
        self, run_id: str, *, worker_id: str, epoch: int, lease_expires_at: datetime,
        now: datetime,
    ) -> bool:
        """Extend this worker's lease. False means it no longer holds it.

        A false here is the fencing signal: something else claimed the run
        because this worker looked dead. The caller must stop working on it —
        continuing would mean two workers executing one run.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(lease_expires_at=lease_expires_at, updated_at=now)
        )
        return result.rowcount > 0

    async def release(self, run_id: str, *, worker_id: str, epoch: int) -> bool:
        """Drop the job once its run is terminal.

        Conditional on still holding the lease so a fenced worker finishing
        late cannot delete the row the new owner is working under.
        """
        result = await self._conn.execute(
            delete(run_jobs).where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
        )
        return result.rowcount > 0

    async def discard(self, run_id: str) -> None:
        """Remove the job regardless of owner — for a run reconciled to a
        terminal state, where there is nothing left for any worker to own."""
        await self._conn.execute(delete(run_jobs).where(run_jobs.c.run_id == run_id))

    async def claimable(
        self,
        *,
        now: datetime,
        limit: int = 100,
        queue_names: Sequence[str] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Jobs whose lease has expired, by priority then oldest first.

        A live lease is deliberately not returned: another replica is running
        that job right now, and the whole point of I17 is that starting a new
        process must not disturb it. Neither is a job deferred until later.

        ``queue_names`` narrows to the queues a worker serves. A row with no
        queue (written before 0014) is returned regardless, and the caller
        derives its queue from the envelope.
        """
        conditions = [
            run_jobs.c.lease_expires_at <= now,
            or_(run_jobs.c.available_at.is_(None), run_jobs.c.available_at <= now),
        ]
        if queue_names is not None:
            conditions.append(
                or_(run_jobs.c.queue_name.in_(list(queue_names)), run_jobs.c.queue_name.is_(None))
            )
        rows = (
            await self._conn.execute(
                select(run_jobs)
                .where(and_(*conditions))
                .order_by(run_jobs.c.priority, run_jobs.c.created_at)
                .limit(limit)
            )
        ).mappings().all()
        return [dict(row) for row in rows]

    async def claim(
        self, run_id: str, *, worker_id: str, expected_epoch: int,
        lease_expires_at: datetime, now: datetime,
    ) -> int | None:
        """Take over an expired lease. Returns the new epoch, or None if lost.

        `expected_epoch` is the epoch this worker read when it decided the job
        was claimable. If another worker claimed it in between, the epoch has
        moved and this returns None — one winner, no coordination needed
        beyond the row itself.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.lease_epoch == expected_epoch,
                run_jobs.c.lease_expires_at <= now,
            ))
            .values(
                worker_id=worker_id, lease_expires_at=lease_expires_at,
                lease_epoch=expected_epoch + 1, attempts=run_jobs.c.attempts + 1,
                updated_at=now,
            )
        )
        return expected_epoch + 1 if result.rowcount > 0 else None

    async def get(self, run_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(select(run_jobs).where(run_jobs.c.run_id == run_id))
        ).mappings().first()
        return dict(row) if row else None

    async def held_by(self, worker_id: str) -> Sequence[str]:
        """Run ids this worker currently leases — the set it must watch for a
        cancellation requested through some other replica."""
        rows = (
            await self._conn.execute(
                select(run_jobs.c.run_id).where(run_jobs.c.worker_id == worker_id)
            )
        ).scalars().all()
        return list(rows)

    async def defer(
        self, run_id: str, *, worker_id: str, epoch: int, available_at: datetime,
        error_code: str, now: datetime,
    ) -> bool:
        """Give a claimed job back, not before ``available_at`` (PR-15).

        For a job that could not get its concurrency slots. It never started,
        so the claim is not counted as an attempt — a run that waited behind a
        cap ten times has not failed ten times.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(
                worker_id=None, lease_expires_at=available_at, available_at=available_at,
                last_error_code=error_code, attempts=run_jobs.c.attempts - 1, updated_at=now,
            )
        )
        return result.rowcount > 0

    async def hand_off(
        self, run_id: str, *, worker_id: str, epoch: int, now: datetime,
        error_code: str = "worker_draining",
    ) -> bool:
        """Release a job that was executing, for another worker to take now.

        Unlike ``defer`` this counts: the run did start, may have reached a
        provider, and the attempt bound is what stops a job that kills every
        worker it lands on from circulating forever.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(
                worker_id=None, lease_expires_at=now, last_error_code=error_code,
                updated_at=now,
            )
        )
        return result.rowcount > 0

    async def position(self, run_id: str, *, now: datetime) -> dict[str, Any] | None:
        """Where a waiting job stands in its queue. An estimate, and says so."""
        job = await self.get(run_id)
        if job is None:
            return None
        queue_name = job.get("queue_name")
        ahead = 0
        if job.get("worker_id") is None and queue_name:
            ahead = int(
                (
                    await self._conn.execute(
                        select(func.count())
                        .select_from(run_jobs)
                        .where(and_(
                            run_jobs.c.queue_name == queue_name,
                            run_jobs.c.worker_id.is_(None),
                            or_(
                                run_jobs.c.priority < job["priority"],
                                and_(
                                    run_jobs.c.priority == job["priority"],
                                    run_jobs.c.created_at < job["created_at"],
                                ),
                            ),
                        ))
                    )
                ).scalar()
                or 0
            )
        return {**job, "ahead": ahead}


class SqlConcurrencySlotStore:
    """Slot leases for the caps in ``application.concurrency`` (PR-15).

    Every write is conditional and says in its rowcount whether it won, so the
    caller never has to read the table to decide anything.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    def _insert(self):
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        return dialect_insert(concurrency_slots)

    async def take(
        self, *, scope: str, scope_key: str, limit: int, run_id: str, worker_id: str,
        expires_at: datetime, now: datetime,
    ) -> bool:
        slot = concurrency_slots.c
        # Already holding one in this scope — a job deferred and re-claimed by
        # the same run. Refresh it rather than take a second.
        refreshed = await self._conn.execute(
            update(concurrency_slots)
            .where(and_(
                slot.scope == scope, slot.scope_key == scope_key, slot.run_id == run_id,
                slot.expires_at > now,
            ))
            .values(worker_id=worker_id, expires_at=expires_at)
        )
        if refreshed.rowcount > 0:
            return True
        for index in range(limit):
            reclaimed = await self._conn.execute(
                update(concurrency_slots)
                .where(and_(
                    slot.scope == scope, slot.scope_key == scope_key,
                    slot.slot_index == index, slot.expires_at <= now,
                ))
                .values(
                    run_id=run_id, worker_id=worker_id, expires_at=expires_at,
                    acquired_at=now,
                )
            )
            if reclaimed.rowcount > 0:
                return True
            inserted = await self._conn.execute(
                self._insert()
                .values(
                    scope=scope, scope_key=scope_key, slot_index=index, run_id=run_id,
                    worker_id=worker_id, expires_at=expires_at, acquired_at=now,
                )
                .on_conflict_do_nothing()
            )
            if inserted.rowcount > 0:
                return True
        return False

    async def renew(self, *, run_id: str, worker_id: str, expires_at: datetime) -> int:
        result = await self._conn.execute(
            update(concurrency_slots)
            .where(and_(
                concurrency_slots.c.run_id == run_id,
                concurrency_slots.c.worker_id == worker_id,
            ))
            .values(expires_at=expires_at)
        )
        return result.rowcount

    async def release(self, *, run_id: str, worker_id: str) -> None:
        # Scoped to the worker: a worker that was fenced and finishes late must
        # not free the slots its successor now holds under the same run id.
        await self._conn.execute(
            delete(concurrency_slots).where(and_(
                concurrency_slots.c.run_id == run_id,
                concurrency_slots.c.worker_id == worker_id,
            ))
        )

    async def in_use(self, *, scope: str, scope_key: str, now: datetime) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(concurrency_slots)
                    .where(and_(
                        concurrency_slots.c.scope == scope,
                        concurrency_slots.c.scope_key == scope_key,
                        concurrency_slots.c.expires_at > now,
                    ))
                )
            ).scalar()
            or 0
        )


class SqlAnalysisStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, snapshot: AnalysisSnapshot) -> None:
        await self._conn.execute(insert(analysis_snapshots).values(m.analysis_to_row(snapshot)))

    async def get(self, analysis_id: str, *, session_id: str) -> AnalysisSnapshot | None:
        row = (
            await self._conn.execute(
                select(analysis_snapshots).where(
                    and_(
                        analysis_snapshots.c.id == analysis_id,
                        analysis_snapshots.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_analysis(row) if row else None

    async def find_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> AnalysisSnapshot | None:
        row = (
            await self._conn.execute(
                select(analysis_snapshots).where(
                    and_(
                        analysis_snapshots.c.session_id == session_id,
                        analysis_snapshots.c.idempotency_key == idempotency_key,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_analysis(row) if row else None

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> Sequence[AnalysisSnapshot]:
        rows = (
            await self._conn.execute(
                select(analysis_snapshots)
                .where(analysis_snapshots.c.session_id == session_id)
                .order_by(analysis_snapshots.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_analysis(r) for r in rows]


class SqlObservationStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, observation: Observation, *, analysis_id: str | None = None) -> None:
        await self._conn.execute(
            insert(observations).values(m.observation_to_row(observation, analysis_id))
        )

    async def get(self, observation_id: str, *, session_id: str) -> Observation | None:
        row = (
            await self._conn.execute(
                select(observations).where(
                    and_(
                        observations.c.id == observation_id,
                        observations.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_observation(row) if row else None

    async def list_for_run(self, run_id: str) -> Sequence[Observation]:
        rows = (
            await self._conn.execute(
                select(observations)
                .where(observations.c.run_id == run_id)
                .order_by(observations.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_observation(r) for r in rows]

    async def list_for_analysis(self, analysis_id: str) -> Sequence[Observation]:
        rows = (
            await self._conn.execute(
                select(observations)
                .where(observations.c.analysis_id == analysis_id)
                .order_by(observations.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_observation(r) for r in rows]


class SqlExplanationCheckpointStore:
    """Explanations that have already been computed and committed.

    I19: a bundle ran predict and every requested explanation before writing
    anything, so a crash after the fifth of eight targets discarded all five,
    and the retry paid for them again. Each target is committed as it lands
    here instead, addressed by what it is an explanation *of* — the canonical
    molecule, the endpoint, the task and the model that produced the
    probability. A retry looks the completed ones up and does only what is
    left.

    Written once and read back; there is no update path, because an
    explanation of a fixed input by a fixed model does not change.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get_many(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not keys:
            return {}
        rows = (
            await self._conn.execute(
                select(explanation_checkpoints.c.key, explanation_checkpoints.c.payload)
                .where(explanation_checkpoints.c.key.in_(list(keys)))
            )
        ).mappings().all()
        return {row["key"]: row["payload"] for row in rows}

    async def put(
        self,
        key: str,
        *,
        session_id: str,
        endpoint: str,
        task: str | None,
        model_id: str | None,
        canonical_smiles: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        """Record one completed explanation.

        A duplicate key means another attempt committed the same artifact
        first; that is the checkpoint working, not a conflict, so the insert
        is skipped rather than raised.
        """
        existing = (
            await self._conn.execute(
                select(explanation_checkpoints.c.key)
                .where(explanation_checkpoints.c.key == key)
            )
        ).scalar()
        if existing is not None:
            return
        await self._conn.execute(
            insert(explanation_checkpoints).values(
                key=key, session_id=session_id, endpoint=endpoint, task=task,
                model_id=model_id, canonical_smiles=canonical_smiles, payload=payload,
                created_at=now,
            )
        )


class SqlEvidenceStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, record: EvidenceRecord) -> None:
        await self._conn.execute(insert(evidence_records).values(m.evidence_to_row(record)))

    async def get(self, evidence_id: str, *, session_id: str) -> EvidenceRecord | None:
        row = (
            await self._conn.execute(
                select(evidence_records).where(
                    and_(
                        evidence_records.c.id == evidence_id,
                        evidence_records.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_evidence(row) if row else None

    async def find_by_dedupe_key(self, session_id: str, dedupe_key: str) -> EvidenceRecord | None:
        row = (
            await self._conn.execute(
                select(evidence_records).where(
                    and_(
                        evidence_records.c.session_id == session_id,
                        evidence_records.c.dedupe_key == dedupe_key,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_evidence(row) if row else None

    async def set_status(
        self, evidence_id: str, status: EvidenceStatus, *, reason: str | None = None
    ) -> None:
        await self._conn.execute(
            update(evidence_records)
            .where(evidence_records.c.id == evidence_id)
            .values(status=status.value, rejection_reason=reason)
        )

    async def list_for_session(
        self,
        session_id: str,
        *,
        status: EvidenceStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[EvidenceRecord]:
        query = select(evidence_records).where(evidence_records.c.session_id == session_id)
        if status is not None:
            query = query.where(evidence_records.c.status == status.value)
        rows = (
            await self._conn.execute(
                query.order_by(evidence_records.c.retrieved_at.desc()).limit(limit).offset(offset)
            )
        ).mappings().all()
        return [m.row_to_evidence(r) for r in rows]


class SqlEvidenceRelationStore:
    """One source-vs-proposition assessment per row (ADS plan section 9.3,
    ADR 0010). Append-only: a relation is never edited in place, a later
    reassessment is a new row for the same proposition."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, assessment: EvidenceRelationAssessment) -> None:
        await self._conn.execute(
            insert(evidence_relation_assessments).values(
                m.evidence_relation_to_row(assessment)
            )
        )

    async def list_for_run(self, run_id: str) -> Sequence[EvidenceRelationAssessment]:
        rows = (
            await self._conn.execute(
                select(evidence_relation_assessments)
                .where(evidence_relation_assessments.c.run_id == run_id)
                .order_by(evidence_relation_assessments.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_evidence_relation(r) for r in rows]

    async def list_for_proposition(
        self, session_id: str, proposition_id: str
    ) -> Sequence[EvidenceRelationAssessment]:
        rows = (
            await self._conn.execute(
                select(evidence_relation_assessments).where(
                    and_(
                        evidence_relation_assessments.c.session_id == session_id,
                        evidence_relation_assessments.c.proposition_id == proposition_id,
                    )
                ).order_by(evidence_relation_assessments.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_evidence_relation(r) for r in rows]


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
        from ...domain.decision_state import DecisionSupportStateV1

        row = (
            await self._conn.execute(
                select(decision_support_states).where(decision_support_states.c.run_id == run_id)
            )
        ).mappings().first()
        return DecisionSupportStateV1.from_dict(row["state"]) if row is not None else None

    async def latest_for_session(self, session_id: str, *, exclude_run_id: str | None = None):
        from ...domain.decision_state import DecisionSupportStateV1

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
        from ...domain.decision_state import DecisionSupportStateV1

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
        from ...domain.scientific_case import ScientificCaseV1

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
        from ...domain.scientific_case import CaseUpdate

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


class SqlAnswerStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, answer: GroundedAnswer) -> None:
        await self._conn.execute(insert(answers).values(m.answer_to_row(answer)))
        if not answer.claims:
            return
        await self._conn.execute(
            insert(claims),
            [m.claim_to_row(c, answer.id, i) for i, c in enumerate(answer.claims)],
        )
        citations = [
            {"claim_id": c.claim_id, "evidence_id": e}
            for c in answer.claims
            for e in c.citation_ids
        ]
        if citations:
            await self._conn.execute(insert(claim_sources), citations)

    async def _load(self, row) -> GroundedAnswer:
        claim_rows = (
            await self._conn.execute(
                select(claims).where(claims.c.answer_id == row["id"]).order_by(claims.c.position)
            )
        ).mappings().all()
        citation_rows = (
            await self._conn.execute(
                select(claim_sources).where(
                    claim_sources.c.claim_id.in_([c["id"] for c in claim_rows] or [""])
                )
            )
        ).mappings().all()
        by_claim: dict[str, list[str]] = {}
        for c in citation_rows:
            by_claim.setdefault(c["claim_id"], []).append(c["evidence_id"])
        return m.row_to_answer(
            row,
            tuple(m.row_to_claim(c, tuple(sorted(by_claim.get(c["id"], [])))) for c in claim_rows),
        )

    async def get(self, answer_id: str, *, session_id: str) -> GroundedAnswer | None:
        row = (
            await self._conn.execute(
                select(answers).where(
                    and_(answers.c.id == answer_id, answers.c.session_id == session_id)
                )
            )
        ).mappings().first()
        return await self._load(row) if row else None

    async def get_for_run(self, run_id: str) -> GroundedAnswer | None:
        row = (
            await self._conn.execute(
                select(answers)
                .where(answers.c.run_id == run_id)
                .order_by(answers.c.candidate_generation.desc())
                .limit(1)
            )
        ).mappings().first()
        return await self._load(row) if row else None

    async def candidate_generations(self, run_id: str) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count()).select_from(answers).where(answers.c.run_id == run_id)
                )
            ).scalar()
            or 0
        )

    async def claims_for(self, answer_id: str) -> Sequence[Claim]:
        rows = (
            await self._conn.execute(
                select(claims).where(claims.c.answer_id == answer_id).order_by(claims.c.position)
            )
        ).mappings().all()
        return [m.row_to_claim(r) for r in rows]

    async def claim_id_exists(self, claim_id: str) -> bool:
        row = (
            await self._conn.execute(
                select(literal(1)).select_from(claims).where(claims.c.id == claim_id).limit(1)
            )
        ).first()
        return row is not None


class SqlRuntimeBindingStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, binding: RuntimeBinding) -> None:
        await self._conn.execute(insert(runtime_bindings).values(m.binding_to_row(binding)))

    async def get(self, binding_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings).where(runtime_bindings.c.id == binding_id)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def active_for_session(self, session_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings)
                .where(
                    and_(
                        runtime_bindings.c.session_id == session_id,
                        runtime_bindings.c.status == "active",
                    )
                )
                .order_by(runtime_bindings.c.created_at.desc())
                .limit(1)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def set_status(self, binding_id: str, status: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(runtime_bindings)
            .where(runtime_bindings.c.id == binding_id)
            .values(status=status, closed_at=now)
        )


class SqlRuntimeUsageStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, event: RuntimeUsageEvent) -> None:
        await self._conn.execute(insert(runtime_usage_events).values(m.usage_to_row(event)))

    async def list_for_run(self, run_id: str) -> Sequence[RuntimeUsageEvent]:
        rows = (
            await self._conn.execute(
                select(runtime_usage_events)
                .where(runtime_usage_events.c.run_id == run_id)
                .order_by(runtime_usage_events.c.reported_at, runtime_usage_events.c.id)
            )
        ).mappings().all()
        return [m.row_to_usage(row) for row in rows]


class SqlToolCallStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def try_reserve(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, now: datetime, max_calls: int | None, max_identical: int,
    ) -> bool:
        """Atomically check the per-run budget and duplicate-call cap and, if
        both allow it, reserve the call by inserting its ``running`` row — all
        as one ``INSERT ... SELECT ... WHERE`` statement, so the count and the
        reservation cannot be observed and acted on separately by two
        concurrent calls (the race a plain "SELECT count, then INSERT" allows:
        several callers can each see room under the budget before any of them
        has actually taken a slot). ``status='denied'`` rows are excluded from
        both counts — a denied attempt is kept for audit but must not itself
        shrink the budget for the next real attempt. ``max_calls=None`` (used
        for the final-answer tool) skips the budget check entirely; the
        duplicate-call cap still applies.

        Returns whether the reservation succeeded.
        """
        if self._conn.dialect.name == "postgresql":
            # One statement is not one serialization point. Under READ
            # COMMITTED each statement takes its own snapshot and an
            # INSERT ... SELECT takes no lock that would stop a concurrent
            # transaction inserting the row its own count did not see, so five
            # concurrent calls against a budget of two were all admitted. The
            # docstring above was true of SQLite only, where a database-level
            # write lock serialized them for reasons that have nothing to do
            # with this query.
            #
            # Serializing on the parent run makes the claim true on both:
            # reservations for one run queue behind each other, and the count
            # below then runs in a statement whose snapshot includes whatever
            # the previous holder committed. NO KEY UPDATE for the same reason
            # `get_for_admission` uses it — a reservation never changes the
            # run's key, and child rows still pass their foreign-key check.
            await self._conn.execute(
                select(runs.c.id).where(runs.c.id == run_id).with_for_update(key_share=True)
            )
        not_denied = tool_calls.c.status != "denied"
        conditions = [
            (
                select(func.count())
                .select_from(tool_calls)
                .where(and_(tool_calls.c.run_id == run_id, not_denied))
                .scalar_subquery()
                < max_calls
            )
        ] if max_calls is not None else []
        conditions.append(
            select(func.count())
            .select_from(tool_calls)
            .where(
                and_(
                    tool_calls.c.run_id == run_id,
                    tool_calls.c.tool_name == tool_name,
                    tool_calls.c.arguments_sha256 == arguments_sha256,
                    not_denied,
                )
            )
            .scalar_subquery()
            < max_identical
        )

        source = select(
            literal(call_id), literal(session_id), literal(run_id), literal(tool_name),
            literal(arguments_sha256), literal("running"),
            literal([], type_=tool_calls.c.observation_ids.type), literal(now),
        ).where(and_(*conditions))
        stmt = insert(tool_calls).from_select(
            ["id", "session_id", "run_id", "tool_name", "arguments_sha256", "status",
             "observation_ids", "started_at"],
            source,
        )
        result = await self._conn.execute(stmt)
        return result.rowcount == 1

    async def record_denied(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, error_code: str, now: datetime,
    ) -> None:
        """A denied call still leaves an audit trail, distinct from ``running``/
        ``completed``/``error`` (which describe a call that was actually
        admitted) so ``count_for_run``-style budget accounting can exclude it
        while `get_run`'s tool-call listing still shows every attempt."""
        await self._conn.execute(
            insert(tool_calls).values(
                id=call_id, session_id=session_id, run_id=run_id, tool_name=tool_name,
                arguments_sha256=arguments_sha256, status="denied", error_code=error_code,
                observation_ids=[], started_at=now, ended_at=now, duration_ms=0,
            )
        )

    async def finish(
        self, call_id: str, *, status: str, error_code: str | None,
        observation_ids: list[str], duration_ms: int, now: datetime,
    ) -> None:
        await self._conn.execute(
            update(tool_calls)
            .where(tool_calls.c.id == call_id)
            .values(
                status=status, error_code=error_code, observation_ids=observation_ids,
                duration_ms=duration_ms, ended_at=now,
            )
        )

    async def count_for_run(self, run_id: str) -> int:
        """Admitted calls only — same accounting ``try_reserve`` enforces, so
        a denied attempt never itself counts against the budget it was
        denied under."""
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(and_(tool_calls.c.run_id == run_id, tool_calls.c.status != "denied"))
                )
            ).scalar()
            or 0
        )

    async def count_for_run_and_tool(self, run_id: str, tool_name: str) -> int:
        """Admitted calls to one tool, for a per-tool budget (ADS plan
        section 9.2/W4-04) distinct from ``count_for_run``'s whole-run cap —
        e.g. a decision_support run may search several times within its
        overall step budget, but only up to its own, tighter query budget."""
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(
                        and_(
                            tool_calls.c.run_id == run_id,
                            tool_calls.c.tool_name == tool_name,
                            tool_calls.c.status != "denied",
                        )
                    )
                )
            ).scalar()
            or 0
        )

    async def duplicate_count(self, run_id: str, tool_name: str, arguments_sha256: str) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(
                        and_(
                            tool_calls.c.run_id == run_id,
                            tool_calls.c.tool_name == tool_name,
                            tool_calls.c.arguments_sha256 == arguments_sha256,
                            tool_calls.c.status != "denied",
                        )
                    )
                )
            ).scalar()
            or 0
        )

    async def list_for_run(self, run_id: str) -> Sequence[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(tool_calls)
                .where(tool_calls.c.run_id == run_id)
                .order_by(tool_calls.c.started_at)
            )
        ).mappings().all()
        # Unlike every other repository, these rows go straight out as raw
        # dicts rather than through a row_to_* mapper — so they were the one
        # place SQLite's naive (no-tzinfo) datetimes reached a client
        # unnormalized, rendering as local time in whatever timezone the
        # browser happened to be in instead of UTC.
        results = [dict(r) for r in rows]
        for row in results:
            row["started_at"] = m.utc(row["started_at"])
            row["ended_at"] = m.utc(row["ended_at"])
        return results


class SqlCapabilityTokenStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def issue(
        self, *, jti: str, session_id: str, run_id: str, runtime_binding_id: str | None,
        allowed_tools: list[str], issued_at: datetime, expires_at: datetime,
    ) -> None:
        await self._conn.execute(
            insert(capability_tokens).values(
                jti=jti, session_id=session_id, run_id=run_id,
                runtime_binding_id=runtime_binding_id, allowed_tools=allowed_tools,
                issued_at=issued_at, expires_at=expires_at,
            )
        )

    async def is_valid(self, jti: str, *, now: datetime) -> bool:
        row = (
            await self._conn.execute(
                select(capability_tokens).where(capability_tokens.c.jti == jti)
            )
        ).mappings().first()
        if row is None or row["revoked_at"] is not None:
            return False
        return m.utc(row["expires_at"]) > now

    async def revoke(self, jti: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(capability_tokens)
            .where(capability_tokens.c.jti == jti)
            .values(revoked_at=now)
        )


class SqlAttachmentStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, attachment: Attachment) -> None:
        await self._conn.execute(insert(attachments).values(m.attachment_to_row(attachment)))

    async def get(self, attachment_id: str, *, owner_id: str) -> Attachment | None:
        row = (
            await self._conn.execute(
                select(attachments).where(
                    and_(attachments.c.id == attachment_id, attachments.c.owner_id == owner_id)
                )
            )
        ).mappings().first()
        return m.row_to_attachment(row) if row else None


class SqlReportStore:
    """Report builds, artifacts, figures, renderings and their link rows.

    One store rather than six, because everything here is written inside the
    same transaction as the thing it describes: an artifact and its claim links
    committed separately could leave a report whose citations nobody can audit.

    Artifacts are stored as the canonical document they already are, and read
    back verbatim. Rebuilding a ``ReportArtifact`` dataclass graph from JSON on
    every read would be a second implementation of what the report says, and
    the two would eventually disagree; the content hash is verified on read
    instead, which is a stronger guarantee than a reconstruction that "looks
    right" (spec section 12.1, workstream R2's exit gate).
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    # --- builds ------------------------------------------------------------

    async def add_build(self, build) -> None:
        await self._conn.execute(insert(report_builds).values(m.report_build_to_row(build)))

    async def get_build(self, build_id: str, *, session_id: str):
        row = (
            await self._conn.execute(
                select(report_builds).where(
                    and_(
                        report_builds.c.id == build_id,
                        report_builds.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_report_build(row) if row else None

    async def save_build(self, build) -> None:
        """Upsert the mutable build state. The build aggregate is the one thing
        in the report path that legitimately changes: it is a state machine."""
        row = m.report_build_to_row(build)
        result = await self._conn.execute(
            update(report_builds)
            .where(report_builds.c.id == build.id)
            .values({k: v for k, v in row.items() if k != "id"})
        )
        if result.rowcount == 0:
            await self._conn.execute(insert(report_builds).values(row))

    async def list_builds_for_session(self, session_id: str, *, limit: int = 50):
        rows = (
            await self._conn.execute(
                select(report_builds)
                .where(report_builds.c.session_id == session_id)
                .order_by(report_builds.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_report_build(row) for row in rows]

    # --- artifacts ---------------------------------------------------------

    async def add_artifact(
        self,
        document: dict[str, Any],
        *,
        claim_links: Sequence[Mapping[str, Any]] = (),
        evidence_links: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        await self._conn.execute(
            insert(report_artifacts).values(
                id=document["report_id"],
                report_build_id=document["report_build_id"],
                session_id=document["session_id"],
                analysis_id=document["analysis_id"],
                schema_version=document["schema_version"],
                title=document["title"],
                status=document["status"],
                report_language=document.get("report_language", "en"),
                document=document,
                content_sha256=document["content_sha256"],
                supersedes_report_id=document.get("supersedes_report_id"),
                version=document.get("version", 1),
                created_at=_parse_ts(document["created_at"]),
            )
        )
        # Deduplicated here rather than trusted from the caller: the link rows
        # are a primary key pair, and one claim cited in two sections would
        # otherwise abort the whole artifact write.
        seen: set[tuple[str, str]] = set()
        rows = []
        for link in claim_links:
            key = (document["report_id"], link["claim_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"report_id": document["report_id"], **dict(link)})
        if rows:
            await self._conn.execute(insert(report_claim_links).values(rows))

        seen.clear()
        rows = []
        for link in evidence_links:
            key = (document["report_id"], link["evidence_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"report_id": document["report_id"], **dict(link)})
        if rows:
            await self._conn.execute(insert(report_evidence_links).values(rows))

    async def get_artifact(self, report_id: str, *, session_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_artifacts).where(
                    and_(
                        report_artifacts.c.id == report_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        if row is None:
            return None
        document = dict(row["document"])
        stored = row["content_sha256"]
        if document.get("content_sha256") != stored:
            # The row and the document it holds disagree about what this report
            # is. Refusing is the only safe answer: a report served from a
            # document whose hash nobody can vouch for is not auditable, which
            # is the whole reason it is stored hashed.
            raise ValueError(
                f"report {report_id} has a document hash that does not match its row; "
                "refusing to serve it"
            )
        document["renderings"] = await self.list_renderings(report_id)
        return document

    async def list_artifacts_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Metadata only: a listing that returned every full document would
        grow with the size of the reports rather than their number."""
        rows = (
            await self._conn.execute(
                select(
                    report_artifacts.c.id, report_artifacts.c.report_build_id,
                    report_artifacts.c.analysis_id, report_artifacts.c.title,
                    report_artifacts.c.status, report_artifacts.c.report_language,
                    report_artifacts.c.content_sha256, report_artifacts.c.version,
                    report_artifacts.c.supersedes_report_id, report_artifacts.c.created_at,
                )
                .where(report_artifacts.c.session_id == session_id)
                .order_by(report_artifacts.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [
            {
                "report_id": row["id"],
                "report_build_id": row["report_build_id"],
                "analysis_id": row["analysis_id"],
                "title": row["title"],
                "status": row["status"],
                "report_language": row["report_language"],
                "content_sha256": row["content_sha256"],
                "version": row["version"],
                "supersedes_report_id": row["supersedes_report_id"],
                "created_at": m.utc(row["created_at"]).isoformat(),
            }
            for row in rows
        ]

    async def get_latest_artifact_for_analysis(
        self, analysis_id: str, *, session_id: str
    ) -> dict[str, Any] | None:
        """The most recent version's full document, or ``None`` if this
        analysis has no report yet. "Latest" is ``version`` DESC, not
        ``created_at`` DESC (ADS plan section 7.3/W3-03): a rebuild's
        ``supersedes_report_id`` chain is what makes a version newer, and
        ``version`` is the field that encodes that ordering directly."""
        report_id = (
            await self._conn.execute(
                select(report_artifacts.c.id)
                .where(
                    and_(
                        report_artifacts.c.analysis_id == analysis_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
                .order_by(report_artifacts.c.version.desc())
                .limit(1)
            )
        ).scalar()
        if report_id is None:
            return None
        return await self.get_artifact(report_id, session_id=session_id)

    async def latest_version_for_analysis(self, analysis_id: str, *, session_id: str) -> int:
        row = (
            await self._conn.execute(
                select(func.max(report_artifacts.c.version)).where(
                    and_(
                        report_artifacts.c.analysis_id == analysis_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).scalar()
        return int(row or 0)

    # --- figures -----------------------------------------------------------

    async def add_figure(self, figure, *, session_id: str, created_at: datetime) -> None:
        await self._conn.execute(
            insert(report_figures).values(
                figure_id=figure.figure_id,
                session_id=session_id,
                attachment_id=figure.attachment_id,
                observation_id=figure.observation_id,
                endpoint=figure.endpoint,
                task=figure.task,
                media_type=figure.media_type,
                caption=figure.caption,
                alt_text=figure.alt_text,
                content_sha256=figure.content_sha256,
                renderer_version=figure.renderer_version,
                created_at=created_at,
            )
        )

    async def get_figure(self, figure_id: str, *, session_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_figures).where(
                    and_(
                        report_figures.c.figure_id == figure_id,
                        report_figures.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return dict(row) if row else None

    # --- renderings --------------------------------------------------------

    async def add_rendering(self, report_id: str, rendering) -> None:
        await self._conn.execute(
            insert(report_renderings).values(
                id=rendering.rendering_id,
                report_id=report_id,
                format=rendering.format,
                media_type=rendering.media_type,
                object_uri=rendering.object_uri,
                content_sha256=rendering.content_sha256,
                size_bytes=rendering.size_bytes,
                renderer_version=rendering.renderer_version,
                created_at=rendering.created_at,
            )
        )

    async def list_renderings(self, report_id: str) -> list[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(report_renderings)
                .where(report_renderings.c.report_id == report_id)
                .order_by(report_renderings.c.format)
            )
        ).mappings().all()
        return [
            {
                "rendering_id": row["id"],
                "format": row["format"],
                "media_type": row["media_type"],
                "object_uri": row["object_uri"],
                "content_sha256": row["content_sha256"],
                "size_bytes": row["size_bytes"],
                "renderer_version": row["renderer_version"],
                "created_at": m.utc(row["created_at"]).isoformat(),
            }
            for row in rows
        ]

    async def get_rendering(
        self, report_id: str, fmt: str, *, session_id: str
    ) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_renderings)
                .select_from(
                    report_renderings.join(
                        report_artifacts,
                        report_renderings.c.report_id == report_artifacts.c.id,
                    )
                )
                .where(
                    and_(
                        report_renderings.c.report_id == report_id,
                        report_renderings.c.format == fmt,
                        # Scoped through the artifact, so a rendering id from
                        # another session resolves to nothing rather than to
                        # someone else's file.
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        if row is None:
            return None
        return {
            "rendering_id": row["id"],
            "report_id": row["report_id"],
            "format": row["format"],
            "media_type": row["media_type"],
            "object_uri": row["object_uri"],
            "content_sha256": row["content_sha256"],
            "size_bytes": row["size_bytes"],
            "renderer_version": row["renderer_version"],
        }


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
