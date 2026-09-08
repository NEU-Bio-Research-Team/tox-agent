"""SQLAlchemy Core repositories.

Each takes the connection of the enclosing unit of work, so everything a
workflow touches — including the events it emits — lands in one transaction.
None of these classes opens a transaction of its own.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import and_, delete, func, insert, literal, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ...domain.analysis import AnalysisSnapshot
from ...domain.answer import Claim, GroundedAnswer
from ...domain.attachment import Attachment
from ...domain.errors import Conflict
from ...domain.evidence import EvidenceRecord, EvidenceStatus
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
    evidence_records,
    explanation_checkpoints,
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
                  predictor_bindings: dict[str, str], now: datetime) -> None:
        await self._conn.execute(insert(run_configuration_snapshots).values(
            run_id=run_id, ai_profile_id=ai_profile_id,
            predictor_bindings=dict(predictor_bindings), created_at=now,
        ))

    async def get(self, run_id: str) -> dict[str, Any] | None:
        row = (await self._conn.execute(select(run_configuration_snapshots).where(
            run_configuration_snapshots.c.run_id == run_id
        ))).mappings().first()
        if row is None:
            return None
        return {"ai_profile_id": row["ai_profile_id"], "predictor_bindings": dict(row["predictor_bindings"] or {}), "created_at": row["created_at"]}
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
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime,
    ) -> int:
        """Record the run's execution input, claimed by this worker.

        Written in the transaction that creates the run, so there is no window
        in which an accepted run exists with no way to execute it. Returns the
        fencing epoch the caller must present for every later write.
        """
        await self._conn.execute(
            insert(run_jobs).values(
                run_id=run_id, envelope=envelope, worker_id=worker_id,
                lease_expires_at=lease_expires_at, lease_epoch=1, attempts=1,
                created_at=now, updated_at=now,
            )
        )
        return 1

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

    async def claimable(self, *, now: datetime, limit: int = 100) -> Sequence[dict[str, Any]]:
        """Jobs whose lease has expired, oldest first.

        A live lease is deliberately not returned: another replica is running
        that job right now, and the whole point of I17 is that starting a new
        process must not disturb it.
        """
        rows = (
            await self._conn.execute(
                select(run_jobs)
                .where(run_jobs.c.lease_expires_at <= now)
                .order_by(run_jobs.c.created_at)
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
