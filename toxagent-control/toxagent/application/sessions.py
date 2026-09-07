"""Session lifecycle and the read projections behind the REST endpoints.

The read side exists so that a client which lost its stream can rebuild
everything it had (PROD-05). Every projection here is derived from stored state
alone — never from something a runtime still holds in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..domain.errors import SessionNotFound
from ..domain.events import EventType
from ..domain.session import Language, Session
from .policy import Actor
from .projections import display_projection


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionService:
    database: Any

    async def create(
        self,
        actor: Actor,
        *,
        preferred_language: str = "en",
        title: str | None = None,
        client_session_id: str | None = None,
    ) -> Session:
        async with self.database.unit_of_work() as uow:
            if client_session_id:
                existing = await uow.sessions.find_by_client_id(
                    actor.subject_id, client_session_id
                )
                if existing is not None:
                    return existing
            session = Session.create(
                actor.subject_id, now=_now(),
                preferred_language=Language(preferred_language), title=title,
            )
            await uow.sessions.add(session, client_session_id=client_session_id)
            uow.emit(
                session_id=session.id, type=EventType.SESSION_CREATED,
                entity_type="session", entity_id=session.id,
                payload={"preferred_language": session.preferred_language.value},
            )
            await uow.commit()
        return session

    async def get(self, actor: Actor, session_id: str) -> Session:
        async with self.database.unit_of_work() as uow:
            session = await uow.sessions.get(session_id, owner_id=actor.subject_id)
        if session is None:
            raise SessionNotFound("no such session", session_id=session_id)
        return session

    async def projection(self, actor: Actor, session_id: str) -> dict[str, Any]:
        async with self.database.unit_of_work() as uow:
            session = await uow.sessions.get(session_id, owner_id=actor.subject_id)
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)
            runs = await uow.runs.list_for_session(session_id, limit=10)
            active_analysis = None
            if session.active_analysis_id:
                snapshot = await uow.analyses.get(
                    session.active_analysis_id, session_id=session_id
                )
                if snapshot is not None:
                    active_analysis = display_projection(snapshot)
            latest_sequence = session.event_sequence

        active_run = next((r for r in runs if not r.is_terminal), None)
        return {
            "session_id": session.id,
            "status": session.status.value,
            "preferred_language": session.preferred_language.value,
            "title": session.title,
            "version": session.version,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "latest_event_sequence": latest_sequence,
            "active_run": run_projection(active_run) if active_run else None,
            "recent_runs": [run_projection(r) for r in runs],
            "active_analysis": active_analysis,
        }

    async def list(
        self, actor: Actor, *, limit: int = 25, offset: int = 0
    ) -> dict[str, Any]:
        """A row per session, enough to render a session list without opening
        each one. The cursor is deliberately just the offset it produced: an
        opaque string here would buy nothing a client couldn't work out from
        the row count it already asked for."""
        async with self.database.unit_of_work() as uow:
            sessions_page = await uow.sessions.list_for_owner(
                actor.subject_id, limit=limit + 1, offset=offset
            )
            has_more = len(sessions_page) > limit
            sessions_page = sessions_page[:limit]

            rows: list[dict[str, Any]] = []
            for session in sessions_page:
                runs = await uow.runs.list_for_session(session.id, limit=10)
                active_run = next((r for r in runs if not r.is_terminal), None)
                messages = await uow.messages.list_for_session(session.id, limit=50)
                last_message_preview = None
                if messages:
                    last = messages[-1]
                    text_part = next(
                        (p for p in last.parts if p.type.value == "text"), None
                    )
                    if text_part is not None:
                        text = str(text_part.content.get("text", "")).strip()
                        last_message_preview = (
                            text if len(text) <= 160 else f"{text[:160]}…"
                        )
                rows.append(
                    {
                        "session_id": session.id,
                        "title": session.title,
                        "status": session.status.value,
                        "preferred_language": session.preferred_language.value,
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat(),
                        "active_run": (
                            {"run_id": active_run.id, "status": active_run.status.value,
                             "intent": active_run.intent.value}
                            if active_run else None
                        ),
                        "run_count": len(runs),
                        "last_message_preview": last_message_preview,
                    }
                )
        return {
            "sessions": rows,
            "next_offset": offset + len(rows) if has_more else None,
        }

    async def messages(
        self, actor: Actor, session_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        async with self.database.unit_of_work() as uow:
            session = await uow.sessions.get(session_id, owner_id=actor.subject_id)
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)
            messages = await uow.messages.list_for_session(
                session_id, after_sequence=after_sequence, limit=limit
            )
        return [
            {
                "message_id": m.id,
                "role": m.role.value,
                "sequence": m.sequence,
                "created_at": m.created_at.isoformat(),
                "client_message_id": m.client_message_id,
                "parts": [
                    {"part_id": p.id, "index": p.index, "type": p.type.value, "content": p.content}
                    for p in m.parts
                ],
            }
            for m in messages
        ]


def run_projection(run) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "status": run.status.value,
        "lane": run.lane.value,
        "intent": run.intent.value,
        "trigger_message_id": run.trigger_message_id,
        "runtime_binding_id": run.runtime_binding_id,
        "recovery_of_run_id": run.recovery_of_run_id,
        "failure_code": run.failure_code,
        "potentially_billed": run.potentially_billed,
        "deadline_at": run.deadline_at.isoformat(),
        "created_at": run.created_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
    }
