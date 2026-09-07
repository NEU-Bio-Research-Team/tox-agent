"""The product-owned gateway for one agentic run (plan section 10).

The gateway deliberately does *not* implement an agent loop.  A runtime owns
reasoning and its own local transcript; ToxAgent owns the session, the tool
authorization, observations, accepted answer and every state transition that a
client can observe.  This module is the narrow seam between those two worlds.

Only a validated ``GroundedAnswer`` becomes an assistant message.  Runtime text
deltas are not product truth: persisting them before ``submit_grounded_answer``
would allow an ungrounded number to survive in the transcript even when the
validator correctly refused the final candidate.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from ..application.create_analysis import CreateAnalysis
from ..application.run_scheduler import RunContext
from ..application.runs import advance
from ..config import RuntimeSettings
from ..domain.errors import DeadlineExceeded, RuntimeProtocolError, RuntimeUnavailable
from ..domain.events import EventType
from ..domain.evidence import EvidenceStatus
from ..domain.message import Message, PartType, Role
from ..domain.provenance import content_sha256
from ..domain.run import Intent, RunStatus
from ..domain.runtime import BindingStatus, RuntimeBinding, RuntimeKind
from ..domain.usage import RuntimeUsageEvent
from ..tools.capability import CapabilityTokenService
from ..tools.registry import ToolContext, ToolRegistry
from .context import PinnedReference, SessionCheckpoint, build_system_prompt
from .provider import (
    AgentRuntimeProvider,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeSession,
    RuntimeSessionSpec,
    RuntimeTurn,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


log = logging.getLogger("toxagent.gateway")

#: Live sweep 2026-09-06 (progress log section 14): a live run can end in
#: TURN_IDLE having called only tools, no submit_grounded_answer, and no
#: product record exists of what the runtime actually said instead — the
#: exception raised for it (below) names the symptom, not the cause. This
#: caps how much of the runtime's own final text a diagnostic log line may
#: hold: enough to see whether the model wrote a plain-prose answer instead
#: of calling the tool, never the full turn.
_DIAGNOSTIC_DELTA_PREVIEW_CHARS = 400


class AgentRuntimeGateway:
    """Run one product turn through one pinned runtime provider.

    A provider is injected rather than selected by an import side effect.  The
    composition root decides which exact adapter/binary is deployed; this
    gateway verifies its reported kind and records the corresponding manifest.
    That makes frozen scripted fixtures and future OpenCode/DSH adapters use
    the same state-machine and tool boundary.
    """

    def __init__(
        self,
        database,
        registry: ToolRegistry,
        capability_tokens: CapabilityTokenService,
        provider: AgentRuntimeProvider,
        settings: RuntimeSettings,
        *,
        create_analysis: CreateAnalysis | None = None,
        mcp_url: str = "",
    ) -> None:
        self._db = database
        self._registry = registry
        self._capability_tokens = capability_tokens
        self._provider = provider
        self._settings = settings
        self._create_analysis = create_analysis
        self._mcp_url = mcp_url

    async def execute(self, context: RunContext) -> None:
        """Drive an admitted agentic/mixed run until product completion.

        The scheduler owns the outer exception-to-failed-run policy.  Raising a
        typed error here is intentional: it prevents an unavailable or
        malformed runtime from looking like a completed conversation.
        """
        if context.intent not in {
            Intent.REPORT_QA,
            Intent.ATTRIBUTION,
            Intent.EVIDENCE_RESEARCH,
        }:
            raise RuntimeProtocolError(
                "the runtime gateway only accepts conversational intents",
                intent=context.intent.value,
            )

        if context.needs_snapshot_first:
            await self._snapshot_before_runtime(context)

        health = await self._probe_health_with_retries()
        if not health.healthy:
            raise RuntimeUnavailable("the selected runtime is not healthy", detail=health.detail)
        capabilities = await self._provider.capabilities()
        kind = self._runtime_kind()

        system_prompt, profile, deadline = await self._prepare_context(context)
        tool_schema = tuple(self._registry.descriptors(profile))
        tool_schema_hash = self._registry.schema_hash(profile)
        profile_hash = content_sha256(
            {
                "profile": profile,
                "visible_tools": [tool["name"] for tool in tool_schema],
            }
        )
        local_context = ToolContext(
            session_id=context.session_id,
            run_id=context.run_id,
            actor=context.actor,
            profile=profile,
            deadline_at=deadline,
            language=context.language,
        )
        spec = RuntimeSessionSpec(
            session_id=context.session_id,
            run_id=context.run_id,
            provider_id=self._settings.provider_id,
            model_id=self._settings.model_id,
            profile=profile,
            system_prompt=system_prompt,
            system_prompt_hash=content_sha256(system_prompt),
            tool_schema=tool_schema,
            tool_schema_hash=tool_schema_hash,
            mcp_url=self._mcp_url,
            max_steps=self._max_steps(context.intent),
            deadline_at=deadline,
            local_tool_context=local_context,
        )

        runtime_session: RuntimeSession | None = None
        binding: RuntimeBinding | None = None
        capability_token: str | None = None
        binding_lost = False
        completed = False
        provider_turn_accepted = False
        receipt = None
        try:
            runtime_session = await self._provider.create_session(spec)
            binding = RuntimeBinding.create(
                session_id=context.session_id,
                runtime_kind=kind,
                runtime_version=self._runtime_version(kind),
                runtime_session_id=runtime_session.runtime_session_id,
                provider_id=runtime_session.provider_id,
                model_id=runtime_session.model_id,
                profile_hash=profile_hash,
                tool_schema_hash=tool_schema_hash,
                system_prompt_hash=spec.system_prompt_hash,
                capabilities=capabilities,
                now=_now(),
                selection_reason="deployment-pinned runtime provider",
            )
            await self._persist_started_run(context, binding)

            # The database now has a truthful runtime session id, so the JTI
            # and the signed token can be scoped to the binding without a
            # placeholder or a post-hoc repair.
            capability_token = await self._capability_tokens.issue(
                session_id=context.session_id,
                run_id=context.run_id,
                profile=profile,
                owner_id=context.actor.subject_id,
                roles=context.actor.roles,
                runtime_binding_id=binding.id,
                deadline_at=deadline,
                language=context.language,
            )
            receipt = await self._provider.send(
                runtime_session,
                RuntimeTurn(
                    turn_id=context.run_id,
                    user_message=context.text,
                    deadline_at=deadline,
                    capability_token=capability_token,
                ),
            )
            if not receipt.accepted:
                raise RuntimeUnavailable(
                    "the runtime refused the turn", detail=receipt.detail or "no detail"
                )
            # From here on, the runtime has confirmed it received this turn —
            # whatever happens next, we no longer know the charge outcome was
            # nothing. A receipt this side never distinguishes "queued" from
            # "the provider was actually called"; treating acceptance as the
            # line is the conservative, honest reading of plan section 6.6 /
            # remaining-plan W2-12, not a claim about OpenCode's own billing
            # internals we cannot see.
            provider_turn_accepted = True
            if receipt.turn_id != context.run_id:
                raise RuntimeProtocolError(
                    "the runtime receipt names a different turn",
                    expected_turn_id=context.run_id,
                    actual_turn_id=receipt.turn_id,
                )

            binding_lost = await self._consume_events(runtime_session, context, binding, deadline)
            await self._commit_answer_and_complete(context)
            completed = True
        except asyncio.CancelledError:
            # Scheduler cancellation only becomes an honest terminal state
            # after the runtime has received its real abort request.  V1
            # supports it; adapters that do not report that precisely through
            # their own CancelOutcome.
            if runtime_session is not None and receipt is not None:
                await self._cancel_quietly(runtime_session, receipt)
            raise
        finally:
            if provider_turn_accepted and not completed:
                await self._mark_potentially_billed_quietly(context.run_id)
            if capability_token:
                await self._revoke_token_quietly(capability_token)
            if runtime_session is not None:
                closed = await self._close_quietly(runtime_session)
                if binding is not None:
                    status = BindingStatus.LOST if binding_lost or not closed else BindingStatus.CLOSED
                    await self._set_binding_status_quietly(binding.id, status)
            elif binding is not None and not completed:
                # A failure after the binding was written but before a runtime
                # session can be closed has unknown external state; never
                # advertise it as reusable.
                await self._set_binding_status_quietly(binding.id, BindingStatus.LOST)

    async def _snapshot_before_runtime(self, context: RunContext) -> None:
        if self._create_analysis is None or not context.smiles:
            raise RuntimeProtocolError(
                "a mixed run asked for a snapshot but no analysis service or SMILES was supplied"
            )
        await self._create_analysis.execute(
            actor=context.actor,
            session_id=context.session_id,
            run_id=context.run_id,
            smiles=context.smiles,
            endpoints=context.endpoints,
            threshold_overrides=context.threshold_overrides,
            owns_run=False,
        )

    async def health(self) -> bool:
        """Public readiness probe (used by ``GET /health/ready``) — the same
        check ``execute`` makes before dispatching a turn, so readiness
        reflects what a real request would actually hit rather than merely
        naming a configured runtime kind."""
        try:
            health = await self._health()
        except RuntimeUnavailable:
            return False
        return health.healthy

    async def _health(self):
        try:
            return await self._provider.health()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter errors must stay typed
            raise RuntimeUnavailable(
                "the selected runtime could not be probed", reason=type(exc).__name__
            ) from exc

    async def _probe_health_with_retries(self) -> RuntimeHealth:
        """Pre-flight probe for ``execute`` — a fresh run and a recovery run
        both start here. A single point-in-time check races a runtime that
        was restarted moments earlier (progress log §3.8/§5.2): the process
        accepts TCP connections before it has finished loading its config, so
        the very next request after a restart can still see it as unhealthy.
        Retrying a few times, bounded by ``runtime_health_check_retries``,
        gives that window a chance to close without weakening the check —
        a runtime that is genuinely down still fails after the same attempts
        it always did.
        """
        attempts = max(1, self._settings.runtime_health_check_retries)
        delay = max(0.0, self._settings.runtime_health_check_retry_delay_s)
        health: RuntimeHealth | None = None
        for attempt in range(attempts):
            try:
                health = await self._health()
            except RuntimeUnavailable as exc:
                health = RuntimeHealth(healthy=False, detail=str(exc))
            if health.healthy or attempt == attempts - 1:
                return health
            await asyncio.sleep(delay)
        return health

    def _runtime_kind(self) -> RuntimeKind:
        try:
            kind = RuntimeKind(self._provider.kind)
        except ValueError as exc:
            raise RuntimeUnavailable(
                "the configured runtime adapter reports an unknown kind",
                kind=self._provider.kind,
            ) from exc
        if self._settings.kind != kind.value:
            raise RuntimeUnavailable(
                "the configured runtime kind does not match its injected adapter",
                configured=self._settings.kind,
                adapter=kind.value,
            )
        return kind

    def _runtime_version(self, kind: RuntimeKind) -> str:
        if kind is RuntimeKind.OPENCODE:
            return self._settings.opencode_version
        if kind is RuntimeKind.DSH:
            return self._settings.dsh_version
        return "in-process-scripted-v1"

    def _max_steps(self, intent: Intent) -> int:
        return (
            self._settings.max_steps_research
            if intent is Intent.EVIDENCE_RESEARCH
            else self._settings.max_steps_qa
        )

    async def _prepare_context(self, context: RunContext) -> tuple[str, str, datetime]:
        """Load product-owned state and construct a bounded prompt projection."""
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            if run is None or run.session_id != context.session_id:
                raise RuntimeProtocolError("the runtime run does not exist in this session")
            if run.is_terminal:
                raise RuntimeProtocolError("a terminal run cannot be sent to a runtime")
            session = await uow.sessions.get_unscoped(context.session_id)
            if session is None:
                raise RuntimeProtocolError("the runtime session does not exist")
            messages = list(await uow.messages.list_for_session(context.session_id, limit=100))
            # The current user message is the RuntimeTurn payload (step seven
            # in §10.4), not part of the prefix transcript.
            recent = [message for message in messages if message.id != run.trigger_message_id]
            pinned: list[PinnedReference] = []
            # The run's own resolved target wins over whatever the session
            # happens to have active right now; it falls back to the active
            # analysis only when the run never named one. A run that
            # named a specific analysis_id and can't find it is a protocol
            # error, not a silent fall-through to the wrong molecule.
            target_analysis_id = context.analysis_id or session.active_analysis_id
            if target_analysis_id:
                analysis = await uow.analyses.get(
                    target_analysis_id, session_id=context.session_id
                )
                if analysis is None and context.analysis_id:
                    raise RuntimeProtocolError(
                        "the run's target analysis no longer exists in this session",
                        analysis_id=context.analysis_id,
                    )
                if analysis is not None:
                    pinned.append(
                        PinnedReference(
                            kind="analysis",
                            id=analysis.id,
                            summary=(
                                f"canonical SMILES={analysis.canonical_smiles}; "
                                f"sections={', '.join(analysis.served_endpoints) or 'none'}; "
                                "read values only with get_analysis_slice"
                            ),
                        )
                    )
            # Evidence already accepted earlier in this session is worth a
            # turn's model knowing about without spending a redundant
            # search_toxicology_evidence call on it (plan section 10.4 step
            # 5) — bounded small, same reasoning as the analysis reference
            # above: a pointer plus enough to judge relevance, not the value.
            accepted_evidence = await uow.evidence.list_for_session(
                context.session_id, status=EvidenceStatus.ACCEPTED, limit=5
            )
            for record in accepted_evidence:
                pinned.append(
                    PinnedReference(
                        kind="evidence",
                        id=record.id,
                        summary=f"{record.title[:120]!r}; read with get_evidence_record",
                    )
                )

        profile = self._registry.profile_for_intent(context.intent.value)
        deadline = min(
            run.deadline_at,
            _now() + timedelta(seconds=self._settings.turn_deadline_s),
        )
        if deadline <= _now():
            raise DeadlineExceeded("the run deadline elapsed before runtime dispatch")
        prompt = build_system_prompt(
            capability_profile=profile,
            checkpoint=SessionCheckpoint(),
            pinned=pinned,
            recent_messages=recent,
        )
        return prompt, profile, deadline

    async def _persist_started_run(self, context: RunContext, binding: RuntimeBinding) -> None:
        async with self._db.unit_of_work() as uow:
            current = await uow.runs.get(context.run_id)
            if current is None or current.session_id != context.session_id or current.is_terminal:
                raise RuntimeProtocolError("the run changed before runtime binding could be stored")
            await uow.runtime_bindings.add(binding)
            await advance(
                uow,
                current,
                RunStatus.RUNNING,
                runtime_binding_id=binding.id,
                payload={"runtime": binding.manifest()},
            )
            await uow.commit()

    async def _consume_events(
        self,
        runtime_session: RuntimeSession,
        context: RunContext,
        binding: RuntimeBinding,
        deadline: datetime,
    ) -> bool:
        """Wait for a normalized terminal event; return whether the binding was lost.

        Tool lifecycle is already persisted by ToolRunner.  MESSAGE_DELTA is
        deliberately not mirrored into product state — persisting it before
        ``submit_grounded_answer`` would let an ungrounded number survive in
        the transcript even when the validator correctly refuses the final
        candidate. A short, bounded tail of it is kept in memory only for the
        diagnostic log below; it is never written to the database or exposed
        over the API.
        """
        stream = self._provider.events(runtime_session, after=None)
        lost = False
        delta_tail = ""
        while True:
            remaining = (deadline - _now()).total_seconds()
            if remaining <= 0:
                if await self._has_answer(context):
                    return lost
                raise DeadlineExceeded("the runtime turn exceeded its deadline")
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except asyncio.TimeoutError:
                # `remaining` bounds this one wait, not the whole turn — the
                # deadline itself is re-checked at the top of the loop, which
                # is where DeadlineExceeded actually gets raised. Found live
                # (2026-09-05): letting this propagate raw meant the run
                # scheduler's catch-all logged a turn that simply ran out of
                # time as failure_code "internal_error" instead of the typed
                # "deadline_exceeded" this exact case exists for.
                continue
            except StopAsyncIteration:
                if await self._has_answer(context):
                    return lost
                raise RuntimeProtocolError("the runtime event stream ended without a final answer")

            if event.type is RuntimeEventType.MESSAGE_DELTA:
                delta_tail = (delta_tail + str(event.payload.get("text", "")))[
                    -_DIAGNOSTIC_DELTA_PREVIEW_CHARS:
                ]
                continue
            if event.type is RuntimeEventType.USAGE_REPORTED:
                await self._record_usage_event(context, binding, event)
                continue
            if event.type is RuntimeEventType.TURN_IDLE:
                if delta_tail and not await self._has_answer(context):
                    log.warning(
                        "run %s reached TURN_IDLE with no submit_grounded_answer call; "
                        "last %d chars the runtime wrote instead: %r",
                        context.run_id, len(delta_tail), delta_tail,
                    )
                return lost
            if event.type is RuntimeEventType.SESSION_LOST:
                lost = True
                # A persisted, validated answer is sufficient product state.
                # Do not turn it into a failed report merely because the
                # provider died after the authoritative tool call completed.
                if await self._has_answer(context):
                    return lost
                raise RuntimeUnavailable("the runtime session was lost", **event.payload)
            if event.type is RuntimeEventType.TURN_FAILED:
                if await self._has_answer(context):
                    return lost
                raise RuntimeProtocolError("the runtime turn failed", **event.payload)

    async def _record_usage_event(
        self, context: RunContext, binding: RuntimeBinding, event: RuntimeEvent
    ) -> None:
        """Persist a report as received, without inventing a total.

        A zero is faithfully retained; an absent/malformed field is ``None``.
        The event is an immutable audit record because providers disagree on
        whether a later report is a delta or a cumulative snapshot.
        """
        usage = RuntimeUsageEvent.from_provider_payload(
            session_id=context.session_id,
            run_id=context.run_id,
            runtime_binding_id=binding.id,
            provider_id=binding.provider_id,
            model_id=binding.model_id,
            payload=event.payload,
            reported_at=event.occurred_at,
        )
        async with self._db.unit_of_work() as uow:
            await uow.runtime_usage.add(usage)
            uow.emit(
                session_id=context.session_id,
                type=EventType.RUNTIME_USAGE_REPORTED,
                entity_type="runtime_usage",
                entity_id=usage.id,
                run_id=context.run_id,
                payload={
                    "usage_event_id": usage.id,
                    "provider_id": usage.provider_id,
                    "model_id": usage.model_id,
                },
            )
            await uow.commit()

    async def _has_answer(self, context: RunContext) -> bool:
        async with self._db.unit_of_work() as uow:
            return await uow.answers.get_for_run(context.run_id) is not None

    async def _commit_answer_and_complete(self, context: RunContext) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            if run is None:
                raise RuntimeProtocolError("the runtime run disappeared before completion")
            answer = await uow.answers.get_for_run(context.run_id)
            if answer is None:
                raise RuntimeProtocolError(
                    "the runtime reached a terminal event without submit_grounded_answer"
                )
            if run.status is not RunStatus.RUNNING:
                raise RuntimeProtocolError(
                    "the run changed before its accepted answer could be committed",
                    status=run.status.value,
                )
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id,
                Role.ASSISTANT,
                sequence,
                now=_now(),
                parts=(
                    (PartType.TEXT, {"text": answer.answer_markdown}),
                    (PartType.ANSWER_REF, {"answer_id": answer.id}),
                ),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=context.session_id,
                type=EventType.MESSAGE_CREATED,
                entity_type="message",
                entity_id=reply.id,
                run_id=context.run_id,
                payload={"role": "assistant", "answer_id": answer.id},
            )
            await advance(
                uow,
                run,
                RunStatus.COMPLETED,
                payload={"answer_id": answer.id, "is_fallback": answer.is_fallback},
            )
            await uow.commit()

    async def _revoke_token_quietly(self, token: str) -> None:
        try:
            claims = await self._capability_tokens.verify(token)
            await self._capability_tokens.revoke(claims.jti)
        except Exception:  # noqa: BLE001 - expiry/revocation must not hide a run result
            return

    async def _close_quietly(self, runtime_session: RuntimeSession) -> bool:
        try:
            return (await self._provider.close(runtime_session)).closed
        except Exception:  # noqa: BLE001 - status below records that the state is unknown
            return False

    async def _cancel_quietly(self, runtime_session: RuntimeSession, receipt) -> None:
        try:
            await self._provider.cancel(runtime_session, receipt)
        except Exception:  # noqa: BLE001 - scheduler records the exact terminal state
            return

    async def _set_binding_status_quietly(self, binding_id: str, status: BindingStatus) -> None:
        try:
            async with self._db.unit_of_work() as uow:
                await uow.runtime_bindings.set_status(binding_id, status.value, now=_now())
                await uow.commit()
        except Exception:
            # The worker will already have written a typed run failure if this
            # operation was reached on an error path.  Do not replace it with
            # a secondary persistence exception that says less.
            return

    async def _mark_potentially_billed_quietly(self, run_id: str) -> None:
        """Read-then-write, deliberately separate from the terminal
        ``advance()`` call the scheduler makes right after this (plan section
        6.6 / remaining-plan W2-12/15): this runs from ``execute()``'s own
        ``finally``, before the exception that triggered it has even reached
        the scheduler's exception handler, and a terminal transition is not
        this method's job. ``Run.mark_potentially_billed`` bumps version, so
        the scheduler's own fresh ``uow.runs.get()`` immediately afterward
        sees it and its transition preserves the flag (nothing in
        ``advance()``/``transition()`` touches it)."""
        try:
            async with self._db.unit_of_work() as uow:
                run = await uow.runs.get(run_id)
                if run is None:
                    return
                await uow.runs.update(run.mark_potentially_billed(), expected_version=run.version)
                await uow.commit()
        except Exception:
            # Best-effort audit enrichment; never let it mask the real
            # terminal failure the scheduler is about to record.
            return
