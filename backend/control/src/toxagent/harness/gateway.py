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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

from ..application.create_analysis import CreateAnalysis
from ..application.run_scheduler import RunContext
from ..connections.model import ConnectionStatus
from ..connections.secrets import SecretStore
from ..domain.runtime import AuthMode
from ..application.runs import advance
from ..config import RuntimeSettings
from ..domain.errors import DeadlineExceeded, RuntimeProtocolError, RuntimeUnavailable
from ..domain.events import EventType
from ..domain.evidence import EvidenceStatus
from ..domain.message import Message, PartType, Role
from ..domain.observation import ObservationKind
from ..domain.provenance import content_sha256
from ..domain.run import Intent, RunStatus
from ..domain.report import BuildStage, ReportBuild, ReportBuildRequest
from ..domain.runtime import BindingStatus, RuntimeBinding, RuntimeKind
from ..domain.usage import RuntimeUsageEvent
from ..flags import is_enabled
from ..tools.capability import CapabilityTokenService
from ..tools.registry import ToolContext, ToolRegistry
from ..application import decision_state_service, scientific_case_service
from ..domain import decision_state, scientific_case
from .context import PinnedReference, SessionCheckpoint, build_system_prompt
from .report_profile import compose_report_profile
from ..application.skill_catalog import load_catalog, render_index, render_static
from .synthesis_profile import PROFILE_NAME as SYNTHESIS_PROFILE, compose_synthesis_profile
from .prompt_budget import measure as measure_prompt, split_system_prompt
from .runtime_profiles import RuntimeProfileRegistry
from .usage_normalizer import RuntimeUsageNormalizer
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


async def _no_commit(context: RunContext) -> None:
    """A turn whose product is committed by its caller, not by the gateway."""
    return None


@dataclass(frozen=True)
class ResolvedProfile:
    """Everything a run's AI profile actually configured.

    A tuple of (provider, model, connection_id) was what the gateway used to
    resolve, which is why the endpoint and the credential never reached the
    runtime (I12).
    """

    provider_id: str
    model_id: str
    connection_id: str | None
    base_url: str | None
    credential: str | None
    auth_mode: AuthMode


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
        secrets: SecretStore | None = None,
        profiles_dir: Path | None = None,
        policy=None,
    ) -> None:
        self._db = database
        #: Read only to state a decision_support run's budget snapshot.
        self._policy = policy
        self._registry = registry
        self._capability_tokens = capability_tokens
        self._provider = provider
        self._settings = settings
        self._create_analysis = create_analysis
        self._mcp_url = mcp_url
        # Reading a stored credential is what makes a chosen AI profile
        # actually take effect (I12). A gateway without one refuses any run
        # whose profile has a credential, rather than dispatching it under the
        # runtime host's own authentication.
        self._secrets = secrets
        self._profiles_dir = profiles_dir
        # Loaded once and validated: a malformed skill package fails start-up
        # rather than a run whose record would claim instructions it never got.
        from ..config import PACKAGE_ROOT

        self._skill_catalog = load_catalog(profiles_dir or PACKAGE_ROOT / "agent_profiles")
        # Which agent and which *real* step cap each intent runs under. Built
        # once: it reads the shipped agent profile files, and those do not
        # change under a running process.
        self._missing_agents: tuple[str, ...] = ()
        self._runtime_profiles = RuntimeProfileRegistry(
            profiles_dir=profiles_dir,
            agent_name=settings.agent_name,
            report_agent_name=getattr(settings, "report_agent_name", "toxagent-report"),
            max_steps_qa=settings.max_steps_qa,
            max_steps_research=settings.max_steps_research,
            max_steps_report=settings.max_steps_report,
            runtime_kind=settings.kind,
            select_named_agents=is_enabled("runtime_profile_selector_v2"),
        )

    async def execute(self, context: RunContext) -> None:
        """Drive an admitted agentic/mixed run until product completion.

        The scheduler owns the outer exception-to-failed-run policy.  Raising a
        typed error here is intentional: it prevents an unavailable or
        malformed runtime from looking like a completed conversation.
        """
        if context.intent not in {
            Intent.DECISION_SUPPORT,
            Intent.BUILD_REPORT,
        }:
            raise RuntimeProtocolError(
                "the runtime gateway only accepts conversational intents",
                intent=context.intent.value,
            )

        if context.needs_snapshot_first:
            await self._snapshot_before_runtime(context)

        if context.intent is Intent.BUILD_REPORT:
            from dataclasses import replace
            context = replace(
                context, report_build_id=await self._ensure_report_build(context)
            )

        keeps_case = (
            context.intent is Intent.DECISION_SUPPORT and is_enabled("scientific_case_v1")
        )
        if keeps_case:
            # Before the prompt is built, so the turn sees its own case.
            await self._begin_scientific_case(context)
        system_prompt, profile, deadline, instructions_hash = await self._prepare_context(context)
        if context.intent is not Intent.DECISION_SUPPORT:
            await self._dispatch(
                context,
                system_prompt=system_prompt,
                profile=profile,
                deadline=deadline,
                instructions_hash=instructions_hash,
                has_product=self._has_answer,
                commit=self._commit_product_and_complete,
            )
            return

        await self._begin_decision_state(context)
        mode, offered = self._skills_for(profile)
        if mode != "off":
            await decision_state_service.advance(
                self._db, context.run_id,
                lambda state: decision_state.record_skills_offered(
                    state, mode=mode, offered=[skill.pin() for skill in offered],
                ),
            )
        run_status = "failed"
        try:
            await self._dispatch(
                context,
                system_prompt=system_prompt,
                profile=profile,
                deadline=deadline,
                instructions_hash=instructions_hash,
                has_product=self._has_answer,
                commit=(
                    self._commit_with_case if keeps_case else self._commit_product_and_complete
                ),
            )
            run_status = "completed"
        except asyncio.CancelledError:
            run_status = "cancelled"
            raise
        finally:
            await decision_state_service.advance(
                self._db, context.run_id,
                lambda state: decision_state.finalize(state, run_status=run_status),
            )
            if keeps_case and run_status != "completed":
                # A run that ended without committing its answer: the case
                # records how it ended and the dossier says so. A completed
                # run did this in _commit_with_case, before it completed.
                await scientific_case_service.finish_run(
                    self._db, session_id=context.session_id, run_id=context.run_id,
                )

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
        from ..application.run_budget import effective_run_budget
        from ..config import PolicySettings

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

    async def snapshot_before_runtime(self, context: RunContext) -> None:
        """The mixed-run snapshot, for a caller that drives its own workflow."""
        await self._snapshot_before_runtime(context)

    async def ensure_report_build(self, context: RunContext) -> str:
        """Create the build manifest without walking any stage (WS05).

        The orchestrator emits a stage event when a stage actually starts; the
        old path's walk through five stages at one timestamp is exactly what it
        replaces, so it must not happen on the way in either.
        """
        return await self._ensure_report_build(context, walk_stages=False)

    async def run_report_synthesis(
        self,
        context: RunContext,
        *,
        report_build_id: str,
        fact_view: dict,
        deadline_at: datetime | None,
        has_synthesis: Callable[[RunContext], Awaitable[bool]],
    ) -> None:
        """Dispatch the one LLM turn of an orchestrated report build (PR-12).

        The prompt is the synthesis brief and the fact bundle, nothing else: no
        answer-format guide, no required-limitations guide and no transcript,
        because none of them describe work this turn can do. The capability
        token names ``report_synthesis``, so the runtime sees one tool.

        Completion is not this method's business. The turn ends, and the
        orchestrator's synthesizing handler decides from the stored build
        whether a synthesis was accepted.
        """
        if self._profiles_dir is None:
            raise RuntimeProtocolError("the report synthesis profile directory is not configured")
        from dataclasses import replace
        import json

        composed = compose_synthesis_profile(self._profiles_dir)
        profile = SYNTHESIS_PROFILE
        system_prompt = "\n\n".join(
            (
                composed.instructions,
                f"Capability profile for this turn: {profile}. Only the tools listed by "
                "the MCP server for this connection exist.",
                "Pinned references:\n"
                f"- report_build {report_build_id}: the fact bundle below is the only "
                "source of values for this report.\n"
                "```json\n"
                + json.dumps(fact_view, sort_keys=True, ensure_ascii=False, default=str)
                + "\n```",
            )
        )
        now = _now()
        deadline = now + timedelta(seconds=self._settings.report_turn_deadline_s)
        if deadline_at is not None:
            deadline = min(deadline, deadline_at)
        if deadline <= now:
            raise DeadlineExceeded("the report build deadline elapsed before synthesis")
        await self._dispatch(
            replace(
                context,
                report_build_id=report_build_id,
                text=f"Write and submit the synthesis for report build {report_build_id}.",
            ),
            system_prompt=system_prompt,
            profile=profile,
            deadline=deadline,
            instructions_hash=composed.content_sha256,
            has_product=has_synthesis,
            commit=_no_commit,
        )

    async def _dispatch(
        self,
        context: RunContext,
        *,
        system_prompt: str,
        profile: str,
        deadline: datetime,
        instructions_hash: str | None,
        has_product: Callable[[RunContext], Awaitable[bool]],
        commit: Callable[[RunContext], Awaitable[None]],
    ) -> None:
        """One runtime turn: bind, authorize, send, consume, commit, close."""
        health = await self._probe_health_with_retries()
        if not health.healthy:
            raise RuntimeUnavailable("the selected runtime is not healthy", detail=health.detail)
        capabilities = await self._provider.capabilities()
        kind = self._runtime_kind()

        resolved = await self._resolve_ai_profile(context)
        tool_schema = tuple(self._registry.descriptors(profile))
        tool_schema_hash = self._registry.schema_hash(profile)
        profile_hash = content_sha256(
            {
                "profile": profile,
                "visible_tools": [tool["name"] for tool in tool_schema],
                "instructions_sha256": instructions_hash,
            }
        )
        runtime_profile = self._runtime_profiles.resolve(
            context.intent.value, capability_profile=profile
        )
        # What this dispatch actually costs, by component. Recorded on the
        # binding rather than logged: "the report used 28,408 tokens" is not
        # actionable, and the audit had no way to say which part of the prompt
        # they were (P1-10).
        policy_prefix, pinned_facts, history = split_system_prompt(system_prompt)
        budget = measure_prompt(
            profile=profile,
            policy_prefix=policy_prefix,
            tool_schemas=tool_schema,
            pinned_facts=pinned_facts,
            history=history,
            user_message=context.text,
            context={"model_id": resolved.model_id, "intent": context.intent.value},
        )
        if budget.over_budget_by:
            log.info(
                "prompt is over its component budget",
                extra={
                    "run_id": context.run_id,
                    "profile": profile,
                    "total_tokens": budget.total_tokens,
                    "target_tokens": budget.target_tokens,
                    "largest_component": (
                        budget.largest_component.name if budget.largest_component else None
                    ),
                },
            )
        discrepancy = runtime_profile.discrepancy
        if discrepancy:
            # Dispatch still happens — refusing a report because its profile
            # is one step short would be worse than running it — but the
            # difference is stated, never inferred later from a truncated run.
            log.warning(
                "runtime step cap will not be honoured",
                extra={
                    "run_id": context.run_id,
                    "intent": context.intent.value,
                    "runtime_agent_name": runtime_profile.runtime_agent_name,
                    "requested_step_cap": runtime_profile.requested_step_cap,
                    "effective_step_cap": runtime_profile.effective_step_cap,
                    "detail": discrepancy,
                },
            )
        local_context = ToolContext(
            session_id=context.session_id,
            run_id=context.run_id,
            actor=context.actor,
            profile=profile,
            deadline_at=deadline,
            language=context.language,
            intent=context.intent.value,
            # Server-injected, alongside session_id and run_id and for the
            # same reason: a tool must inherit the run's predictor binding
            # rather than let the model choose one (I10).
            model_selection=context.model_selection,
        )
        spec = RuntimeSessionSpec(
            session_id=context.session_id,
            run_id=context.run_id,
            provider_id=resolved.provider_id,
            model_id=resolved.model_id,
            profile=profile,
            system_prompt=system_prompt,
            system_prompt_hash=content_sha256(system_prompt),
            tool_schema=tool_schema,
            tool_schema_hash=tool_schema_hash,
            mcp_url=self._mcp_url,
            max_steps=runtime_profile.requested_step_cap,
            deadline_at=deadline,
            runtime_agent_name=runtime_profile.runtime_agent_name,
            effective_max_steps=runtime_profile.effective_step_cap,
            connection_id=resolved.connection_id,
            provider_base_url=resolved.base_url,
            provider_credential=resolved.credential,
            auth_mode=resolved.auth_mode.value,
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
                connection_id=resolved.connection_id,
                profile_hash=profile_hash,
                tool_schema_hash=tool_schema_hash,
                system_prompt_hash=spec.system_prompt_hash,
                capabilities=capabilities,
                now=_now(),
                selection_reason="deployment-pinned runtime provider",
                runtime_manifest={
                    **runtime_profile.to_manifest(),
                    "prompt_budget": budget.to_manifest(),
                },
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
                intent=context.intent.value,
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

            binding_lost = await self._consume_events(
                runtime_session, context, binding, deadline, has_product
            )
            await commit(context)
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
        # Every field of the run's resolved configuration, not a subset (I08).
        # This dropped model_selection and the explanation settings, so a
        # session that had chosen model B could snapshot with the default A
        # the moment the question happened to arrive with a molecule attached
        # — and the run's own config and its result then disagreed.
        await self._create_analysis.execute(
            actor=context.actor,
            session_id=context.session_id,
            run_id=context.run_id,
            smiles=context.smiles,
            endpoints=context.endpoints,
            model_selection=context.model_selection,
            threshold_overrides=context.threshold_overrides,
            explanation_mode=context.explanation_mode,
            explanation_targets=context.explanation_targets,
            owns_run=False,
        )

    async def _resolve_ai_profile(self, context: RunContext) -> ResolvedProfile:
        """Resolve the run-pinned AI profile immediately before dispatch.

        A session stores just an opaque profile id. Resolving it here keeps a
        browser from supplying a provider/model pair directly to the runtime,
        proves ownership again at the execution boundary, and writes the
        chosen connection into the immutable runtime binding.

        I12: this used to return the provider/model pair and stop, leaving the
        endpoint and the credential in the database. The runtime then used
        whatever authentication its host happened to have, so a profile that
        passed its connection test proved nothing about which account a run
        would bill or which server it would reach. The credential is read here,
        at the trust boundary that has already re-proved ownership, and handed
        to the adapter for this turn only.
        """
        if context.ai_profile_id is None:
            return ResolvedProfile(
                provider_id=self._settings.provider_id,
                model_id=self._settings.model_id,
                connection_id=None,
                base_url=None,
                credential=None,
                auth_mode=AuthMode.NONE,
            )
        async with self._db.unit_of_work() as uow:
            connection = await uow.model_connections.get(
                context.ai_profile_id, owner_id=context.actor.subject_id
            )
        if connection is None:
            raise RuntimeUnavailable("the selected AI provider profile is unavailable")
        if connection.status is not ConnectionStatus.READY:
            raise RuntimeUnavailable(
                "the selected AI provider profile has not passed its connection test",
                profile_id=connection.id,
                status=connection.status.value,
            )
        credential = None
        if connection.credential_ref:
            if self._secrets is None:
                # Never proceed without it: the run would silently use the
                # runtime host's own credentials instead of this owner's.
                raise RuntimeUnavailable(
                    "this deployment cannot read the credential for the selected AI "
                    "profile, so the run would use the runtime's own authentication",
                    profile_id=connection.id,
                )
            credential = self._secrets.get(connection.credential_ref)
        return ResolvedProfile(
            provider_id=connection.provider_id,
            model_id=connection.model_id,
            connection_id=connection.id,
            base_url=connection.base_url,
            credential=credential,
            auth_mode=connection.auth_mode,
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

    @property
    def missing_runtime_agents(self) -> tuple[str, ...]:
        """Named agents the last probe found absent from the runtime host.

        Read by the capability resolver, which is synchronous: the probe that
        fills this runs on readiness and before every dispatch, so the value is
        as fresh as the last time anything asked the runtime a question.
        """
        return self._missing_agents

    async def _health(self):
        try:
            health = await self._provider.health()
            self._missing_agents = tuple(getattr(health, "missing_agents", ()) or ())
            return health
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

    def agent_for_capability(self, capability: str) -> str | None:
        return self._runtime_profiles.agent_for_capability(capability)

    def _runtime_version(self, kind: RuntimeKind) -> str:
        if kind is RuntimeKind.OPENCODE:
            return self._settings.opencode_version
        if kind is RuntimeKind.DSH:
            return self._settings.dsh_version
        return "in-process-scripted-v1"

    def _max_steps(self, intent: Intent) -> int:
        if intent is Intent.BUILD_REPORT:
            return self._settings.max_steps_report
        return (
            self._settings.max_steps_research
            if intent is Intent.DECISION_SUPPORT
            else self._settings.max_steps_qa
        )

    async def _prepare_context(self, context: RunContext) -> tuple[str, str, datetime, str | None]:
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
            checkpoint = SessionCheckpoint()
            case_summary = ""
            if context.intent is Intent.DECISION_SUPPORT and is_enabled("scientific_case_v1"):
                case = await scientific_case_service.case_for_run(
                    uow, session_id=context.session_id, run_id=context.run_id
                )
                if case is not None:
                    case_summary = scientific_case.checkpoint_summary(case)
            if context.intent is Intent.DECISION_SUPPORT:
                # P1-05: memory is scoped to the subject, not the transcript.
                # Prior decision states for *this* analysis contribute their
                # resolved/open propositions and the evidence they actually
                # used; states about another compound contribute nothing, so a
                # subject switch cannot pull the wrong compound's evidence in.
                prior = [
                    state for state in await uow.decision_states.list_for_session(
                        context.session_id, limit=20
                    )
                    if state.run_id != context.run_id and state.stop_reason is not None
                ]
                subject = f"analysis:{target_analysis_id}" if target_analysis_id else None
                same_subject = [s for s in prior if subject and subject in s.subject_refs]
                if prior:
                    scoped_ids = {
                        ref.split(":", 1)[1]
                        for state in same_subject
                        for ref in (
                            *state.available_refs,
                            *state.found_refs,
                            *(r for p in state.propositions for r in p.artifact_refs),
                        )
                        if ref.startswith("evidence:")
                    }
                    accepted_evidence = [r for r in accepted_evidence if r.id in scoped_ids]
                if same_subject:
                    checkpoint = SessionCheckpoint(
                        summary=decision_state.checkpoint_summary(same_subject[0]),
                        open_intent="; ".join(
                            p.question[:120] for p in same_subject[0].unresolved[:3]
                        ),
                    )
            for record in accepted_evidence:
                pinned.append(
                    PinnedReference(
                        kind="evidence",
                        id=record.id,
                        summary=f"{record.title[:120]!r}; read with get_evidence_record",
                    )
                )
            if context.intent is Intent.DECISION_SUPPORT and target_analysis_id:
                # ADS plan section 8.1/W3: which explanations already exist for
                # this analysis, so a follow-up turn reuses get_explanation_slice
                # / get_attribution instead of guessing from prose or recomputing
                # one (RC-04). Pointer only — endpoint/task/observation_id — the
                # highlights themselves are still read through those tools.
                explanation_observations = [
                    obs
                    for obs in await uow.observations.list_for_analysis(target_analysis_id)
                    if obs.kind is ObservationKind.ATTRIBUTION
                ]
                for obs in explanation_observations:
                    endpoint = obs.model_projection.get("endpoint", "?")
                    task = obs.model_projection.get("task")
                    target = f"{endpoint}/{task}" if task else endpoint
                    pinned.append(
                        PinnedReference(
                            kind="explanation",
                            id=obs.id,
                            summary=(
                                f"endpoint/task={target}; already computed — read with "
                                "get_explanation_slice or get_attribution instead of "
                                "recomputing"
                            ),
                        )
                    )
                # W3-03: the latest non-superseded report for this analysis, now
                # that get_report_summary (W2-04) gives the model a tool that can
                # actually read it — pinning an id with nothing to read it with
                # would have been a false affordance, which is why this was
                # deferred past the first ADS pass. "Latest" is the same
                # version-DESC query get_report_summary itself defaults to, so
                # the pointer and the read agree about which report "latest"
                # means.
                latest_report = await uow.reports.get_latest_artifact_for_analysis(
                    target_analysis_id, session_id=context.session_id
                )
                if latest_report is not None:
                    gaps = latest_report.get("gaps") or []
                    recommendations = latest_report.get("recommendations") or []
                    pinned.append(
                        PinnedReference(
                            kind="report",
                            id=latest_report.get("report_id") or latest_report.get("id"),
                            summary=(
                                f"status={latest_report.get('status')}; "
                                f"{len(gaps)} open gap(s), {len(recommendations)} "
                                "recommendation(s) already on file — read with "
                                "get_report_summary before treating a gap as evidence "
                                "against anything"
                            ),
                        )
                    )
            if context.intent is Intent.BUILD_REPORT:
                builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
                build = next((item for item in builds if item.id == context.report_build_id), None)
                if build is None:
                    raise RuntimeProtocolError("the report build manifest is missing")
                pinned.append(PinnedReference(
                    kind="report_build", id=build.id,
                    summary="call get_report_context with this report_build_id before any other work",
                ))

        profile = self._registry.profile_for_intent(context.intent.value)
        deadline = min(
            run.deadline_at,
            _now() + timedelta(seconds=(
                self._settings.report_turn_deadline_s
                if context.intent is Intent.BUILD_REPORT
                else self._settings.turn_deadline_s
            )),
        )
        if deadline <= _now():
            raise DeadlineExceeded("the run deadline elapsed before runtime dispatch")
        skills_mode, offered_skills = self._skills_for(profile)
        prompt = build_system_prompt(
            capability_profile=profile,
            checkpoint=checkpoint,
            pinned=pinned,
            recent_messages=recent,
            scientific_case=case_summary,
            scientific_skills=(
                render_index(offered_skills) if skills_mode == "dynamic"
                else render_static(offered_skills) if skills_mode == "static" else ""
            ),
            answer_schema=(
                "grounded-answer-v2" if is_enabled("answer_draft_v2") else "grounded-answer-v1"
            ),
        )
        instructions_hash = None
        if context.intent is Intent.BUILD_REPORT:
            if self._profiles_dir is None:
                raise RuntimeProtocolError("the report instruction profile directory is not configured")
            composed = compose_report_profile(self._profiles_dir)
            prompt = composed.instructions + "\n\n---\n\n# Run context\n\n" + prompt
            instructions_hash = composed.content_sha256
            async with self._db.unit_of_work() as uow:
                builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
                build = next((item for item in builds if item.id == context.report_build_id), None)
                if build is not None:
                    from dataclasses import replace
                    state = dict(build.stage_state)
                    state["instruction_manifest"] = composed.manifest()
                    await uow.reports.save_build(replace(build, stage_state=state, updated_at=_now()))
                    await uow.commit()
        return prompt, profile, deadline, instructions_hash

    async def _ensure_report_build(self, context: RunContext, *, walk_stages: bool = True) -> str:
        """Create the durable build manifest after an optional new snapshot exists.

        ``walk_stages`` is the old path's behaviour and stays its default: the
        model-driven build expects to find the pointer at synthesizing.
        """
        now = _now()
        async with self._db.unit_of_work() as uow:
            existing = await uow.reports.list_builds_for_session(context.session_id, limit=50)
            current = (
                next((item for item in existing if item.id == context.report_build_id), None)
                if context.report_build_id
                else next((item for item in existing if item.run_id == context.run_id), None)
            )
            if current is not None:
                return current.id
            session = await uow.sessions.get_unscoped(context.session_id)
            analysis_id = context.analysis_id or (session.active_analysis_id if session else None)
            snapshot = (
                await uow.analyses.get(analysis_id, session_id=context.session_id)
                if analysis_id else None
            )
            if snapshot is None:
                raise RuntimeProtocolError("a report build requires an immutable analysis snapshot")
            selected = tuple(context.endpoints or snapshot.served_endpoints)
            unavailable = sorted(set(selected) - set(snapshot.served_endpoints))
            if unavailable:
                raise RuntimeProtocolError(
                    "selected report endpoints are not served by this analysis",
                    unavailable_endpoints=unavailable,
                )
            request = ReportBuildRequest(
                session_id=context.session_id,
                analysis_id=snapshot.id,
                selected_endpoints=selected,
                selected_tox21_tasks=tuple(
                    task for endpoint, task in context.explanation_targets
                    if endpoint == "tox21" and task
                ),
                report_language=context.report_language,
                audience=context.report_audience,
                include_explanations=context.explanation_mode != "none",
                include_external_evidence=context.include_external_evidence,
                output_formats=context.report_output_formats,
            )
            build = ReportBuild.start(
                session_id=context.session_id, run_id=context.run_id, request=request,
                now=now,
                deadline_at=now + timedelta(seconds=self._settings.report_turn_deadline_s),
            )
            await uow.reports.add_build(build)
            uow.emit(
                session_id=context.session_id, type=EventType.REPORT_BUILD_STARTED,
                entity_type="report_build", entity_id=build.id, run_id=context.run_id,
                payload={"analysis_id": snapshot.id, "selected_endpoints": list(selected)},
            )
            if not walk_stages:
                await uow.commit()
                return build.id
            stages = [
                BuildStage.PREPARING_ANALYSIS, BuildStage.ASSEMBLING_SUBSTANCE,
                BuildStage.ASSEMBLING_PREDICTIONS,
            ]
            if request.include_explanations:
                stages.append(BuildStage.GENERATING_EXPLANATIONS)
            if request.include_external_evidence:
                stages.append(BuildStage.RESEARCHING_EVIDENCE)
            stages.append(BuildStage.SYNTHESIZING)
            for stage in stages:
                build = build.advance(stage, now=now)
            await uow.reports.save_build(build)
            uow.emit(
                session_id=context.session_id, type=EventType.REPORT_STAGE_CHANGED,
                entity_type="report_build", entity_id=build.id, run_id=context.run_id,
                payload={"stage": build.stage.value},
            )
            await uow.commit()
            return build.id

    async def _persist_started_run(self, context: RunContext, binding: RuntimeBinding) -> None:
        async with self._db.unit_of_work() as uow:
            current = await uow.runs.get(context.run_id)
            if current is None or current.session_id != context.session_id or current.is_terminal:
                raise RuntimeProtocolError("the run changed before runtime binding could be stored")
            await uow.runtime_bindings.add(binding)
            if current.status is RunStatus.RUNNING:
                # An orchestrated build started its run before any runtime was
                # involved: the deterministic stages are work too. Attach the
                # binding without pretending the run started a second time —
                # and attach it, because recovery after a lost runtime is only
                # permitted for a run that demonstrably had one.
                from dataclasses import replace
                await uow.runs.update(
                    replace(
                        current, runtime_binding_id=binding.id, version=current.version + 1
                    ),
                    expected_version=current.version,
                )
                await uow.commit()
                return
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
        has_product: Callable[[RunContext], Awaitable[bool]] | None = None,
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
        has_product = has_product or self._has_answer
        stream = self._provider.events(runtime_session, after=None)
        # One normalizer per runtime session. It is the fast path only: the
        # partial unique index on (runtime_binding_id, source_event_id) is what
        # holds when a worker is replaced mid-run and the replacement re-reads
        # the stream from the beginning.
        normalizer = (
            RuntimeUsageNormalizer(
                runtime_session.runtime_session_id, provider=self._provider.kind
            )
            if is_enabled("normalized_usage_v2")
            else None
        )
        lost = False
        delta_tail = ""
        while True:
            remaining = (deadline - _now()).total_seconds()
            if remaining <= 0:
                if await has_product(context):
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
                if await has_product(context):
                    return lost
                raise RuntimeProtocolError("the runtime event stream ended without a final answer")

            if event.type is RuntimeEventType.MESSAGE_DELTA:
                delta_tail = (delta_tail + str(event.payload.get("text", "")))[
                    -_DIAGNOSTIC_DELTA_PREVIEW_CHARS:
                ]
                continue
            if event.type is RuntimeEventType.USAGE_REPORTED:
                await self._record_usage_event(context, binding, event, normalizer)
                continue
            if event.type is RuntimeEventType.TURN_IDLE:
                if delta_tail and not await has_product(context):
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
                if await has_product(context):
                    return lost
                raise RuntimeUnavailable("the runtime session was lost", **event.payload)
            if event.type is RuntimeEventType.TURN_FAILED:
                if await has_product(context):
                    return lost
                raise RuntimeProtocolError("the runtime turn failed", **event.payload)

    async def _record_usage_event(
        self,
        context: RunContext,
        binding: RuntimeBinding,
        event: RuntimeEvent,
        normalizer: "RuntimeUsageNormalizer | None" = None,
    ) -> None:
        """Persist a report as a fact, without inventing a total.

        A zero is faithfully retained; an absent/malformed field is ``None``.
        With normalization on, a report that restates a total already held
        establishes no new fact, and nothing is written or emitted — which is
        the difference between 21 rows and the 7 the audit's Q&A run actually
        measured (P1-1). Providers disagree on whether a later report is a
        delta or a cumulative snapshot, so the row says which this one is
        rather than leaving a reader to assume.
        """
        normalized = None
        if normalizer is not None:
            normalized = normalizer.accept(event.payload)
            if normalized is None:
                return
        usage = RuntimeUsageEvent.from_provider_payload(
            session_id=context.session_id,
            run_id=context.run_id,
            runtime_binding_id=binding.id,
            provider_id=binding.provider_id,
            model_id=binding.model_id,
            payload=event.payload,
            reported_at=event.occurred_at,
            source_event_id=normalized.source_event_id if normalized else None,
            source_event_type=normalized.source_event_type if normalized else None,
            provider_message_id=normalized.provider_message_id if normalized else None,
            provider_step_id=normalized.provider_step_id if normalized else None,
            revision=normalized.revision if normalized else None,
            semantics=normalized.semantics.value if normalized else "unknown",
            is_normalized=normalized is not None,
            raw_payload_hash=normalized.raw_payload_hash if normalized else None,
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
            if context.intent is Intent.BUILD_REPORT:
                builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
                build = next((item for item in builds if item.id == context.report_build_id), None)
                return bool(build and build.report_id)
            return await uow.answers.get_for_run(context.run_id) is not None

    async def _commit_product_and_complete(self, context: RunContext) -> None:
        if context.intent is Intent.BUILD_REPORT:
            await self._commit_report_and_complete(context)
            return
        await self._commit_answer_and_complete(context)

    async def _commit_report_and_complete(self, context: RunContext) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
            build = next((item for item in builds if item.id == context.report_build_id), None)
            if run is None or build is None or not build.report_id:
                # Say which of the three it was. "without submit_report_draft"
                # was emitted for all of them, including the common case where
                # the tool *was* called and the draft was refused — which sent
                # whoever read the failure looking for a runtime that skipped a
                # tool call, when the actual record showed two calls and two
                # rejections.
                raise RuntimeProtocolError(self._report_failure_detail(run, build))
            artifact = await uow.reports.get_artifact(
                build.report_id, session_id=context.session_id
            )
            if artifact is None:
                raise RuntimeProtocolError("the accepted report artifact cannot be reconstructed")
            if run.status is not RunStatus.RUNNING:
                raise RuntimeProtocolError(
                    "the run changed before its accepted report could be committed",
                    status=run.status.value,
                )
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id, Role.ASSISTANT, sequence, now=_now(),
                parts=(
                    (PartType.TEXT, {
                        "text": f"Report completed: {artifact['title']}",
                    }),
                    (PartType.REPORT_REF, {
                        "report_id": artifact["report_id"],
                        "report_build_id": build.id,
                        "status": artifact["status"],
                    }),
                ),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=context.session_id, type=EventType.MESSAGE_CREATED,
                entity_type="message", entity_id=reply.id, run_id=context.run_id,
                payload={"role": "assistant", "report_id": artifact["report_id"]},
            )
            await advance(
                uow, run, RunStatus.COMPLETED,
                payload={"report_id": artifact["report_id"], "report_build_id": build.id},
            )
            await uow.commit()

    @staticmethod
    def _report_failure_detail(run, build) -> str:
        """Why no report exists, in the terms the record actually supports."""
        if run is None:
            return "the runtime run disappeared before completion"
        if build is None:
            return "the report build this run was started for no longer resolves"
        if build.stage is BuildStage.FAILED:
            reason = build.failure_detail or build.failure_code or "no reason recorded"
            return (
                "submit_report_draft was called and the draft did not pass validation, so "
                f"the build failed and no report was produced: {reason}"
            )
        return (
            "the runtime reached a terminal event without a report: the build is "
            f"{build.stage.value} and submit_report_draft never produced an artifact"
        )

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
            parts: list[tuple[PartType, dict]] = [
                (PartType.TEXT, {"text": answer.answer_markdown}),
                (PartType.ANSWER_REF, {"answer_id": answer.id}),
            ]
            if context.intent is Intent.DECISION_SUPPORT and is_enabled("scientific_case_v1"):
                # Stored before the run completes (_commit_with_case), so it
                # exists here for every completed case run.
                dossier = await uow.scientific_cases.get_dossier(
                    context.run_id, session_id=context.session_id
                )
                if dossier is not None:
                    parts.append((PartType.DOSSIER_REF, {
                        "case_id": dossier["case_id"], "run_id": context.run_id,
                        "case_revision": dossier.get("case_revision"),
                    }))
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id,
                Role.ASSISTANT,
                sequence,
                now=_now(),
                parts=tuple(parts),
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
