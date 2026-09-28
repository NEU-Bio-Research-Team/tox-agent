"""``AgentRuntimeGateway``: the entry points and the dispatch; the rest is in the mixins."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from ...application.investigation import decision_state_service, scientific_case_service
from ...application.investigation.skill_catalog import load_catalog
from ...application.prediction.create_analysis import CreateAnalysis
from ...application.runs.scheduler import RunContext
from ...connections.secrets import SecretStore
from ...domain import decision_state
from ...domain.errors import RuntimeProtocolError, RuntimeUnavailable
from ...domain.provenance import content_sha256
from ...domain.run import Intent
from ...domain.runtime import BindingStatus, RuntimeBinding
from ...platform.config import RuntimeSettings
from ...platform.flags import is_enabled
from ...tools.capability import CapabilityTokenService
from ...tools.registry import ToolContext, ToolRegistry
from ..prompt_budget import measure as measure_prompt
from ..prompt_budget import split_system_prompt
from ..provider import (
    AgentRuntimeProvider,
    RuntimeSession,
    RuntimeSessionSpec,
    RuntimeTurn,
)
from ..runtime_profiles import RuntimeProfileRegistry
from ._common import _now, log
from .case import ScientificCaseMixin
from .completion import CompletionMixin
from .context import ContextMixin
from .health import HealthMixin
from .reports import ReportsMixin


class AgentRuntimeGateway(ScientificCaseMixin, ContextMixin, ReportsMixin, CompletionMixin, HealthMixin):
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
        from ...platform.config import PACKAGE_ROOT

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
                commit=self._with_claim_review(
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
