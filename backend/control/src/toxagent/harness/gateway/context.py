"""Preparing a turn: the prompt context, the AI profile, the pre-runtime snapshot."""
from __future__ import annotations

from datetime import datetime, timedelta

from ...application.investigation import scientific_case_service
from ...application.investigation.skill_catalog import render_index, render_static
from ...application.runs.scheduler import RunContext
from ...connections.model import ConnectionStatus
from ...domain import decision_state, scientific_case
from ...domain.errors import DeadlineExceeded, RuntimeProtocolError, RuntimeUnavailable
from ...domain.evidence import EvidenceStatus
from ...domain.observation import ObservationKind
from ...domain.run import Intent
from ...domain.runtime import AuthMode
from ...platform.flags import is_enabled
from ..context import PinnedReference, SessionCheckpoint, build_system_prompt
from ..report_profile import compose_report_profile
from ._common import ResolvedProfile, _now


class ContextMixin:
    """Everything assembled before the runtime is dispatched."""

    async def snapshot_before_runtime(self, context: RunContext) -> None:
        """The mixed-run snapshot, for a caller that drives its own workflow."""
        await self._snapshot_before_runtime(context)

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
            subjectless=(
                context.intent is Intent.DECISION_SUPPORT
                and is_enabled("subjectless_research_v1")
                and not any(p.kind == "analysis" for p in pinned)
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
