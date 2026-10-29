"""Preparing a turn: the prompt context, the AI profile, the pre-runtime snapshot."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from ...application.investigation import case_view, goal_contract, router, scientific_case_service
from ...application.investigation.skill_catalog import render_index, render_static
from ...application.runs.scheduler import RunContext
from ...connections.model import ConnectionStatus
from ...domain import decision_state, scientific_case
from ...domain.errors import (
    DeadlineExceeded,
    PhaseBudgetExhausted,
    RuntimeProtocolError,
    RuntimeUnavailable,
)
from ...domain.evidence import EvidenceStatus
from ...domain.observation import ObservationKind
from ...domain.run import Intent
from ...domain.runtime import AuthMode
from ...platform.flags import is_enabled
from ...application.conversation.subject_resolution import binding_of, subject_summary
from ...domain.subject import SubjectKind
from ..context import PinnedReference, SessionCheckpoint, build_system_prompt
from ..report_profile import compose_report_profile
from ._common import ResolvedProfile, _now


class ContextMixin:
    """Everything assembled before the runtime is dispatched."""

    #: Ledger entries that restate the run's own artifacts rather than a
    #: source's measurement (the same split as the case view's groups).
    _ARTIFACT_FACT_CLASSES = frozenset({"predictor_fact", "explanation_fact", "report_fact"})

    async def _session_candidates(self, context: RunContext) -> list[dict[str, Any]]:
        """The subjects a compare question could be about: each screening, and
        each analysed structure that is not a screening's own ligand -- a
        screening's toxicity analysis is that screening, not a second candidate."""
        async with self._db.unit_of_work() as uow:
            screenings = await uow.screenings.list_for_session(context.session_id)
            analyses = await uow.analyses.list_for_session(context.session_id, limit=50)
        ligands = {
            ((a.record.get("screening_state") or {}).get("subject") or {}).get("canonical_smiles")
            or ((a.record.get("activity_estimate") or {}).get("subject") or {})
            .get("canonical_smiles")
            for a in screenings
        }
        rows = [{"subject_key": f"screening:{a.screening_id}"} for a in screenings]
        seen: set[str] = set()
        for analysis in analyses:
            if analysis.canonical_smiles in ligands or analysis.canonical_smiles in seen:
                continue
            seen.add(analysis.canonical_smiles)
            rows.append({"subject_key": f"analysis:{analysis.id}"})
        return rows

    async def _researcher_observations(self, context: RunContext, case) -> dict[str, str]:
        """The case's researcher values in force, as observations of this run (#61).

        A claim can only cite an observation, and a value the researcher typed
        into the case was not one -- which is why no arm could quote an
        exposure the researcher supplied (live run 2026-10-22, T5 0/2). Each
        current version is projected once per run; a context build that runs
        again for the same run reuses what it projected.
        """
        from ...application.investigation.researcher_values import project_value
        from ...domain.observation import ObservationKind
        from ...domain.scientific_case import current_values

        if case is None or not case.context:
            return {}
        async with self._db.unit_of_work() as uow:
            existing = {
                o.canonical_payload.get("ref"): o.id
                for o in await uow.observations.list_for_run(context.run_id)
                if o.kind is ObservationKind.RESEARCHER_VALUE
            }
            projected = dict(existing)
            for version in current_values(case):
                if version.ref in projected:
                    continue
                observation = project_value(
                    version, case_id=case.id, session_id=context.session_id,
                    run_id=context.run_id, now=_now(),
                )
                await uow.observations.add(observation)
                projected[version.ref] = observation.id
            if len(projected) != len(existing):
                await uow.commit()
        return projected

    def _contract_and_view(
        self, context: RunContext, *, case, profile: str, pinned, carried_over,
        researcher_observations: dict[str, str] | None = None,
        previous_contract: Mapping[str, Any] | None = None,
        candidates: Sequence[Mapping[str, Any]] = (),
    ):
        """#69/#63/#68: the route, the contract and the six-group view.

        Built here because this is where the subject, the case, the carried
        propositions and the registered tools are all in scope. Stashed for
        ``_begin_decision_state`` to record, because the state is opened after
        the prompt -- the contract has to be the one the turn actually read,
        not one rebuilt later from the same inputs.
        """
        from ...application.runs.budget import effective_run_budget
        from ...domain.scientific_case import current_values, needs_reassessment
        from ...platform.config import PolicySettings

        stale = [c.id for c in needs_reassessment(case.derived_claims)] if case else []
        artifact_refs = [f"{p.kind}:{p.id}" for p in pinned if p.kind != "report_build"]
        in_force = list(current_values(case)) if case else []
        # §4.1 update route: new is what the previous run of this subject did
        # not see -- read off its contract, a recorded fact, not the wording.
        # A subject with no earlier contract has seen nothing yet.
        seen = set((previous_contract or {}).get("researcher_values_seen") or ())
        new_values = [
            {"ref": v.ref, "key": v.key, "value": v.value, "unit": v.unit}
            for v in in_force if v.ref not in seen
        ]
        decision = router.classify(
            question=context.text or "",
            has_subject=any(p.kind in ("analysis", "screening") for p in pinned),
            artifact_refs=artifact_refs,
            stale_claims=stale,
            new_researcher_input=bool(new_values),
            candidate_count=len(candidates),
        )
        # §4.1 "Cách chạy": the route decides the capability profile, so a
        # read is cheaper in fact -- fewer tools, no search, no skills.
        profile = router.profile_for(decision.route)
        budget = effective_run_budget(
            context.intent.value, self._policy or PolicySettings(), self._settings
        ).to_dict()
        needs, gap = goal_contract.requirements(
            context.text or "",
            has_measurement=any(
                entry.source_class not in self._ARTIFACT_FACT_CLASSES
                for entry in (case.evidence if case else ())
            ),
            # A researcher value keyed as an exposure. Keys are the researcher's
            # words, so this reads the two that name one, and only those.
            has_exposure=any(
                any(word in version.key.lower() for word in ("cmax", "exposure"))
                for version in (current_values(case) if case else ())
            ),
        )
        contract = goal_contract.build(
            route=decision.route,
            question=context.text or "",
            question_needs=needs,
            question_gap=gap,
            subject=dict(case.subject) if case is not None else {},
            admissible_actions=[t["name"] for t in self._registry.descriptors(profile)],
            budget=budget,
            open_questions=carried_over,
            stale_claims=stale,
            new_values=new_values if decision.route == router.UPDATE else (),
            researcher_values_seen=[v.ref for v in in_force],
            candidates=candidates if decision.route == router.COMPARE_BATCH else (),
            clarification=decision.clarification,
        )
        searched = [
            action.action for action in (case.actions if case else ())
            if action.action.startswith("search")
        ]
        view = case_view.build(
            case, contract=contract, researcher_observations=researcher_observations or {},
            open_gaps=[
                {"gap": q.gap, "question": q.question}
                for q in contract.must_answer if q.gap
            ],
            searched=searched,
            budget_left={k: v for k, v in budget.items()
                         if k in ("deadline_s", "max_tool_calls", "max_searches",
                                  "max_evidence_reads") and v is not None},
        )
        self._contracts[context.run_id] = contract.to_dict()
        return view.render(), decision, profile

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

    def _review_reserve_s(self, context: RunContext) -> int:
        """Seconds of the run deadline kept for review and repair, when the
        preset keeps any and the run's answer will be reviewed."""
        from ...platform.config import release_policy, run_budget_preset, total_call_accounting
        from .release import release_gate_on

        if context.intent is not Intent.DECISION_SUPPORT or not total_call_accounting():
            return 0
        reviewed = release_gate_on() or release_policy() == "strict-reviewed-v1"
        return int(run_budget_preset()[1].get("review_reserve_s") or 0) if reviewed else 0

    async def _resolve_pending_subject(self, context: RunContext) -> RunContext:
        """#44: a new-molecule run's subject is the analysis its snapshot just
        made. Bind it now, before the case and the prompt read the subject."""
        from dataclasses import replace

        binding = binding_of(context)
        if binding is None or not binding.pending_snapshot:
            return context
        async with self._db.unit_of_work() as uow:
            session = await uow.sessions.get_unscoped(context.session_id)
            analysis = (
                await uow.analyses.get(session.active_analysis_id, session_id=context.session_id)
                if session is not None and session.active_analysis_id else None
            )
        if analysis is None:
            raise RuntimeProtocolError("the run's snapshot did not produce an analysis to bind")
        resolved = binding.resolved_with_analysis(
            analysis.id, canonical_smiles=analysis.canonical_smiles,
            content_sha256_=analysis.content_sha256,
        )
        async with self._db.unit_of_work() as uow:
            await uow.run_configuration_snapshots.resolve_subject_binding(
                context.run_id, resolved.to_dict(),
            )
            await uow.commit()
        return replace(context, subject=resolved.to_dict(), analysis_id=analysis.id)

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
            binding = binding_of(context)
            # The run's own resolved target wins over whatever the session
            # happens to have active right now; it falls back to the active
            # analysis only when the run never named one. A run that
            # named a specific analysis_id and can't find it is a protocol
            # error, not a silent fall-through to the wrong molecule.
            # #44: a bound run never falls back -- a screening subject has no
            # analysis of its own, and the session's active one is not it.
            target_analysis_id = (
                binding.analysis_id if binding is not None
                else context.analysis_id or session.active_analysis_id
            )
            if binding is not None and binding.kind is SubjectKind.ACTIVITY_SCREENING:
                pinned.append(PinnedReference(
                    kind="screening", id=binding.screening_id or "",
                    summary=subject_summary(binding),
                ))
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
            run_case = None
            carried_over: list = []
            previous_contract: dict = {}
            if context.intent is Intent.DECISION_SUPPORT and is_enabled("scientific_case_v1"):
                run_case = await scientific_case_service.case_for_run(
                    uow, session_id=context.session_id, run_id=context.run_id
                )
                if run_case is not None:
                    case_summary = scientific_case.checkpoint_summary(run_case)
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
                subject = (
                    binding.subject_ref_string if binding is not None
                    else f"analysis:{target_analysis_id}" if target_analysis_id else None
                )
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
                    previous_contract = dict(same_subject[0].goal_contract or {})
                    checkpoint = SessionCheckpoint(
                        summary=decision_state.checkpoint_summary(same_subject[0]),
                        open_intent="; ".join(
                            p.question[:120] for p in same_subject[0].unresolved[:3]
                        ),
                    )
                    carried_over = [
                        {"question": p.question, "required_sources": list(p.required_sources)}
                        for p in same_subject[0].unresolved[:4]
                    ]
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
        route_decision = None
        if context.intent is Intent.DECISION_SUPPORT and is_enabled("investigator_contract_v1"):
            researcher_observations = await self._researcher_observations(context, run_case)
            case_summary, route_decision, profile = self._contract_and_view(
                context, case=run_case, profile=profile, pinned=pinned,
                carried_over=carried_over, researcher_observations=researcher_observations,
                previous_contract=previous_contract,
                candidates=await self._session_candidates(context),
            )
        self._routes[context.run_id] = route_decision
        deadline = min(
            run.deadline_at,
            _now() + timedelta(seconds=(
                self._settings.report_turn_deadline_s
                if context.intent is Intent.BUILD_REPORT
                else self._settings.turn_deadline_s
            )),
        )
        reserve = self._review_reserve_s(context)
        if reserve:
            # #46 (investigator-pilot-v2): one root deadline, and the review
            # and the repair are guaranteed its last seconds; the investigator
            # cannot spend them.
            deadline = min(deadline, run.deadline_at - timedelta(seconds=reserve))
            if deadline <= _now():
                raise PhaseBudgetExhausted(
                    "no time is left for the investigation before the review's reserve",
                    phase="investigator", review_reserve_s=reserve,
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
                and not any(p.kind in ("analysis", "screening") for p in pinned)
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
