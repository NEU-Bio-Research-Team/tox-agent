"""The deterministic half of a report build (WS05 5B / PR-10).

PR-09 gave the orchestrator its loop, its checkpoints and its event discipline
and left the handlers injected. These are the handlers for everything before the
model is involved: resolve the snapshot, resolve the substance, project the
predictions, ensure the explanations exist, run the evidence search. All of it
is work the server was already capable of doing and had been asking the model
to orchestrate instead — 18,149 input tokens and 155 seconds to arrange calls
the control plane could make itself.

Each handler is built over a narrow **port**: a callable the caller supplies.
Nothing here imports a predictor client, a provider or a unit of work. That is
not ceremony — it is what makes "does the evidence stage skip correctly when
the build did not ask for evidence" a test that runs in a millisecond instead
of one that needs a database and a network double.

Two disciplines every handler shares:

**A checkpoint carries refs, not payloads.** A stage returns the ids of what it
produced. The payloads live where they live; copying them into the checkpoint
would give an artifact two sources for one fact, which is the failure the fact
bundle exists to prevent.

**Declining is not failing.** A compound with no resolvable identity, or a
search the build did not ask for, raises ``StageSkipped`` with a reason. The
orchestrator records it, the bundle turns it into a typed gap, and the report
says which of the four evidence states actually happened (P0-2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..domain.report import BuildStage, ExplanationPackage, ExplanationStatus
from ..report.fact_bundle import ReportFactBundle, assemble
from .report_orchestrator import StageContext, StageSkipped

# --- ports ------------------------------------------------------------------
#
# Everything the handlers need from the outside world, named as functions
# rather than as service objects, so a test supplies a lambda and production
# supplies a bound method.


class AnalysisPort(Protocol):
    async def __call__(self, analysis_id: str, *, session_id: str) -> Any | None: ...


class SubstancePort(Protocol):
    async def __call__(self, *, canonical_smiles: str) -> Mapping[str, Any] | None: ...


class ExplanationPort(Protocol):
    async def __call__(
        self, *, analysis_id: str, endpoint: str, task: str | None
    ) -> ExplanationPackage | None: ...


class EvidencePort(Protocol):
    async def __call__(
        self,
        *,
        analysis_id: str,
        endpoint: str,
        compound_names: Sequence[str],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...


# --- handlers ---------------------------------------------------------------


def prepare_analysis(analyses: AnalysisPort) -> Callable[[StageContext], Awaitable[dict]]:
    """Pin the exact snapshot this build is about.

    Resolved once, here, rather than re-resolved by every later stage: a build
    that looked up "the session's current analysis" twice could straddle two
    molecules if the user submitted another one mid-build.
    """

    async def handler(context: StageContext) -> dict[str, Any]:
        build = context.build
        snapshot = await analyses(build.analysis_id, session_id=build.session_id)
        if snapshot is None:
            raise LookupError(
                f"report build {build.id} names analysis {build.analysis_id}, "
                "which does not exist in this session"
            )
        served = tuple(snapshot.served_endpoints)
        selected = tuple(build.request.selected_endpoints)
        return {
            "analysis_id": snapshot.id,
            "canonical_smiles": snapshot.canonical_smiles,
            "served_endpoints": list(served),
            # Computed here and carried forward, because an endpoint that was
            # asked for and is not served is a gap, and the easiest way to lose
            # one is never to subtract (SCI-06).
            "unavailable_endpoints": [e for e in selected if e not in served],
            "content_sha256": snapshot.content_sha256,
        }

    return handler


def assemble_substance(
    substances: SubstancePort | None,
) -> Callable[[StageContext], Awaitable[dict]]:
    """Resolve identity, or decline with a reason.

    A provider outage is not a failed report. It is a report whose substance
    section says the identity could not be resolved — which is exactly what the
    audit's build said, and the only part of that artifact that was honest.
    """

    async def handler(context: StageContext) -> dict[str, Any]:
        prepared = context.completed.get(BuildStage.PREPARING_ANALYSIS.value) or {}
        smiles = str(prepared.get("canonical_smiles") or "")
        if not smiles:
            raise StageSkipped("this build has no canonical SMILES to resolve")
        if substances is None:
            raise StageSkipped("no compound provider is configured for this deployment")
        try:
            record = await substances(canonical_smiles=smiles)
        except Exception as exc:  # noqa: BLE001 - a provider outage is a gap
            raise StageSkipped(
                f"the compound provider could not be reached: {type(exc).__name__}"
            ) from exc
        if not record:
            raise StageSkipped("the compound database has no record for this structure")
        return dict(record)

    return handler


def assemble_predictions() -> Callable[[StageContext], Awaitable[dict]]:
    """Nothing to fetch. The snapshot is already immutable and already read.

    This stage exists so the *projection* is checkpointed: which endpoints this
    build will speak about, and which it will record as gaps, decided once and
    recorded rather than recomputed differently in two places.
    """

    async def handler(context: StageContext) -> dict[str, Any]:
        prepared = context.completed.get(BuildStage.PREPARING_ANALYSIS.value) or {}
        served = list(prepared.get("served_endpoints") or ())
        selected = list(context.build.request.selected_endpoints)
        return {
            "endpoints": [e for e in selected if e in served],
            "gap_endpoints": [e for e in selected if e not in served],
            "tox21_tasks": list(context.build.request.selected_tox21_tasks),
        }

    return handler


def generate_explanations(
    explanations: ExplanationPort,
) -> Callable[[StageContext], Awaitable[dict]]:
    """Every required target gets a package, or a target-specific gap.

    Done before synthesis, not during it. The audit's build had the model
    requesting explanations in the middle of writing, which is how a report
    came to describe an explanation whose numbers had not arrived yet.
    """

    async def handler(context: StageContext) -> dict[str, Any]:
        request = context.build.request
        if not request.include_explanations:
            raise StageSkipped("this build did not ask for explanations")

        projected = context.completed.get(BuildStage.ASSEMBLING_PREDICTIONS.value) or {}
        endpoints = list(projected.get("endpoints") or ())
        tasks = list(projected.get("tox21_tasks") or ())

        produced: dict[str, str] = {}
        failed: dict[str, str] = {}
        for endpoint in endpoints:
            targets = [(endpoint, task) for task in tasks] if endpoint == "tox21" else [
                (endpoint, None)
            ]
            for target_endpoint, task in targets:
                key = target_endpoint if task is None else f"{target_endpoint}.{task}"
                try:
                    package = await explanations(
                        analysis_id=context.build.analysis_id,
                        endpoint=target_endpoint,
                        task=task,
                    )
                except Exception as exc:  # noqa: BLE001 - per-target, not per-build
                    failed[key] = type(exc).__name__
                    continue
                if package is None or package.status is ExplanationStatus.FAILED:
                    failed[key] = (
                        package.failure_reason if package else "no explanation was produced"
                    ) or "no explanation was produced"
                    continue
                produced[key] = package.explanation_id

        if not produced and failed:
            # Nothing worked. A report with no explanations at all should say
            # so once, as a stage-level gap, rather than as one gap per target.
            raise StageSkipped(
                f"no explanation could be produced ({len(failed)} target(s) failed)"
            )
        return {"explanations": produced, "failed_targets": failed}

    return handler


def research_evidence(
    evidence: EvidencePort | None,
) -> Callable[[StageContext], Awaitable[dict]]:
    """The server builds the query and owns the budget.

    Not the model: a query plan made of names the model supplied would let it
    search for what it expected to find, and a budget it interprets is a budget
    (P1-2 persisted five papers against a limit of two).
    """

    async def handler(context: StageContext) -> dict[str, Any]:
        request = context.build.request
        if not request.include_external_evidence:
            raise StageSkipped("this build did not ask for external evidence")
        if evidence is None:
            raise StageSkipped("no literature provider is configured for this deployment")

        substance = context.completed.get(BuildStage.ASSEMBLING_SUBSTANCE.value) or {}
        names = [
            name
            for name in (
                substance.get("preferred_name"),
                *(substance.get("synonyms") or ()),
            )
            if name
        ]
        projected = context.completed.get(BuildStage.ASSEMBLING_PREDICTIONS.value) or {}
        endpoints = list(projected.get("endpoints") or ())
        if not endpoints:
            raise StageSkipped("this build has no served endpoint to search evidence for")

        promoted: list[str] = []
        searched: list[str] = []
        failed: dict[str, str] = {}
        for endpoint in endpoints:
            try:
                records = await evidence(
                    analysis_id=context.build.analysis_id,
                    endpoint=endpoint,
                    compound_names=names,
                    limit=EVIDENCE_LIMIT_PER_ENDPOINT,
                )
            except Exception as exc:  # noqa: BLE001 - a provider outage is a gap
                failed[endpoint] = type(exc).__name__
                continue
            searched.append(endpoint)
            promoted.extend(
                str(record.get("evidence_id"))
                for record in records
                if record.get("evidence_id")
            )

        if failed and not searched:
            raise StageSkipped(
                f"the literature provider could not be reached ({', '.join(sorted(failed))})"
            )
        return {
            "searched_endpoints": searched,
            "promoted_evidence_ids": promoted,
            "failed_endpoints": failed,
            # Distinguishes "searched and found nothing" from "never searched",
            # which the orchestrator's gap derivation needs and the audit's
            # report conflated.
            "search_performed": bool(searched),
        }

    return handler


#: Per-endpoint promotion ceiling for a report build's own search. Deliberately
#: small: a report cites a handful of directly relevant papers or none, and a
#: larger number is a retrieval problem wearing the costume of thoroughness.
EVIDENCE_LIMIT_PER_ENDPOINT = 3


# --- putting it together ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeterministicHandlers:
    """The assembly stages, ready for ``ReportOrchestrator``."""

    analyses: AnalysisPort
    explanations: ExplanationPort
    substances: SubstancePort | None = None
    evidence: EvidencePort | None = None

    def as_mapping(self) -> dict[BuildStage, Callable[[StageContext], Awaitable[dict]]]:
        return {
            BuildStage.PREPARING_ANALYSIS: prepare_analysis(self.analyses),
            BuildStage.ASSEMBLING_SUBSTANCE: assemble_substance(self.substances),
            BuildStage.ASSEMBLING_PREDICTIONS: assemble_predictions(),
            BuildStage.GENERATING_EXPLANATIONS: generate_explanations(self.explanations),
            BuildStage.RESEARCHING_EVIDENCE: research_evidence(self.evidence),
        }


def bundle_from_checkpoints(
    *,
    report_build_id: str,
    analysis_id: str,
    completed: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Any],
    explanations: Sequence[ExplanationPackage] = (),
    evidence: Sequence[Mapping[str, Any]] = (),
    observation_ids: Mapping[str, str] | None = None,
    required_limitations: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    language: str = "en",
) -> ReportFactBundle:
    """Assemble the bundle from what the stages checkpointed.

    The seam between PR-09's loop and PR-10's facts: the orchestrator produced
    refs, this turns them into the facts a report may state. A stage that
    declined leaves its refs absent, and the bundle records the corresponding
    gap rather than a silence.
    """
    prepared = completed.get(BuildStage.PREPARING_ANALYSIS.value) or {}
    substance = completed.get(BuildStage.ASSEMBLING_SUBSTANCE.value) or {}
    projected = completed.get(BuildStage.ASSEMBLING_PREDICTIONS.value) or {}
    researched = completed.get(BuildStage.RESEARCHING_EVIDENCE.value) or {}

    extra_gaps: list[dict[str, str]] = []
    if researched and not researched.get("search_performed"):
        extra_gaps.append(
            {
                "reason": "provider_unavailable",
                "section_id": "external_evidence",
                "detail": "the literature provider could not be reached",
            }
        )

    served = list(prepared.get("served_endpoints") or projected.get("endpoints") or ())
    return assemble(
        report_build_id=report_build_id,
        analysis_id=analysis_id,
        predictions=predictions,
        served_endpoints=served,
        selected_endpoints=list(projected.get("endpoints") or ())
        + list(projected.get("gap_endpoints") or ()),
        selected_tox21_tasks=list(projected.get("tox21_tasks") or ()),
        substance=substance or None,
        explanations=explanations,
        evidence=evidence,
        observation_ids=observation_ids,
        required_limitations=required_limitations,
        policy=policy,
        provenance=provenance,
        extra_gaps=extra_gaps,
        language=language,
    )
