"""What a report build's fact bundle is made of, read from the database (PR-12).

PR-10 made the bundle a pure function of what the stages produced, and left the
reading to "the caller". This is the caller. It exists as one module because
three places need exactly the same bundle — the synthesis tool that judges a
submission, the validating stage that re-judges it, and the rendering stage
that publishes it — and a bundle each of them assembled for itself is three
chances for a report to be validated against facts it is not rendered from.

The bundle is recomputed rather than stored. Every input is immutable (the
analysis snapshot, its observations, promoted evidence rows) and every fact id
is a hash of the build id and a field path, so the same build yields the same
bundle byte for byte. Storing it would add a second copy of the facts with no
way to tell which one was stale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..domain.errors import Conflict
from ..domain.evidence import EvidenceRecord, EvidenceStatus
from ..domain.observation import Observation, ObservationKind
from ..domain.report import BuildStage, ExplanationPackage, ReportBuild, SubstanceProfile
from ..report.fact_bundle import ReportFactBundle
from ..validation.report_semantics import EvidenceSituation
from .explanation import is_explanation_schema, package_from_observation
from .report_stage_handlers import bundle_from_checkpoints
from .report_stages import CHECKPOINT_SCHEMA_VERSION, read_checkpoints, reusable_outputs

#: Always carried. A report is a screening document whatever it found, and the
#: one limitation that must never depend on what a model remembered to include
#: is the one that says so.
ALWAYS_REQUIRED_LIMITATIONS: tuple[str, ...] = ("screening_not_safety_assessment",)


@dataclass(frozen=True, slots=True)
class ReportInputs:
    build: ReportBuild
    bundle: ReportFactBundle
    explanations: tuple[ExplanationPackage, ...]
    #: Promoted evidence, as the bundle and the artifact's references see it.
    evidence: tuple[dict[str, Any], ...]
    subject: SubstanceProfile
    situation: EvidenceSituation

    @property
    def search_performed(self) -> bool:
        return self.situation.search_performed

    @property
    def explanations_by_id(self) -> dict[str, ExplanationPackage]:
        return {package.explanation_id: package for package in self.explanations}


def evidence_view(record: EvidenceRecord) -> dict[str, Any]:
    """A promoted record, as a report may quote it. Still untrusted text."""
    assessment = dict(record.relevance_assessment or {})
    return {
        "evidence_id": record.id,
        "title": record.title,
        "canonical_url": record.canonical_url,
        "provider": record.provider,
        "authors": list(record.authors[:8]),
        "published_at": record.published_at.isoformat() if record.published_at else None,
        "identifier": record.identifier.to_dict(),
        "source_type": record.source_type.value,
        "source_quality_tier": record.source_quality_tier.value,
        "retrieved_at": record.retrieved_at.isoformat(),
        "relevance": assessment.get("relevance"),
        "relevance_reason_codes": list(assessment.get("reason_codes") or ()),
        "untrusted_external_content": True,
    }


async def load_report_inputs(database, build_id: str, *, session_id: str) -> ReportInputs:
    async with database.unit_of_work() as uow:
        build = await uow.reports.get_build(build_id, session_id=session_id)
        if build is None:
            raise Conflict("no such report build in this session", build_id=build_id)
        snapshot = await uow.analyses.get(build.analysis_id, session_id=session_id)
        if snapshot is None:
            raise Conflict(
                "the analysis this build was created for no longer resolves",
                analysis_id=build.analysis_id,
            )
        observations = list(await uow.observations.list_for_analysis(snapshot.id))
        researched = reusable_outputs(read_checkpoints(build.stage_state)).get(
            BuildStage.RESEARCHING_EVIDENCE.value
        ) or {}
        records: list[dict[str, Any]] = []
        for evidence_id in researched.get("promoted_evidence_ids") or ():
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            # Promoted when the stage ran, and still citable now. A record
            # rejected since is not quietly cited because a checkpoint said so.
            if record is not None and record.status is EvidenceStatus.ACCEPTED:
                records.append(evidence_view(record))
    return assemble_inputs(build, snapshot, observations, records)


def assemble_inputs(
    build: ReportBuild,
    snapshot: Any,
    observations: Sequence[Observation],
    evidence: Sequence[Mapping[str, Any]],
) -> ReportInputs:
    """Pure: the loaded rows, into the bundle and everything published beside it."""
    completed = reusable_outputs(read_checkpoints(build.stage_state))
    request = build.request

    prediction = next(
        (item for item in observations if item.kind is ObservationKind.PREDICTION), None
    )
    observation_ids = (
        {endpoint: prediction.id for endpoint in snapshot.served_endpoints}
        if prediction is not None
        else {}
    )

    explained = completed.get(BuildStage.GENERATING_EXPLANATIONS.value) or {}
    wanted = set((explained.get("explanations") or {}).values())
    packages: dict[str, ExplanationPackage] = {}
    for item in observations:
        if not is_explanation_schema(item.schema_version):
            continue
        package = package_from_observation(item)
        if package.explanation_id in wanted and package.explanation_id not in packages:
            packages[package.explanation_id] = package
    explanation_gaps = [
        {
            "reason": "explanation_failed",
            "section_id": "explanation_and_visuals",
            "detail": f"{target}: {reason}",
            "endpoint": target.split(".", 1)[0],
            "task": target.split(".", 1)[1] if "." in target else None,
        }
        for target, reason in sorted((explained.get("failed_targets") or {}).items())
    ]

    limitations: list[str] = []
    for code in (
        *(prediction.required_limitations if prediction is not None else ()),
        *(code for package in packages.values() for code in package.required_limitations),
        *(("evidence_scope_limited",) if request.include_external_evidence else ()),
        *ALWAYS_REQUIRED_LIMITATIONS,
    ):
        if code not in limitations:
            limitations.append(code)

    researched = completed.get(BuildStage.RESEARCHING_EVIDENCE.value) or {}
    search_performed = bool(researched.get("search_performed"))
    situation = EvidenceSituation(
        requested=request.include_external_evidence,
        search_performed=search_performed,
        candidates_found=len(evidence),
        provider_failed=request.include_external_evidence and not search_performed,
        promoted=len(evidence),
    )

    substance = dict(completed.get(BuildStage.ASSEMBLING_SUBSTANCE.value) or {})
    subject = SubstanceProfile(
        canonical_smiles=snapshot.canonical_smiles,
        preferred_name=substance.get("preferred_name") or None,
        synonyms=tuple(str(name) for name in (substance.get("synonyms") or ())[:10]),
        identifiers=dict(substance.get("identifiers") or {}),
    )

    bundle = bundle_from_checkpoints(
        report_build_id=build.id,
        analysis_id=snapshot.id,
        completed=completed,
        predictions=(snapshot.predictor_response or {}).get("predictions") or {},
        explanations=tuple(packages.values()),
        evidence=evidence,
        observation_ids=observation_ids,
        required_limitations=limitations,
        policy={
            "include_external_evidence": request.include_external_evidence,
            "include_explanations": request.include_explanations,
            "audience": request.audience,
            "search_performed": search_performed,
            "evidence_provider_failed": situation.provider_failed,
        },
        provenance={
            "analysis_id": snapshot.id,
            "analysis_content_sha256": snapshot.content_sha256,
            "report_build_id": build.id,
            "stage_checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        },
        extra_gaps=explanation_gaps,
        language=request.report_language,
    )
    return ReportInputs(
        build=build,
        bundle=bundle,
        explanations=tuple(packages.values()),
        evidence=tuple(dict(item) for item in evidence),
        subject=subject,
        situation=situation,
    )
