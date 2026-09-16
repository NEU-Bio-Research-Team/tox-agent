"""Row <-> domain conversion.

Isolated from the repositories so the mapping can be read and tested as one
piece. Two details that bite otherwise:

* SQLite hands back naive datetimes even for ``DateTime(timezone=True)``. Every
  timestamp is normalised to UTC-aware on the way out, so comparisons between a
  stored deadline and ``datetime.now(timezone.utc)`` cannot raise or, worse,
  silently compare a naive value as if it were local time.
* Enums are stored as their values, not their names, so a Python rename does not
  invalidate existing rows.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

from ...domain.analysis import AnalysisSnapshot, PredictorProvenance
from ...domain.answer import Claim, ClaimKind, GroundedAnswer, Limitation, LimitationCode, RecommendedNextStep
from ...domain.attachment import Attachment, RetentionClass
from ...domain.evidence import (
    EvidenceRecord,
    EvidenceStatus,
    SourceIdentifier,
    SourceQualityTier,
    SourceType,
)
from ...domain.events import Event, EventType
from ...domain.evidence_relation import (
    Applicability,
    Directness,
    EvidenceRelationAssessment,
    EvidenceScope,
    RelationLabel,
    SourceClass,
    SourceRef,
    Strength,
)
from ...domain.development_posture import DevelopmentPosture, DevelopmentScope, PostureValue
from ...domain.message import Message, MessagePart, PartType, Role
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.report import BuildStage, ReportBuild, ReportBuildRequest
from ...domain.run import Intent, Lane, Run, RunStatus
from ...domain.runtime import AuthMode, BindingStatus, RuntimeBinding, RuntimeCapabilities, RuntimeKind
from ...domain.usage import RuntimeUsageEvent
from ...domain.session import Language, Session, SessionStatus, TitleSource


def utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


# --- session ---------------------------------------------------------------

def session_to_row(session: Session, client_session_id: str | None = None) -> dict[str, Any]:
    return {
        "id": session.id,
        "owner_id": session.owner_id,
        "status": session.status.value,
        "preferred_language": session.preferred_language.value,
        "title": session.title,
        "title_source": session.title_source.value if session.title_source else None,
        "title_status": session.title_status,
        "title_updated_at": session.title_updated_at,
        "active_analysis_id": session.active_analysis_id,
        "context_epoch": session.context_epoch,
        "event_sequence": session.event_sequence,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "version": session.version,
        "client_session_id": client_session_id,
    }


def row_to_session(row: Mapping[str, Any]) -> Session:
    return Session(
        id=row["id"],
        owner_id=row["owner_id"],
        status=SessionStatus(row["status"]),
        preferred_language=Language(row["preferred_language"]),
        title=row["title"],
        title_source=TitleSource(row["title_source"]) if row.get("title_source") else None,
        title_status=row.get("title_status") or "pending",
        title_updated_at=utc(row.get("title_updated_at")),
        active_analysis_id=row["active_analysis_id"],
        context_epoch=row["context_epoch"],
        event_sequence=row["event_sequence"],
        created_at=utc(row["created_at"]),
        updated_at=utc(row["updated_at"]),
        version=row["version"],
    )


# --- message ---------------------------------------------------------------

def message_to_row(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "client_message_id": message.client_message_id,
        "role": message.role.value,
        "sequence": message.sequence,
        "created_at": message.created_at,
    }


def part_to_row(part: MessagePart) -> dict[str, Any]:
    return {
        "id": part.id,
        "message_id": part.message_id,
        "index": part.index,
        "type": part.type.value,
        "content": part.content,
        "version": part.version,
    }


def row_to_message(row: Mapping[str, Any], part_rows: list[Mapping[str, Any]]) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=Role(row["role"]),
        sequence=row["sequence"],
        created_at=utc(row["created_at"]),
        client_message_id=row["client_message_id"],
        parts=tuple(
            MessagePart(
                id=p["id"],
                message_id=p["message_id"],
                index=p["index"],
                type=PartType(p["type"]),
                content=p["content"],
                version=p["version"],
            )
            for p in sorted(part_rows, key=lambda p: p["index"])
        ),
    )


# --- run -------------------------------------------------------------------

def run_to_row(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "session_id": run.session_id,
        "trigger_message_id": run.trigger_message_id,
        "lane": run.lane.value,
        "intent": run.intent.value,
        "status": run.status.value,
        "runtime_binding_id": run.runtime_binding_id,
        "recovery_of_run_id": run.recovery_of_run_id,
        "deadline_at": run.deadline_at,
        "failure_code": run.failure_code,
        "potentially_billed": run.potentially_billed,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "version": run.version,
    }


def row_to_run(row: Mapping[str, Any]) -> Run:
    return Run(
        id=row["id"],
        session_id=row["session_id"],
        trigger_message_id=row["trigger_message_id"],
        lane=Lane(row["lane"]),
        intent=Intent(row["intent"]),
        status=RunStatus(row["status"]),
        deadline_at=utc(row["deadline_at"]),
        created_at=utc(row["created_at"]),
        started_at=utc(row["started_at"]),
        ended_at=utc(row["ended_at"]),
        runtime_binding_id=row["runtime_binding_id"],
        failure_code=row["failure_code"],
        recovery_of_run_id=row["recovery_of_run_id"],
        potentially_billed=bool(row["potentially_billed"]),
        version=row["version"],
    )


# --- analysis --------------------------------------------------------------

def analysis_to_row(snapshot: AnalysisSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "session_id": snapshot.session_id,
        "run_id": snapshot.run_id,
        "input_smiles": snapshot.input_smiles,
        "canonical_smiles": snapshot.canonical_smiles,
        "requested_endpoints": list(snapshot.requested_endpoints),
        "predictor_response": snapshot.predictor_response,
        "predictor_base_url_id": snapshot.provenance.base_url_id,
        "predictor_service_version": snapshot.provenance.service_version,
        "predictor_git_commit": snapshot.provenance.git_commit,
        "artifact_hashes": list(snapshot.provenance.artifact_hashes),
        "policy_snapshot": snapshot.policy_snapshot,
        "content_sha256": snapshot.content_sha256,
        "idempotency_key": snapshot.idempotency_key,
        "created_at": snapshot.created_at,
    }


def row_to_analysis(row: Mapping[str, Any]) -> AnalysisSnapshot:
    return AnalysisSnapshot(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        input_smiles=row["input_smiles"],
        canonical_smiles=row["canonical_smiles"],
        requested_endpoints=tuple(row["requested_endpoints"]),
        predictor_response=row["predictor_response"],
        provenance=PredictorProvenance(
            base_url_id=row["predictor_base_url_id"],
            service_version=row["predictor_service_version"],
            git_commit=row["predictor_git_commit"],
            artifact_hashes=tuple(row["artifact_hashes"]),
            raw=row["predictor_response"].get("provenance", {}),
        ),
        policy_snapshot=row["policy_snapshot"],
        content_sha256=row["content_sha256"],
        idempotency_key=row["idempotency_key"],
        created_at=utc(row["created_at"]),
    )


# --- observation -----------------------------------------------------------

def observation_to_row(obs: Observation, analysis_id: str | None) -> dict[str, Any]:
    return {
        "id": obs.id,
        "session_id": obs.session_id,
        "run_id": obs.run_id,
        "analysis_id": analysis_id,
        "producer": obs.producer.value,
        "kind": obs.kind.value,
        "schema_version": obs.schema_version,
        "canonical_payload": obs.canonical_payload,
        "model_projection": obs.model_projection,
        "projection_version": obs.projection_version,
        "required_limitations": list(obs.required_limitations),
        "provenance": obs.provenance,
        "content_sha256": obs.content_sha256,
        "created_at": obs.created_at,
    }


def row_to_observation(row: Mapping[str, Any]) -> Observation:
    return Observation(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        producer=Producer(row["producer"]),
        kind=ObservationKind(row["kind"]),
        schema_version=row["schema_version"],
        canonical_payload=row["canonical_payload"],
        model_projection=row["model_projection"],
        provenance=row["provenance"],
        content_sha256=row["content_sha256"],
        created_at=utc(row["created_at"]),
        projection_version=row["projection_version"],
        required_limitations=tuple(row["required_limitations"]),
    )


# --- evidence --------------------------------------------------------------

def evidence_to_row(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "session_id": record.session_id,
        "provider": record.provider,
        "provider_record_id": record.provider_record_id,
        "source_type": record.source_type.value,
        "title": record.title,
        "authors": list(record.authors),
        "published_at": record.published_at.isoformat() if record.published_at else None,
        "retrieved_at": record.retrieved_at,
        "canonical_url": record.canonical_url,
        "identifier": record.identifier.to_dict(),
        "dedupe_key": record.dedupe_key,
        "abstract_or_excerpt": record.abstract_or_excerpt,
        "normalized_facts": record.normalized_facts,
        "source_quality_tier": record.source_quality_tier.value,
        "raw_payload_ref": record.raw_payload_ref,
        "status": record.status.value,
        "rejection_reason": record.rejection_reason,
        "relevance_assessment": dict(record.relevance_assessment) or None,
        "content_sha256": record.content_sha256,
    }


def row_to_evidence(row: Mapping[str, Any]) -> EvidenceRecord:
    ident = row["identifier"] or {}
    return EvidenceRecord(
        id=row["id"],
        session_id=row["session_id"],
        provider=row["provider"],
        provider_record_id=row["provider_record_id"],
        source_type=SourceType(row["source_type"]),
        title=row["title"],
        retrieved_at=utc(row["retrieved_at"]),
        status=EvidenceStatus(row["status"]),
        content_sha256=row["content_sha256"],
        authors=tuple(row["authors"]),
        published_at=_date(row["published_at"]),
        canonical_url=row["canonical_url"],
        identifier=SourceIdentifier(
            doi=ident.get("doi"), pmid=ident.get("pmid"), pmcid=ident.get("pmcid"),
            cid=ident.get("cid"), other=ident.get("other"),
        ),
        abstract_or_excerpt=row["abstract_or_excerpt"],
        normalized_facts=row["normalized_facts"],
        source_quality_tier=SourceQualityTier(row["source_quality_tier"]),
        raw_payload_ref=row["raw_payload_ref"],
        rejection_reason=row["rejection_reason"],
        relevance_assessment=dict(row.get("relevance_assessment") or {}),
    )


# --- answer ----------------------------------------------------------------

def answer_to_row(answer: GroundedAnswer) -> dict[str, Any]:
    return {
        "id": answer.id,
        "session_id": answer.session_id,
        "run_id": answer.run_id,
        "schema_version": answer.schema_version,
        "answer_markdown": answer.answer_markdown,
        "limitations": [l.to_dict() for l in answer.limitations],
        "recommended_next_steps": [s.to_dict() for s in answer.recommended_next_steps],
        "candidate_generation": answer.candidate_generation,
        "is_fallback": answer.is_fallback,
        "content_sha256": answer.content_sha256,
        "created_at": answer.created_at,
    }


def claim_to_row(claim: Claim, answer_id: str, position: int) -> dict[str, Any]:
    return {
        "id": claim.claim_id,
        "answer_id": answer_id,
        "kind": claim.kind.value,
        "text": claim.text,
        "observation_id": claim.observation_id,
        "field_path": claim.field_path,
        "source_value": {"v": claim.source_value},
        "rendered_value": claim.rendered_value,
        "transform": claim.transform,
        "input_claim_ids": list(claim.input_claim_ids),
        "position": position,
    }


def row_to_claim(row: Mapping[str, Any], citation_ids: tuple[str, ...] = ()) -> Claim:
    return Claim(
        claim_id=row["id"],
        kind=ClaimKind(row["kind"]),
        text=row["text"],
        observation_id=row["observation_id"],
        field_path=row["field_path"],
        source_value=(row["source_value"] or {}).get("v"),
        rendered_value=row["rendered_value"],
        transform=row["transform"],
        citation_ids=citation_ids,
        input_claim_ids=tuple(row["input_claim_ids"]),
    )


def row_to_answer(row: Mapping[str, Any], claims: tuple[Claim, ...]) -> GroundedAnswer:
    return GroundedAnswer(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        answer_markdown=row["answer_markdown"],
        claims=claims,
        limitations=tuple(
            Limitation(code=LimitationCode(l["code"]), text=l["text"]) for l in row["limitations"]
        ),
        recommended_next_steps=tuple(
            RecommendedNextStep(text=s["text"], basis_claim_ids=tuple(s["basis_claim_ids"]))
            for s in row["recommended_next_steps"]
        ),
        candidate_generation=row["candidate_generation"],
        content_sha256=row["content_sha256"],
        created_at=utc(row["created_at"]),
        schema_version=row["schema_version"],
        is_fallback=bool(row["is_fallback"]),
    )


# --- runtime binding -------------------------------------------------------

def binding_to_row(binding: RuntimeBinding) -> dict[str, Any]:
    return {
        "id": binding.id,
        "session_id": binding.session_id,
        "runtime_kind": binding.runtime_kind.value,
        "runtime_version": binding.runtime_version,
        "runtime_session_id": binding.runtime_session_id,
        "provider_id": binding.provider_id,
        "model_id": binding.model_id,
        "auth_mode": binding.auth_mode.value,
        "connection_id": binding.connection_id,
        "profile_hash": binding.profile_hash,
        "tool_schema_hash": binding.tool_schema_hash,
        "system_prompt_hash": binding.system_prompt_hash,
        "capabilities": binding.capabilities.to_dict(),
        "status": binding.status.value,
        "selection_reason": binding.selection_reason,
        "runtime_manifest": dict(binding.runtime_manifest) or None,
        "created_at": binding.created_at,
        "closed_at": binding.closed_at,
    }


def row_to_binding(row: Mapping[str, Any]) -> RuntimeBinding:
    caps = row["capabilities"]
    return RuntimeBinding(
        id=row["id"],
        session_id=row["session_id"],
        runtime_kind=RuntimeKind(row["runtime_kind"]),
        runtime_version=row["runtime_version"],
        runtime_session_id=row["runtime_session_id"],
        provider_id=row["provider_id"],
        model_id=row["model_id"],
        auth_mode=AuthMode(row.get("auth_mode") or "none"),
        connection_id=row.get("connection_id"),
        profile_hash=row["profile_hash"],
        tool_schema_hash=row["tool_schema_hash"],
        system_prompt_hash=row["system_prompt_hash"],
        capabilities=RuntimeCapabilities(
            streaming=caps["streaming"], resume=caps["resume"], cancel_turn=caps["cancel_turn"],
            close_session=caps["close_session"], mcp_streamable_http=caps["mcp_streamable_http"],
            native_structured_output=caps["native_structured_output"],
            usage=tuple(caps["usage"]), attachments=tuple(caps["attachments"]),
        ),
        status=BindingStatus(row["status"]),
        created_at=utc(row["created_at"]),
        closed_at=utc(row["closed_at"]),
        selection_reason=row["selection_reason"],
        runtime_manifest=dict(row.get("runtime_manifest") or {}),
    )


# --- runtime usage ---------------------------------------------------------

def usage_to_row(event: RuntimeUsageEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "session_id": event.session_id,
        "run_id": event.run_id,
        "runtime_binding_id": event.runtime_binding_id,
        "provider_id": event.provider_id,
        "model_id": event.model_id,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "reasoning_tokens": event.reasoning_tokens,
        "cache_read_tokens": event.cache_read_tokens,
        "cache_write_tokens": event.cache_write_tokens,
        "total_tokens": event.total_tokens,
        "cost_amount": event.cost_amount,
        "cost_currency": event.cost_currency,
        "reported_at": event.reported_at,
        "source_event_id": event.source_event_id,
        "source_event_type": event.source_event_type,
        "provider_message_id": event.provider_message_id,
        "provider_step_id": event.provider_step_id,
        "revision": event.revision,
        "semantics": event.semantics,
        "is_normalized": event.is_normalized,
        "raw_payload_hash": event.raw_payload_hash,
    }


def row_to_usage(row: Mapping[str, Any]) -> RuntimeUsageEvent:
    amount = row["cost_amount"]
    return RuntimeUsageEvent(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        runtime_binding_id=row["runtime_binding_id"],
        provider_id=row["provider_id"],
        model_id=row["model_id"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        reasoning_tokens=row["reasoning_tokens"],
        cache_read_tokens=row["cache_read_tokens"],
        cache_write_tokens=row["cache_write_tokens"],
        total_tokens=row["total_tokens"],
        cost_amount=Decimal(str(amount)) if amount is not None else None,
        cost_currency=row["cost_currency"],
        reported_at=utc(row["reported_at"]),
        source_event_id=row.get("source_event_id"),
        source_event_type=row.get("source_event_type"),
        provider_message_id=row.get("provider_message_id"),
        provider_step_id=row.get("provider_step_id"),
        revision=row.get("revision"),
        semantics=row.get("semantics") or "unknown",
        is_normalized=bool(row.get("is_normalized")),
        raw_payload_hash=row.get("raw_payload_hash"),
    )


# --- attachment and events -------------------------------------------------

def attachment_to_row(attachment: Attachment) -> dict[str, Any]:
    return {
        "id": attachment.id,
        "owner_id": attachment.owner_id,
        "session_id": attachment.session_id,
        "media_type": attachment.media_type,
        "object_uri": attachment.object_uri,
        "sha256": attachment.sha256,
        "size_bytes": attachment.size_bytes,
        "retention_class": attachment.retention_class.value,
        "created_at": attachment.created_at,
        "expires_at": attachment.expires_at,
    }


def row_to_attachment(row: Mapping[str, Any]) -> Attachment:
    return Attachment(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        media_type=row["media_type"],
        object_uri=row["object_uri"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        retention_class=RetentionClass(row["retention_class"]),
        created_at=utc(row["created_at"]),
        expires_at=utc(row["expires_at"]),
    )


def row_to_event(row: Mapping[str, Any]) -> Event:
    return Event(
        event_id=row["event_id"],
        session_id=row["session_id"],
        sequence=row["sequence"],
        type=EventType(row["type"]),
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        entity_version=row["entity_version"],
        run_id=row["run_id"],
        occurred_at=utc(row["occurred_at"]),
        payload=row["payload"],
    )


# --- report builder ---------------------------------------------------------


def report_build_to_row(build: ReportBuild) -> dict[str, Any]:
    return {
        "id": build.id,
        "session_id": build.session_id,
        "run_id": build.run_id,
        "analysis_id": build.analysis_id,
        "request": build.request.to_dict(),
        "stage": build.stage.value,
        "report_id": build.report_id,
        "correction_attempts": build.correction_attempts,
        "failure_code": build.failure_code,
        "failure_detail": build.failure_detail,
        "stage_state": dict(build.stage_state),
        "deadline_at": build.deadline_at,
        "created_at": build.created_at,
        "updated_at": build.updated_at,
    }


def row_to_report_build(row: Mapping[str, Any]) -> ReportBuild:
    return ReportBuild(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        analysis_id=row["analysis_id"],
        request=ReportBuildRequest.from_dict(dict(row["request"])),
        stage=BuildStage(row["stage"]),
        created_at=utc(row["created_at"]),
        updated_at=utc(row["updated_at"]),
        deadline_at=utc(row["deadline_at"]),
        report_id=row["report_id"],
        correction_attempts=row["correction_attempts"] or 0,
        failure_code=row["failure_code"],
        failure_detail=row["failure_detail"],
        stage_state=dict(row["stage_state"] or {}),
    )


# --- evidence relations (ADS plan section 9.3, ADR 0010) -------------------

def evidence_relation_to_row(assessment: EvidenceRelationAssessment) -> dict[str, Any]:
    return {
        "id": assessment.id,
        "session_id": assessment.session_id,
        "run_id": assessment.run_id,
        "proposition_id": assessment.proposition_id,
        "source_class": assessment.source_ref.source_class.value,
        "source_id": assessment.source_ref.source_id,
        "relation": assessment.relation.value,
        "directness": assessment.directness.value,
        "applicability": assessment.applicability.value,
        "strength": assessment.strength.value,
        "reason_codes": list(assessment.reason_codes),
        "scope": assessment.scope.to_dict(),
        "input_refs": list(assessment.input_refs),
        "assessor": assessment.assessor,
        "method_version": assessment.method_version,
        "created_at": assessment.created_at,
    }


def row_to_evidence_relation(row: Mapping[str, Any]) -> EvidenceRelationAssessment:
    scope = dict(row["scope"] or {})
    return EvidenceRelationAssessment(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        proposition_id=row["proposition_id"],
        source_ref=SourceRef(
            source_class=SourceClass(row["source_class"]), source_id=row["source_id"],
        ),
        relation=RelationLabel(row["relation"]),
        directness=Directness(row["directness"]),
        applicability=Applicability(row["applicability"]),
        strength=Strength(row["strength"]),
        reason_codes=tuple(row["reason_codes"] or ()),
        scope=EvidenceScope(
            endpoint=scope.get("endpoint"), species=scope.get("species"),
            dose=scope.get("dose"), use_context=scope.get("use_context"),
        ),
        created_at=utc(row["created_at"]),
        input_refs=tuple(row.get("input_refs") or ()),
        assessor=row.get("assessor") or "model",
        method_version=row.get("method_version") or "grounded-answer-v2",
    )


def development_posture_to_row(
    posture: DevelopmentPosture, *, answer_id: str, session_id: str, run_id: str, created_at,
) -> dict[str, Any]:
    return {
        "answer_id": answer_id,
        "session_id": session_id,
        "run_id": run_id,
        "value": posture.value.value,
        "scope": posture.scope.value,
        "confidence_band": posture.confidence_band,
        "basis_claim_ids": list(posture.basis_claim_ids),
        "contrary_claim_ids": list(posture.contrary_claim_ids),
        "rationale": posture.rationale,
        "conditions": list(posture.conditions),
        "recommended_next_steps": list(posture.recommended_next_steps),
        "created_at": created_at,
    }


def row_to_development_posture(row: Mapping[str, Any]) -> DevelopmentPosture:
    return DevelopmentPosture(
        value=PostureValue(row["value"]),
        scope=DevelopmentScope(row["scope"]),
        confidence_band=row["confidence_band"],
        basis_claim_ids=tuple(row["basis_claim_ids"] or ()),
        contrary_claim_ids=tuple(row["contrary_claim_ids"] or ()),
        rationale=row["rationale"],
        conditions=tuple(row["conditions"] or ()),
        recommended_next_steps=tuple(row["recommended_next_steps"] or ()),
    )
