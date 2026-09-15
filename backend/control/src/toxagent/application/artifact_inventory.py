"""Artifact inventory for decision support (ADS plan section 8.1, W2-03/W3-01).

One compact, server-authored projection of everything a ``decision_support``
run could read without spending budget on it: the active analysis, the latest
non-superseded report for it, which explanations already exist, which evidence
is already accepted, and which of the profile's tools are actually usable in
this deployment. Pointer and planning metadata only — a numeric value still has
to come from ``get_analysis_slice``, evidence content still has to come from
``get_evidence_record``, report detail still has to come from
``get_report_summary``. Shared between the ``get_artifact_inventory`` tool
(``tools/definitions/inventory.py``) and context pinning
(``harness/gateway.py``) so the two views of "what already exists" cannot
drift apart.
"""
from __future__ import annotations

from typing import Any

from ..domain.evidence import EvidenceStatus
from ..domain.observation import ObservationKind

SCHEMA_VERSION = "artifact-inventory-v1"


async def build_artifact_inventory(
    uow,
    *,
    session_id: str,
    analysis_id: str | None,
    research_provider_configured: bool,
) -> dict[str, Any]:
    """Assemble the inventory for one analysis (or an empty one, if the
    session has none active yet — a decision-support run can still be asked a
    question with no analysis resolved, e.g. "what can this tool do").
    """
    analysis_view: dict[str, Any] | None = None
    latest_report_view: dict[str, Any] | None = None
    explanations: list[dict[str, Any]] = []

    if analysis_id:
        snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
        if snapshot is not None:
            analysis_view = {
                "analysis_id": snapshot.id,
                "canonical_smiles": snapshot.canonical_smiles,
                "served_endpoints": list(snapshot.served_endpoints),
            }
            observations = await uow.observations.list_for_analysis(snapshot.id)
            for obs in observations:
                if obs.kind is not ObservationKind.ATTRIBUTION:
                    continue
                explanations.append(
                    {
                        "endpoint": obs.model_projection.get("endpoint"),
                        "task": obs.model_projection.get("task"),
                        "observation_id": obs.id,
                        "status": obs.model_projection.get("status"),
                    }
                )

            document = await uow.reports.get_latest_artifact_for_analysis(
                snapshot.id, session_id=session_id
            )
            if document is not None:
                gaps = document.get("gaps") or []
                recommendations = document.get("recommendations") or []
                latest_report_view = {
                    "report_id": document.get("report_id") or document.get("id"),
                    "version": document.get("version", 1),
                    "status": document.get("status"),
                    "gap_summaries": [g.get("detail") for g in gaps],
                    "recommendation_summaries": [r.get("text") for r in recommendations],
                }

    accepted_evidence_records = await uow.evidence.list_for_session(
        session_id, status=EvidenceStatus.ACCEPTED, limit=5
    )
    accepted_evidence = [
        {
            "evidence_id": record.id,
            "title": record.title,
            "source_type": record.source_type.value
            if hasattr(record.source_type, "value")
            else record.source_type,
            "retrieved_at": record.retrieved_at.isoformat(),
        }
        for record in accepted_evidence_records
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": analysis_view,
        "latest_report": latest_report_view,
        "explanations": explanations,
        "accepted_evidence": accepted_evidence,
        "available_tools": {
            "evidence_search": research_provider_configured,
            "attribution_generation": True,
            "regulatory_search": False,
        },
    }
