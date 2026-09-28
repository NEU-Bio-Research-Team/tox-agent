"""Artifact inventory and report-summary tools for decision support (ADS plan
section 8.1, W2-03/W2-04).

``get_artifact_inventory`` answers "what already exists" without spending
budget reading any of it — the same projection pinned into context by
``harness/gateway.py`` (``application/investigation/artifact_inventory.py`` is the one place
that assembly happens). ``get_report_summary`` is the bounded read this
inventory's ``latest_report`` pointer promises is actually readable: status,
gaps and recommendations only, never the full sections/tables/figures a report
build tool would return.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation.artifact_inventory import build_artifact_inventory
from ...domain.errors import InvalidRequest
from ..registry import ToolContext, ToolDefinition, ToolOutput


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactInventoryInput(_Input):
    analysis_id: str | None = Field(
        default=None,
        description="Defaults to this run's active analysis. Naming another session's "
                    "analysis resolves to nothing.",
    )


class ReportSummaryInput(_Input):
    report_id: str | None = Field(
        default=None,
        description="Defaults to the latest report for this run's active analysis. "
                    "Give an explicit id only to read an older, superseded version.",
    )
    analysis_id: str | None = Field(
        default=None,
        description="Only used with report_id omitted, to pick which analysis's latest "
                    "report to summarize when it differs from the run's active one.",
    )


def build(database, *, research_provider_configured: bool) -> list[ToolDefinition]:
    async def artifact_inventory(context: ToolContext, payload: ArtifactInventoryInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            session = await uow.sessions.get_unscoped(context.session_id)
            analysis_id = payload.analysis_id or (
                session.active_analysis_id if session else None
            )
            view = await build_artifact_inventory(
                uow,
                session_id=context.session_id,
                analysis_id=analysis_id,
                research_provider_configured=research_provider_configured,
            )
        return ToolOutput(canonical=view, model_view=view, ui_view=view)

    async def report_summary(context: ToolContext, payload: ReportSummaryInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            if payload.report_id:
                document = await uow.reports.get_artifact(
                    payload.report_id, session_id=context.session_id
                )
            else:
                session = await uow.sessions.get_unscoped(context.session_id)
                analysis_id = payload.analysis_id or (
                    session.active_analysis_id if session else None
                )
                if not analysis_id:
                    raise InvalidRequest(
                        "no analysis_id given and this run has no active analysis to "
                        "summarize a report for"
                    )
                document = await uow.reports.get_latest_artifact_for_analysis(
                    analysis_id, session_id=context.session_id
                )
        if document is None:
            raise InvalidRequest("no such report in this session, or this analysis has no report yet")

        gaps = document.get("gaps") or []
        recommendations = document.get("recommendations") or []
        synthesis = document.get("evidence_synthesis") or []
        view: dict[str, Any] = {
            "report_id": document.get("report_id") or document.get("id"),
            "analysis_id": document.get("analysis_id"),
            "version": document.get("version", 1),
            "supersedes_report_id": document.get("supersedes_report_id"),
            "status": document.get("status"),
            "gaps": [
                {
                    "reason": g.get("reason"),
                    "detail": g.get("detail"),
                    "endpoint": g.get("endpoint"),
                    "task": g.get("task"),
                }
                for g in gaps
            ],
            "recommendations": [
                {
                    "text": r.get("text"),
                    "action_category": r.get("action_category"),
                    "priority": r.get("priority"),
                    "conditions": r.get("conditions"),
                }
                for r in recommendations
            ],
            "evidence_synthesis": [
                {
                    "proposition": s.get("proposition"),
                    "relation": s.get("relation"),
                    "endpoint": s.get("endpoint"),
                }
                for s in synthesis
            ],
        }
        return ToolOutput(canonical=view, model_view=view, ui_view=view)

    return [
        ToolDefinition(
            name="get_artifact_inventory",
            title="List what already exists for this analysis",
            description=(
                "Return a compact pointer summary of the active analysis, its latest report "
                "(status, gap and recommendation summaries), which explanations already "
                "exist, which evidence is already accepted, and which of this profile's "
                "tools are actually usable in this deployment. Call this first, before "
                "searching or re-attributing: it exists so you do not have to guess from "
                "prose whether something was already computed. Values still have to be read "
                "with the tool named for them (get_analysis_slice, get_report_summary, "
                "get_evidence_record, get_explanation_slice) — this is pointers only."
            ),
            input_model=ArtifactInventoryInput,
            handler=artifact_inventory,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=8.0,
            cost_class="cheap",
        ),
        ToolDefinition(
            name="get_report_summary",
            title="Read a report's status, gaps and recommendations",
            description=(
                "Return one report's status, gap summaries (what it could not establish and "
                "why) and recommendation summaries, bounded — never the full sections, "
                "tables or figures. A gap is not evidence against the missing claim; it is a "
                "record that nobody established it yet. Defaults to the latest report for "
                "the active analysis."
            ),
            input_model=ReportSummaryInput,
            handler=report_summary,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=8.0,
            cost_class="cheap",
        ),
    ]
