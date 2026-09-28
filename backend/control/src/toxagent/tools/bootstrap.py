"""Assemble the tool registry for a deployment.

Which tools exist is a deployment fact — an evidence provider that is not
configured means those two tools are simply not registered, and the model is
never shown a capability the server cannot honour.
"""
from __future__ import annotations

from ..application.prediction.create_analysis import CreateAnalysis
from ..config import PolicySettings, ResearchSettings
from ..predictor.client import PredictorClient
from ..research.compound import CompoundProvider
from ..research.interfaces import ResearchProvider
from .definitions import analysis as analysis_tools
from .definitions import answer as answer_tools
from .definitions import claim_review as claim_review_tools
from .definitions import compound as compound_tools
from .definitions import decision_plan as decision_plan_tools
from .definitions import evidence as evidence_tools
from .definitions import explanation as explanation_tools
from .definitions import inventory as inventory_tools
from .definitions import primitives as primitive_tools
from .definitions import report as report_tools
from .definitions import report_synthesis as report_synthesis_tools
from .definitions import scientific_case as scientific_case_tools
from .definitions import scientific_skills as scientific_skill_tools
from .definitions import skill_drafts as skill_draft_tools
from .registry import FLAG_GATED_TOOLS, ToolDefinition, ToolRegistry
from ..flags import is_enabled


def _gated_off(definition: ToolDefinition) -> bool:
    flag_name = FLAG_GATED_TOOLS.get(definition.name)
    return flag_name is not None and not is_enabled(flag_name)


def build_registry(
    database,
    predictor: PredictorClient,
    create_analysis: CreateAnalysis,
    settings: PolicySettings | None = None,
    *,
    research_provider: ResearchProvider | None = None,
    research_settings: ResearchSettings | None = None,
    compound_provider: CompoundProvider | None = None,
    object_store=None,
    skill_catalog=None,
    chembl_provider=None,
    extra: list | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    def add(definition: ToolDefinition) -> None:
        # A flag-gated tool with its flag off is not registered at all, so it
        # is absent from tools/list and from the profile's schema hash.
        if not _gated_off(definition):
            registry.register(definition)

    for definition in analysis_tools.build(database, predictor, create_analysis):
        add(definition)
    for definition in answer_tools.build(database, settings or PolicySettings()):
        add(definition)
    # Registered regardless of research_provider — get_artifact_inventory's
    # available_tools.evidence_search flag is how a run learns search is
    # absent, rather than the tool itself vanishing (ADS plan W2-03; unlike
    # evidence_tools below, whose absence *is* the signal for report_qa/
    # evidence_research/report_build).
    for definition in inventory_tools.build(
        database, research_provider_configured=research_provider is not None
    ):
        add(definition)
    if research_provider is not None:
        for definition in evidence_tools.build(
            database, research_provider, research_settings or ResearchSettings()
        ):
            add(definition)
    # The report-builder surface. The explanation and report tools need no
    # provider, so they exist wherever a predictor does; ``resolve_compound_record``
    # follows the same rule as the evidence tools — an unconfigured substance
    # provider means the tool is simply absent, and a report records an
    # identity gap rather than calling something that could only fail.
    for definition in explanation_tools.build(database, predictor, object_store):
        add(definition)
    for definition in report_tools.build(database, object_store, predictor):
        add(definition)
    # Visible only under ``report_synthesis``, the profile an orchestrated
    # build dispatches; registering it unconditionally changes no other
    # profile's tool list or schema hash.
    for definition in report_synthesis_tools.build(database):
        add(definition)
    # Gated by decision_state_plan_tool / scientific_case_v1 through FLAG_GATED_TOOLS.
    for definition in decision_plan_tools.build(database):
        add(definition)
    for definition in scientific_case_tools.build(database):
        add(definition)
    if skill_catalog is None:
        from ..config import PACKAGE_ROOT
        from ..application.investigation.skill_catalog import load_catalog

        skill_catalog = load_catalog(PACKAGE_ROOT / "agent_profiles")
    # Gated by scientific_skills_v1. Visibility is read at call time, from the
    # finished registry: a skill is readable only if its required tools are.
    for definition in scientific_skill_tools.build(
        database, skill_catalog,
        lambda profile: [tool.name for tool in registry.visible_for(profile)],
    ):
        add(definition)
    # Gated by scientific_primitives_v1 (W9-13); the ChEMBL tool also needs a
    # configured provider, like the evidence tools.
    for definition in primitive_tools.build(database, chembl_provider):
        add(definition)
    # Gated by claim_reviewer_v1 (W9-12); visible only under claim_review.
    for definition in claim_review_tools.build(database):
        add(definition)
    # Gated by skill_drafts_v1 (W9-11): a proposal for expert review, never a
    # change to the catalog this or any run is offered.
    for definition in skill_draft_tools.build(database, skill_catalog):
        add(definition)
    if compound_provider is not None:
        for definition in compound_tools.build(database, compound_provider):
            add(definition)
    for definition in extra or []:
        registry.register(definition)
    return registry
