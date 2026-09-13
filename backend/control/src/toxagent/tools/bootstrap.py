"""Assemble the tool registry for a deployment.

Which tools exist is a deployment fact — an evidence provider that is not
configured means those two tools are simply not registered, and the model is
never shown a capability the server cannot honour.
"""
from __future__ import annotations

from ..application.create_analysis import CreateAnalysis
from ..config import PolicySettings, ResearchSettings
from ..predictor.client import PredictorClient
from ..research.compound import CompoundProvider
from ..research.interfaces import ResearchProvider
from .definitions import analysis as analysis_tools
from .definitions import answer as answer_tools
from .definitions import compound as compound_tools
from .definitions import evidence as evidence_tools
from .definitions import explanation as explanation_tools
from .definitions import report as report_tools
from .definitions import report_synthesis as report_synthesis_tools
from .registry import ToolRegistry


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
    extra: list | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for definition in analysis_tools.build(database, predictor, create_analysis):
        registry.register(definition)
    for definition in answer_tools.build(database, settings or PolicySettings()):
        registry.register(definition)
    if research_provider is not None:
        for definition in evidence_tools.build(
            database, research_provider, research_settings or ResearchSettings()
        ):
            registry.register(definition)
    # The report-builder surface. The explanation and report tools need no
    # provider, so they exist wherever a predictor does; ``resolve_compound_record``
    # follows the same rule as the evidence tools — an unconfigured substance
    # provider means the tool is simply absent, and a report records an
    # identity gap rather than calling something that could only fail.
    for definition in explanation_tools.build(database, predictor, object_store):
        registry.register(definition)
    for definition in report_tools.build(database, object_store, predictor):
        registry.register(definition)
    # Visible only under ``report_synthesis``, the profile an orchestrated
    # build dispatches; registering it unconditionally changes no other
    # profile's tool list or schema hash.
    for definition in report_synthesis_tools.build(database):
        registry.register(definition)
    if compound_provider is not None:
        for definition in compound_tools.build(database, compound_provider):
            registry.register(definition)
    for definition in extra or []:
        registry.register(definition)
    return registry
