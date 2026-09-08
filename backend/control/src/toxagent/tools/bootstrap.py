"""Assemble the tool registry for a deployment.

Which tools exist is a deployment fact — an evidence provider that is not
configured means those two tools are simply not registered, and the model is
never shown a capability the server cannot honour.
"""
from __future__ import annotations

from ..application.create_analysis import CreateAnalysis
from ..config import PolicySettings, ResearchSettings
from ..predictor.client import PredictorClient
from ..research.interfaces import ResearchProvider
from .definitions import analysis as analysis_tools
from .definitions import answer as answer_tools
from .definitions import evidence as evidence_tools
from .registry import ToolRegistry


def build_registry(
    database,
    predictor: PredictorClient,
    create_analysis: CreateAnalysis,
    settings: PolicySettings | None = None,
    *,
    research_provider: ResearchProvider | None = None,
    research_settings: ResearchSettings | None = None,
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
    for definition in extra or []:
        registry.register(definition)
    return registry
