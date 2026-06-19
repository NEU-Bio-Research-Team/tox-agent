"""tool_selection_agent.py — ToxAgent ToolSelectionAgent

Fine-grained tool router: given a user request and the plan complexity
decided by PlanningAgent, determines the minimum necessary set of tools
to activate to answer the request.

Architecture decisions (approved):
  Q1 — Gate Execution: ToolManifest actively gates molrag_enabled,
         explain_enabled and other optional runtime parameters.
         Also stored in state["tool_manifest"] for logging + UI.
         First v1 component with direct runtime impact.
  Q2 — After PlanningAgent, Before Execution:
         Uses plan.complexity as the primary input signal.
  Q3 — Co-activate MolRAG + Firestore:
         Either selection → molrag_enabled=True.

Abstract → Concrete tool mapping:
  ScreeningModel      → analyze_molecule()
  ExplainabilityEngine → analyze_molecule() explanation + xsmiles backend
  PubChem             → get_compound_info_pubchem() + get_pubchem_bioassay_data()
  PubMed              → search_toxicity_literature() + synthesize_literature()
  MolRAG              → retrieve_similar_molecules() + run_molrag_reasoning()
  Firestore           → MolRAG Firestore retrieval (co-activates with MolRAG)
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, model_validator

from .adk_compat import LlmAgent

logger = logging.getLogger(__name__)

TOOL_SELECTION_MODEL = os.getenv(
    "AGENT_MODEL_FAST",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolName(str, Enum):
    """Registry of abstract tools available to ToxAgent.

    Each value maps to one or more concrete Python functions in tools/.
    """

    PubChem             = "PubChem"
    PubMed              = "PubMed"
    MolRAG              = "MolRAG"
    Firestore           = "Firestore"
    ScreeningModel      = "ScreeningModel"
    ExplainabilityEngine = "ExplainabilityEngine"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ToolSelection(BaseModel):
    """A single tool choice with its justification."""

    tool: ToolName
    reason: str = Field(default="", min_length=0)


class ToolManifest(BaseModel):
    """The complete set of tools selected for a given user request.

    Fields
    ------
    selected_tools    : Ordered list of ToolSelection entries.
    molrag_enabled    : True when MolRAG or Firestore is selected.
                        Gates molrag_enabled in run_orchestrator_from_text().
    explain_enabled   : True when ExplainabilityEngine is selected.
                        Gates structural attribution in the screening call.
    pubmed_enabled    : True when PubMed is selected.
    pubchem_enabled   : True when PubChem is selected.
    selection_source  : 'deterministic' | 'llm' | 'llm_fallback'.
    """

    selected_tools: List[ToolSelection] = Field(default_factory=list)
    molrag_enabled: bool = Field(default=False)
    explain_enabled: bool = Field(default=False)
    pubmed_enabled: bool = Field(default=False)
    pubchem_enabled: bool = Field(default=False)
    selection_source: str = Field(default="deterministic")

    @model_validator(mode="after")
    def _derive_capability_flags(self) -> "ToolManifest":
        """Auto-derive boolean gates from the selected_tools list."""
        tools: Set[ToolName] = {s.tool for s in self.selected_tools}

        # MolRAG and Firestore co-activate (Q3 decision)
        object.__setattr__(
            self,
            "molrag_enabled",
            ToolName.MolRAG in tools or ToolName.Firestore in tools,
        )
        object.__setattr__(
            self,
            "explain_enabled",
            ToolName.ExplainabilityEngine in tools,
        )
        object.__setattr__(self, "pubmed_enabled", ToolName.PubMed in tools)
        object.__setattr__(self, "pubchem_enabled", ToolName.PubChem in tools)
        return self

    def tool_names(self) -> List[str]:
        """Return a list of selected tool name strings for logging."""
        return [s.tool.value for s in self.selected_tools]


# ---------------------------------------------------------------------------
# Keyword patterns for deterministic selection
# ---------------------------------------------------------------------------

_EXPLAIN_PATTERNS = re.compile(
    r"\b(explain|why|how|mechanism|mechanis[mt]|attribution|attention|"
    r"heatmap|structural basis|mode of action|moa|pathway)\b",
    re.IGNORECASE,
)

_ANALOG_PATTERNS = re.compile(
    r"\b(analog|analogue|similar|compare|comparison|structurally related|"
    r"scaffold|nearest neighbor|nearest neighbour)\b",
    re.IGNORECASE,
)

_HISTORY_PATTERNS = re.compile(
    r"\b(previous|history|stored|cached|earlier|prior analysis|already run)\b",
    re.IGNORECASE,
)

_LITERATURE_PATTERNS = re.compile(
    r"\b(literature|paper|study|studies|evidence|publication|pubmed|"
    r"clinical|in vivo|in vitro|research|cite|citation)\b",
    re.IGNORECASE,
)

_PUBCHEM_PATTERNS = re.compile(
    r"\b(properties|structure|name|iupac|cas|formula|molecular weight|"
    r"pubchem|compound info|bioassay|assay data)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Deterministic fast-path selector
# ---------------------------------------------------------------------------


def _build_deterministic_manifest(
    user_request: str,
    plan_complexity: str,
) -> ToolManifest:
    """Select tools rule-based — zero LLM cost.

    Rules (applied in priority order):
      1. ScreeningModel — always selected.
      2. PubChem + PubMed — for MEDIUM and HIGH complexity.
      3. MolRAG + Firestore — for HIGH complexity OR analog keywords.
      4. ExplainabilityEngine — for mechanism/attribution keywords.
      5. PubMed — if literature keywords present (even on LOW complexity).
      6. PubChem — if compound info keywords present.
      7. MolRAG + Firestore — if analog/history keywords present.
    """
    complexity = str(plan_complexity or "MEDIUM").upper()
    text = str(user_request or "").strip()

    selections: List[ToolSelection] = []

    # --- ScreeningModel: always ---
    selections.append(
        ToolSelection(
            tool=ToolName.ScreeningModel,
            reason="ScreeningModel is required for every toxicity analysis.",
        )
    )

    # --- Complexity-driven: MEDIUM/HIGH → PubChem + PubMed ---
    if complexity in ("MEDIUM", "HIGH"):
        selections.append(
            ToolSelection(
                tool=ToolName.PubChem,
                reason=f"Compound property lookup needed for {complexity} complexity request.",
            )
        )
        selections.append(
            ToolSelection(
                tool=ToolName.PubMed,
                reason=f"Scientific literature retrieval needed for {complexity} complexity request.",
            )
        )

    # --- Complexity-driven: HIGH → MolRAG + Firestore ---
    if complexity == "HIGH":
        selections.append(
            ToolSelection(
                tool=ToolName.MolRAG,
                reason="Structural analog retrieval required for HIGH complexity comparative analysis.",
            )
        )
        selections.append(
            ToolSelection(
                tool=ToolName.Firestore,
                reason="Firestore is the MolRAG retrieval backend (co-activated with MolRAG).",
            )
        )

    # --- Keyword-driven overrides (can promote LOW to use more tools) ---
    existing_tools: Set[ToolName] = {s.tool for s in selections}

    if _EXPLAIN_PATTERNS.search(text) and ToolName.ExplainabilityEngine not in existing_tools:
        selections.append(
            ToolSelection(
                tool=ToolName.ExplainabilityEngine,
                reason="Mechanism explanation or structural attribution was explicitly requested.",
            )
        )
        existing_tools.add(ToolName.ExplainabilityEngine)

    if _LITERATURE_PATTERNS.search(text) and ToolName.PubMed not in existing_tools:
        selections.append(
            ToolSelection(
                tool=ToolName.PubMed,
                reason="Literature or scientific evidence was explicitly requested.",
            )
        )
        existing_tools.add(ToolName.PubMed)

    if _PUBCHEM_PATTERNS.search(text) and ToolName.PubChem not in existing_tools:
        selections.append(
            ToolSelection(
                tool=ToolName.PubChem,
                reason="Compound properties or bioassay data were explicitly requested.",
            )
        )
        existing_tools.add(ToolName.PubChem)

    if (
        (_ANALOG_PATTERNS.search(text) or _HISTORY_PATTERNS.search(text))
        and ToolName.MolRAG not in existing_tools
    ):
        selections.append(
            ToolSelection(
                tool=ToolName.MolRAG,
                reason="Structural analog comparison or retrieval was explicitly requested.",
            )
        )
        selections.append(
            ToolSelection(
                tool=ToolName.Firestore,
                reason="Firestore is the MolRAG retrieval backend (co-activated with MolRAG).",
            )
        )

    return ToolManifest(
        selected_tools=selections,
        selection_source="deterministic",
    )


# ---------------------------------------------------------------------------
# LLM fallback — direct GenAI API (same pattern as writer_agent.py)
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)

_TOOL_SELECTION_SYSTEM = """You are ToolSelectionAgent for ToxAgent.

Your role is to determine which tools should be executed to answer a toxicity-related request.

Available tools:

1. PubChem
   Use for: Compound properties, Canonical SMILES, Similar compounds

2. PubMed
   Use for: Scientific publications, Toxicity studies, Mechanism evidence

3. MolRAG
   Use for: Similar molecule retrieval, Structural analog search

4. Firestore
   Use for: User history, Previous analyses, Stored reports

5. ScreeningModel
   Use for: Tox21 prediction, ClinTox prediction, hERG prediction

6. ExplainabilityEngine
   Use for: GNNExplainer, Attribution maps, Structural explanation

Selection rules:
* Choose the minimum necessary tools.
* Prioritize ScreeningModel for new compounds.
* Use PubMed whenever scientific evidence is requested.
* Use MolRAG for analog comparison.
* Use Firestore when previous analyses may help.
* Use ExplainabilityEngine when mechanism interpretation is requested.

Output ONLY JSON with schema:
{"selected_tools": [{"tool": "...", "reason": "..."}]}
"""


def _call_llm_tool_selection(
    user_request: str,
    plan_complexity: str,
    deterministic: ToolManifest,
) -> Tuple[Optional[ToolManifest], str]:
    """Call GenAI API to get an LLM-based tool selection.

    Returns (ToolManifest, status) on success, (None, error) on failure.
    Caller falls back to deterministic manifest on failure.
    """
    try:
        from services.genai_runtime import (  # type: ignore[import]
            build_genai_client_candidates,
            call_with_retry,
            dedupe_strings,
            is_model_unavailable_error,
            is_resource_exhausted_error,
        )
    except ImportError as exc:
        logger.warning("tool_selection: genai_runtime unavailable (%s)", exc)
        return None, "genai_runtime_unavailable"

    client_candidates = build_genai_client_candidates()
    if not client_candidates:
        return None, "genai_client_unavailable"

    prompt = (
        f"{_TOOL_SELECTION_SYSTEM}\n\n"
        f"Plan complexity: {plan_complexity}\n"
        f"User request: {user_request}"
    )

    model_candidates = dedupe_strings([TOOL_SELECTION_MODEL, TOOL_SELECTION_MODEL])
    errors: List[str] = []

    for client, auth_mode in client_candidates:
        for model_name in model_candidates:
            try:
                response = call_with_retry(
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.1,
                            "response_mime_type": "application/json",
                        },
                    )
                )
                raw = str(getattr(response, "text", "") or "").strip()
                if not raw:
                    errors.append(f"empty_response:{auth_mode}:{model_name}")
                    continue

                # Strip code fences
                fence_match = _JSON_BLOCK.search(raw)
                json_str = fence_match.group(1).strip() if fence_match else raw

                data = json.loads(json_str)
                raw_tools = data.get("selected_tools", [])

                selections: List[ToolSelection] = []
                for item in raw_tools:
                    if not isinstance(item, dict):
                        continue
                    tool_name = str(item.get("tool") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    try:
                        selections.append(
                            ToolSelection(tool=ToolName(tool_name), reason=reason)
                        )
                    except ValueError:
                        logger.warning(
                            "tool_selection: LLM returned unknown tool %r; skipping",
                            tool_name,
                        )

                # ScreeningModel is always required — inject if LLM omitted it
                tool_set = {s.tool for s in selections}
                if ToolName.ScreeningModel not in tool_set:
                    selections.insert(
                        0,
                        ToolSelection(
                            tool=ToolName.ScreeningModel,
                            reason="ScreeningModel is always required (injected by safety rule).",
                        ),
                    )

                manifest = ToolManifest(
                    selected_tools=selections,
                    selection_source="llm",
                )
                return manifest, f"llm_success:{auth_mode}:{model_name}"

            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"llm_error:{type(exc).__name__}:{auth_mode}:{model_name}:{str(exc)[:150]}"
                )
                try:
                    if is_resource_exhausted_error(exc) or is_model_unavailable_error(exc):
                        continue
                except Exception:
                    pass
                return None, errors[-1]

    return None, (errors[0] if errors else "llm_failed")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tool_selection(
    user_request: str,
    plan_complexity: str = "MEDIUM",
) -> ToolManifest:
    """Produce a ToolManifest for the given user request.

    Strategy
    --------
    1. Run the deterministic selector (always — zero LLM cost).
    2. For HIGH complexity, call the LLM reviewer to refine selection.
    3. On LLM failure, fall back to deterministic manifest ('llm_fallback').
    4. ScreeningModel is always injected as a safety rule.
    5. Never raises — always returns a valid ToolManifest.

    Parameters
    ----------
    user_request    : Raw free-text from the user.
    plan_complexity : 'LOW' | 'MEDIUM' | 'HIGH' from AgentPlan.complexity.

    Returns
    -------
    ToolManifest — always valid.
    """
    if not user_request or not user_request.strip():
        logger.warning("run_tool_selection: empty user_request; returning minimal manifest")
        return ToolManifest(
            selected_tools=[
                ToolSelection(
                    tool=ToolName.ScreeningModel,
                    reason="Default: ScreeningModel always required.",
                )
            ],
            selection_source="deterministic",
        )

    complexity = str(plan_complexity or "MEDIUM").upper()

    # Step 1 — Deterministic fast-path (always)
    try:
        deterministic = _build_deterministic_manifest(user_request, complexity)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_tool_selection: deterministic selector failed (%s)", exc)
        deterministic = ToolManifest(
            selected_tools=[
                ToolSelection(
                    tool=ToolName.ScreeningModel,
                    reason="Fallback: deterministic selector failed.",
                )
            ],
            selection_source="deterministic",
        )

    logger.info(
        "run_tool_selection: deterministic tools=%s  complexity=%s",
        deterministic.tool_names(),
        complexity,
    )

    # Step 2 — LLM refinement for HIGH complexity only
    if complexity != "HIGH":
        return deterministic

    llm_manifest, status = _call_llm_tool_selection(
        user_request, complexity, deterministic
    )
    if llm_manifest is not None:
        logger.info(
            "run_tool_selection: LLM refined tools=%s  status=%s",
            llm_manifest.tool_names(),
            status,
        )
        return llm_manifest

    logger.warning(
        "run_tool_selection: LLM failed (%s); using deterministic fallback", status
    )
    object.__setattr__(deterministic, "selection_source", "llm_fallback")
    return deterministic


# ---------------------------------------------------------------------------
# tool_selection_agent LlmAgent — ADK wrapper (future graph integration)
# ---------------------------------------------------------------------------

_TOOL_SELECTION_INSTRUCTION = """You are ToolSelectionAgent.

Your role is to determine which tools should be executed to answer a toxicity-related request.

Available tools:

1. PubChem
   Use for:

   * Compound properties
   * Canonical SMILES
   * Similar compounds

2. PubMed
   Use for:

   * Scientific publications
   * Toxicity studies
   * Mechanism evidence

3. MolRAG
   Use for:

   * Similar molecule retrieval
   * Structural analog search

4. Firestore
   Use for:

   * User history
   * Previous analyses
   * Stored reports

5. ScreeningModel
   Use for:

   * Tox21 prediction
   * ClinTox prediction
   * hERG prediction

6. ExplainabilityEngine
   Use for:

   * GNNExplainer
   * Attribution maps
   * Structural explanation

Selection rules:

* Choose the minimum necessary tools.
* Prioritize ScreeningModel for new compounds.
* Use PubMed whenever scientific evidence is requested.
* Use MolRAG for analog comparison.
* Use Firestore when previous analyses may help.
* Use ExplainabilityEngine when mechanism interpretation is requested.

Output ONLY JSON.

Schema:

{
"selected_tools": [
{
"tool": "...",
"reason": "..."
}
]
}
"""

tool_selection_agent = LlmAgent(
    name="ToolSelectionAgent",
    model=TOOL_SELECTION_MODEL,
    description=(
        "Fine-grained tool router: determines the minimum necessary set of tools "
        "(ScreeningModel, PubChem, PubMed, MolRAG, Firestore, ExplainabilityEngine) "
        "to activate for a given user request and plan complexity."
    ),
    instruction=_TOOL_SELECTION_INSTRUCTION,
    tools=[],
    output_key="tool_manifest",
)
