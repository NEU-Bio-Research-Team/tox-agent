from __future__ import annotations

import os
from typing import Any, Dict

from tools import (
    get_compound_info_pubchem,
    get_pubchem_bioassay_data,
    search_toxicity_literature,
)

from .adk_compat import LlmAgent

RESEARCH_MODEL = os.getenv("AGENT_MODEL_PRO", "gemini-2.5-pro")


def run_research(smiles_input: str, max_results: int = 5) -> Dict[str, Any]:
    """Deterministic research flow used for local tests and orchestration."""
    compound_info = get_compound_info_pubchem(smiles_input)

    preferred_name = (
        compound_info.get("common_name")
        or compound_info.get("iupac_name")
        or smiles_input
    )
    literature = search_toxicity_literature(preferred_name, max_results=max_results)

    cid = compound_info.get("cid")
    bioassay_summary = None
    if cid:
        bioassay_summary = get_pubchem_bioassay_data(cid)

    research_result = {
        "compound_info": compound_info,
        "literature": literature,
        "bioassay_summary": bioassay_summary,
        "query_name_used": preferred_name,
    }

    return {
        "research_result": research_result,
        "research_error": None,
    }


researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model=RESEARCH_MODEL,
    description=(
        "Gather PubChem and PubMed context for a molecule."
    ),
    instruction="""
You are a drug safety literature researcher.

Task:
1. Read SMILES from {smiles_input}.
2. Call get_compound_info_pubchem(smiles={smiles_input}).
3. Use common_name (or iupac_name if common_name is missing) to call
   search_toxicity_literature(compound_name=<best_name>, max_results=5).
4. If CID exists, call get_pubchem_bioassay_data(cid=<CID>).
5. Return JSON for key research_result with fields:
   - compound_info
   - literature
   - bioassay_summary
   - query_name_used

Rules:
- Continue gracefully if one tool fails.
- Do not invent PMID/CID values.
- Keep original tool errors in returned payload.
""",
    tools=[
        get_compound_info_pubchem,
        search_toxicity_literature,
        get_pubchem_bioassay_data,
    ],
    output_key="research_result",
)
