from __future__ import annotations

import os
from typing import Any, Dict

from tools import (
    get_compound_info_pubchem,
    get_pubchem_bioassay_data,
    search_toxicity_literature,
    synthesize_literature,
)

from .adk_compat import LlmAgent

RESEARCH_MODEL = os.getenv("AGENT_MODEL_PRO", "gemini-2.5-pro")


def run_research(smiles_input: str, max_results: int = 5, language: str = "vi") -> Dict[str, Any]:
    """Deterministic research flow used for local tests and orchestration."""
    compound_info = get_compound_info_pubchem(smiles_input)

    preferred_name = (
        compound_info.get("common_name")
        or compound_info.get("iupac_name")
        or smiles_input
    )
    literature = search_toxicity_literature(
        preferred_name,
        max_results=max_results,
        compound_smiles=smiles_input,
    )
    literature_synthesis = None
    articles = literature.get("articles", []) if isinstance(literature, dict) else []
    if isinstance(articles, list) and articles:
        literature_synthesis = synthesize_literature(
            articles=articles,
            compound_name=preferred_name,
            compound_smiles=smiles_input,
            language=language,
        )

    cid = compound_info.get("cid")
    bioassay_summary = None
    if cid:
        bioassay_summary = get_pubchem_bioassay_data(cid)

    research_result = {
        "compound_info": compound_info,
        "literature": literature,
        "literature_synthesis": literature_synthesis,
        "bioassay_summary": bioassay_summary,
        "query_name_used": preferred_name,
        "language": language,
    }

    return {
        "research_result": research_result,
        "research_error": None,
    }


researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model=RESEARCH_MODEL,
    description=(
        "Gather PubChem and literature context, then synthesize key toxicology findings."
    ),
    instruction="""
You are a drug safety literature researcher and toxicology synthesis assistant.

Task:
1. Read SMILES from {smiles_input}.
2. Read language from {language} and use it for user-facing text fields.
2. Call get_compound_info_pubchem(smiles={smiles_input}).
3. Use common_name (or iupac_name if common_name is missing) to call
   search_toxicity_literature(compound_name=<best_name>, max_results=5).
4. If articles are returned, call synthesize_literature(
   articles=<articles_from_step3>,
   compound_name=<best_name>,
   compound_smiles={smiles_input},
   language={language}
).
5. If CID exists, call get_pubchem_bioassay_data(cid=<CID>).
5. Return JSON for key research_result with fields:
   - compound_info
   - literature
   - literature_synthesis
   - bioassay_summary
   - query_name_used

Rules:
- Continue gracefully if one tool fails.
- Do not invent PMID/CID values.
- Keep original tool errors in returned payload.
- If abstracts are unavailable, keep synthesis confidence low and note the limitation.
""",
    tools=[
        get_compound_info_pubchem,
        search_toxicity_literature,
        synthesize_literature,
        get_pubchem_bioassay_data,
    ],
    output_key="research_result",
)
