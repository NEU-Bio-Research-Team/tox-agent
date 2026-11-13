"""PubChem BioAssay search, scoped to the bound subject and its data policy."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from ...application.conversation.subject_resolution import binding_of
from ...application.investigation import scientific_case_service
from ...domain.errors import AnalysisNotFound, ToolDenied
from ...domain.events import EventType
from ...domain.evidence import EvidenceStatus
from ...platform.flags import is_enabled
from ...research.normalization import hit_to_evidence
from ...research.policy import decide_acceptance
from ...research.relevance import RELEVANCE_POLICY_VERSION
from .. import trust
from ..registry import ToolContext, ToolDefinition, ToolOutput
from .primitives import _assay_subject

TOOL_NAME = "get_pubchem_bioassays"


class PubChemAssayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str | None = Field(default=None, description="Analysis to search, or omit for the bound subject.")
    target: str = Field(default="herg", pattern="^herg$")
    limit: int = Field(default=10, ge=1, le=20)


def build(database, provider) -> list[ToolDefinition]:
    async def assays(context: ToolContext, payload: PubChemAssayInput) -> ToolOutput:
        binding = binding_of(context)
        analysis_id, smiles = await _assay_subject(database, context, payload, binding)
        async with database.unit_of_work() as uow:
            snapshot = (await uow.analyses.get(analysis_id, session_id=context.session_id)
                        if analysis_id else None)
            refusal = await scientific_case_service.external_search_refusal(
                uow, session_id=context.session_id,
                analysis_id=binding.analysis_id if binding is not None else analysis_id,
                binding=binding,
            )
        if analysis_id and snapshot is None:
            raise AnalysisNotFound("no such analysis in this session", analysis_id=analysis_id)
        if refusal is not None:
            raise ToolDenied("the researcher's data scope does not allow external lookups; "
                             "PubChem was not queried", reason="case_data_scope")
        found = await provider.assays(canonical_smiles=snapshot.canonical_smiles if snapshot else smiles,
                                      target=payload.target, limit=payload.limit)
        records, ids = [], []
        async with database.unit_of_work() as uow:
            for hit in found.hits:
                candidate = hit_to_evidence(hit, provider=provider.name, session_id=context.session_id,
                                            retrieved_at=datetime.now(timezone.utc))
                final = await uow.evidence.find_by_dedupe_key(context.session_id, candidate.dedupe_key)
                if final is None:
                    final = decide_acceptance(candidate, allowed_hosts=provider.allowed_hosts)
                    final = replace(final, relevance_assessment={
                        "relevance": "direct", "reason_codes": ["pubchem_structure_resolution", "target_match"],
                        "policy_version": RELEVANCE_POLICY_VERSION,
                    })
                    stored = await uow.evidence.add_if_absent(final)
                    if stored is final:
                        uow.emit(session_id=context.session_id, type=EventType.EVIDENCE_CREATED,
                                 entity_type="evidence", entity_id=final.id, run_id=context.run_id,
                                 payload={"provider": provider.name, "status": final.status.value})
                    final = stored
                if final.status is EvidenceStatus.ACCEPTED:
                    ids.append(final.id)
                    view = final.model_view()
                    records.append(trust.wrap_evidence_view(view, provider=provider.name,
                                                            record_id=final.provider_record_id)
                                   if is_enabled("trust_envelope_v1") else view)
            await uow.commit()
        view = {
            "pubchem_cid": found.cid, "target": payload.target, "total_matching_assays": found.total_matches,
            "returned": len(records), "truncated": found.total_matches > len(found.hits),
            "records": records,
            "note": ("PubChem assay summaries are screening results, not a safety verdict. "
                     "Check each assay's protocol, endpoint, target and original source before "
                     "comparing results; the same experiment may appear in other databases. "
                     "PubChem standardizes submitted structures; check stereochemistry and salt "
                     "forms when they matter. Missing results do not prove inactivity."),
        }
        return ToolOutput(canonical=view, model_view=view, ui_view=view,
                          observation_ids=tuple(ids), provenance={"provider": provider.name,
                                                                  "analysis_id": analysis_id})

    return [ToolDefinition(
        name=TOOL_NAME, title="Find PubChem hERG bioassays",
        description=("Look up PubChem BioAssay summaries for the bound structure and human hERG "
                     "(GeneID 3757 or UniProt Q12809). Returns citable assay records including "
                     "activity outcome, assay description and source links. Results can include screening "
                     "endpoints and repeated studies; "
                     "inspect the assay before comparing or making quantitative claims."),
        input_model=PubChemAssayInput, handler=assays, profiles=frozenset({"decision_support"}),
        soft_timeout_s=15.0, hard_timeout_s=45.0, max_retries=0, cost_class="moderate",
    )]
