"""``propose_skill_draft``: the model writes down a method for an expert to review.

RETHINK §4.8: a skill a model distils from its own work may be kept as a draft
for an expert; it never activates by itself. This tool is how a decision-support
run proposes one. The draft is validated with the catalog's own loader, stored
with the session and run it came from, and reaches the catalog only if an
expert approves it and a reviewed change ships it. Nothing the model writes
here is offered to any run, including its own.

Registered only while ``skill_drafts_v1`` is on (FLAG_GATED_TOOLS).
"""
from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ...application import skill_drafts
from ...domain.errors import InvalidRequest
from ...domain.skill_draft import DraftAuthor
from ..registry import ToolContext, ToolDefinition, ToolOutput

PROPOSE_TOOL: Final[str] = "propose_skill_draft"


class ProposeSkillDraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(
        min_length=3, max_length=64,
        description="Lowercase words joined by hyphens, naming the situation, e.g. "
                    "'weigh-species-differences'.",
    )
    description: str = Field(
        min_length=20, max_length=1024,
        description="When the skill applies and when it does not — what a later run reads "
                    "to decide whether to load it.",
    )
    instructions: str = Field(
        min_length=100, max_length=8000,
        description="The method in Markdown: what to check, signs to change course, how to use "
                    "tools that already exist, when to stop, what to leave in the case. Not a "
                    "fixed sequence of tool calls.",
    )
    required_tools: list[str] = Field(
        min_length=1, max_length=8,
        description="Existing tools the method relies on. A skill never adds a tool.",
    )
    output_contract: str = Field(min_length=10, max_length=500)
    rationale: str = Field(
        min_length=20, max_length=2000,
        description="What in this investigation showed the method was missing.",
    )


def build(database, skill_catalog) -> list[ToolDefinition]:
    async def propose(context: ToolContext, payload: ProposeSkillDraftInput) -> ToolOutput:
        skill_md, manifest = skill_drafts.compose_package(
            skill_id=payload.skill_id, description=payload.description,
            body=payload.instructions, required_capabilities=payload.required_tools,
            output_contract=payload.output_contract,
        )
        async with database.unit_of_work() as uow:
            try:
                draft = await skill_drafts.propose(
                    uow,
                    author=DraftAuthor(actor="model", subject_id=f"model:{context.run_id}",
                                       session_id=context.session_id, run_id=context.run_id),
                    skill_md=skill_md, manifest=manifest, references={},
                    rationale=payload.rationale, catalog=skill_catalog,
                )
            except skill_drafts.DraftRefused as exc:
                raise InvalidRequest(f"the draft was not stored: {exc}") from None
            await uow.commit()
        view = {
            "draft_id": draft.id, "skill_id": draft.skill_id, "status": draft.status,
            "note": "Stored for expert review. It is not available to this or any run "
                    "unless an expert approves it and it ships with a release.",
        }
        return ToolOutput(canonical=draft.to_dict(), model_view=view, ui_view=view,
                          provenance={"draft_id": draft.id})

    return [
        ToolDefinition(
            name=PROPOSE_TOOL,
            title="Propose a scientific skill draft for expert review",
            description=(
                "Only when this investigation exposed a reusable method the listed skills do not "
                "cover: write it down as a draft for an expert to review. It changes nothing for "
                "this run and is never loaded automatically. Most turns should not call this."
            ),
            input_model=ProposeSkillDraftInput,
            handler=propose,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=5.0,
            hard_timeout_s=15.0,
            idempotent=False,
            cost_class="cheap",
        )
    ]
