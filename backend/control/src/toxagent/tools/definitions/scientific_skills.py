"""``read_scientific_skill`` / ``read_skill_reference``: skills loaded on demand.

ADR 0012, RETHINK §4.9. The dynamic arm puts only skill names and descriptions
in the prompt; these two tools return a skill's pinned body, or one of its
declared references, when the model decides the situation calls for it. They
read the catalog the control plane loaded and hashed — never the filesystem on
the model's behalf, so the runtime's own ``read``/``skill`` permissions stay
denied.

A skill is offered only if every tool it requires is visible to the run; the
same check runs here, so a model cannot read its way into instructions for a
capability it does not have. Every read is recorded in the run's
DecisionSupportStateV1 with the skill's version and content hash: "which
instructions shaped this answer" stays answerable from the run alone.

Registered only while ``scientific_skills_v1`` is on (FLAG_GATED_TOOLS).
"""
from __future__ import annotations

from typing import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from ...application import decision_state_service
from ...domain import decision_state as ds
from ...domain.errors import InvalidRequest
from ...application.skill_catalog import SkillCatalog
from ..registry import ToolContext, ToolDefinition, ToolOutput

READ_SKILL = "read_scientific_skill"
READ_REFERENCE = "read_skill_reference"


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadSkillInput(_Input):
    skill_id: str = Field(min_length=1, max_length=64,
                          description="A skill name from the scientific skills list in the prompt.")


class ReadReferenceInput(_Input):
    skill_id: str = Field(min_length=1, max_length=64)
    reference: str = Field(min_length=1, max_length=120,
                           description="One reference file name the skill lists.")


def build(
    database, catalog: SkillCatalog, visible_tools: Callable[[str], Iterable[str]],
) -> list[ToolDefinition]:
    def offered(context: ToolContext, skill_id: str):
        skill = next(
            (s for s in catalog.available(context.profile, visible_tools(context.profile))
             if s.skill_id == skill_id),
            None,
        )
        if skill is None:
            names = [s.skill_id for s in catalog.available(context.profile, visible_tools(context.profile))]
            raise InvalidRequest(
                f"no scientific skill {skill_id!r} is available to this run; available: {names}"
            )
        return skill

    async def read_skill(context: ToolContext, payload: ReadSkillInput) -> ToolOutput:
        skill = offered(context, payload.skill_id)
        await decision_state_service.advance(
            database, context.run_id, lambda state: ds.record_skill_loaded(state, pin=skill.pin()),
        )
        view = {
            **skill.pin(),
            "instructions": skill.body,
            "references": sorted(skill.references),
            "note": "These instructions sit below the scientific invariants and the answer "
                    "validator; they never add a tool.",
        }
        return ToolOutput(canonical=skill.metadata(), model_view=view, ui_view=skill.metadata(),
                          provenance={"skill": skill.pin()})

    async def read_reference(context: ToolContext, payload: ReadReferenceInput) -> ToolOutput:
        skill = offered(context, payload.skill_id)
        name = payload.reference.removeprefix("references/")
        text = skill.references.get(name)
        if text is None:
            raise InvalidRequest(
                f"{skill.skill_id} has no reference {payload.reference!r}; "
                f"it lists {sorted(skill.references)}"
            )
        await decision_state_service.advance(
            database, context.run_id,
            lambda state: ds.record_skill_loaded(state, pin=skill.pin(), reference=name),
        )
        view = {**skill.pin(), "reference": name, "content": text}
        return ToolOutput(canonical={**skill.pin(), "reference": name},
                          model_view=view, ui_view={**skill.pin(), "reference": name},
                          provenance={"skill": skill.pin(), "reference": name})

    return [
        ToolDefinition(
            name=READ_SKILL,
            title="Read a scientific skill",
            description=(
                "Load the instructions of one scientific skill listed in the prompt, when the "
                "situation its description names applies to this turn. Returns the pinned text "
                "and the names of its references. Costs no evidence budget."
            ),
            input_model=ReadSkillInput,
            handler=read_skill,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            cost_class="cheap",
        ),
        ToolDefinition(
            name=READ_REFERENCE,
            title="Read one reference of a scientific skill",
            description=(
                "Load one reference file a skill lists, only when the skill says it is needed. "
                "References are background for reasoning, not citable sources."
            ),
            input_model=ReadReferenceInput,
            handler=read_reference,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            cost_class="cheap",
        ),
    ]
