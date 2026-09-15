"""The submit_grounded_answer tool (plan section 8.4).

The only path by which a conversational answer becomes canonical. Its input
model *is* the wire candidate — no wrapper, no session id argument, because the
session and run are the token's, not the model's to state (plan section 8.5).
Validation, the correction policy and the deterministic fallback all live in
``application.submit_answer``; this module is only the tool-plane adapter.
"""
from __future__ import annotations

from typing import Final

from ...application.submit_answer import SubmitAnswer
from ...config import PolicySettings
from ...flags import is_enabled
from ...validation.wire import GroundedAnswerCandidate
from ...validation.wire_v2 import GroundedAnswerDraftV2
from ..registry import ToolContext, ToolDefinition, ToolOutput

#: Referenced by tools/runner.py so the final-answer tool can be excluded from
#: the per-run tool-call budget (plan section 14.5) without a duplicated
#: string literal.
ANSWER_TOOL_NAME: Final[str] = "submit_grounded_answer"


#: The v2 surface. Notice what is missing: no identifier format, no worked
#: example of one, and no instruction about decimal commas. The server issues
#: the ids and renders the values, so there is nothing here for a model to
#: imitate wrongly (P1-4).
_V2_DESCRIPTION: Final[str] = (
    "Submit the final answer to this turn's question. Give each claim a short "
    "local_ref label ('herg_p', 'claim_1') unique within this answer; the server "
    "issues the permanent identifier. Every numeric or classification claim names "
    "the observation_id and field_path it comes from — obtained from "
    "get_analysis_slice or get_attribution — and the server reads, renders and "
    "checks the value itself, so do not send the number. Every scientific or "
    "comparison claim needs either such a citation or an accepted evidence "
    "citation_id.\n\n"
    "To compare two predictor values ('how much higher is X than Y'): submit each "
    "value as its own numeric claim, then a third with kind=comparison, "
    "transform='difference' (first minus second) or 'ratio', and input_local_refs "
    "naming those two claims in that order. The server computes the result. Do not "
    "write the two source numbers directly into answer_markdown — coverage checking "
    "rejects an unclaimed number.\n\n"
    "A rejected draft returns typed violations naming exactly what to correct; at "
    "most one correction attempt is allowed per run."
)


def build(database, settings: PolicySettings) -> list[ToolDefinition]:
    submit_answer = SubmitAnswer(database, settings)

    async def submit(context: ToolContext, payload) -> ToolOutput:
        outcome = await submit_answer.execute(
            session_id=context.session_id, run_id=context.run_id, candidate=payload,
            language=context.language,
        )
        return ToolOutput(
            canonical=outcome.answer.to_dict(),
            model_view={
                "answer_id": outcome.answer.id,
                "accepted": True,
                "is_fallback": outcome.is_fallback,
                "candidate_generation": outcome.answer.candidate_generation,
            },
            ui_view=outcome.answer.to_dict(),
            observation_ids=tuple(outcome.answer.cited_observation_ids),
            provenance={"candidate_generation": outcome.answer.candidate_generation},
        )

    if is_enabled("answer_draft_v2"):
        return [
            ToolDefinition(
                name=ANSWER_TOOL_NAME,
                title="Submit a grounded answer",
                description=_V2_DESCRIPTION,
                input_model=GroundedAnswerDraftV2,
                handler=submit,
                profiles=frozenset({"analysis", "report_qa", "evidence_research", "decision_support"}),
                soft_timeout_s=5.0,
                hard_timeout_s=10.0,
                idempotent=False,
            )
        ]

    return [
        ToolDefinition(
            name=ANSWER_TOOL_NAME,
            title="Submit a grounded answer",
            description=(
                "Submit the final answer to this turn's question. Every claim_id must be "
                "\"clm_\" followed by 32 lowercase hex characters, unique across every answer "
                "this deployment has ever accepted, not only within this candidate — a "
                "low-entropy pattern like repeating one digit is rejected the first time it "
                "collides with an unrelated answer. Generate 32 characters that look random "
                "(e.g. \"clm_7a3f9c21b6e84d0f9a1c5e7b2d8f4a6c\"), not a short label like \"c1\" or "
                "a repeated digit. Every numeric "
                "or classification claim must cite an observation_id and field_path obtained "
                "from get_analysis_slice or get_attribution; every scientific or comparison "
                "claim needs either such a citation or an accepted evidence citation_id. For a "
                "numeric claim, rendered_value is a single number only — e.g. \"0.731\", or "
                "\"0,731\" with a Vietnamese decimal comma, optionally with a trailing \"%\"; put "
                "any \"0,0315 (3,15%)\"-style phrasing in the claim's text, never in "
                "rendered_value. A rejected candidate returns typed violations naming exactly "
                "what to correct; at most one correction attempt is allowed per run.\n\n"
                "To compare two predictor values (e.g. \"how much higher is X than Y\", "
                "\"is X above or below Y\"): first submit each value as its own numeric claim "
                "(citing its own observation_id/field_path, as always), then submit a third claim "
                "with kind=comparison, transform=\"difference\" (first minus second) or \"ratio\", "
                "input_claim_ids naming exactly those two claim_ids in that order, source_value "
                "equal to the actual difference/ratio of their source_values, and rendered_value a "
                "single number rendering that same difference/ratio — never the two source numbers "
                "written directly into answer_markdown, which coverage checking rejects as "
                "unclaimed. Do not use kind=numeric or kind=classification for a comparison "
                "between two fields; a numeric claim's field_path must resolve to exactly one "
                "predictor field, and a difference between two fields is not one."
            ),
            input_model=GroundedAnswerCandidate,
            handler=submit,
            profiles=frozenset({"analysis", "report_qa", "evidence_research", "decision_support"}),
            soft_timeout_s=5.0,
            hard_timeout_s=10.0,
            idempotent=False,
        )
    ]
