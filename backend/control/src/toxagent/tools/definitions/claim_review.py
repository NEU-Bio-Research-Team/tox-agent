"""``submit_claim_review``: the independent reviewer's one tool (W9-12).

The reviewer turn's only action. Verdicts are checked against the accepted
answer (every reviewed claim once, no other id) and stored on the run's
DecisionSupportStateV1; the answer is never touched. Registered only while
``claim_reviewer_v1`` is on (FLAG_GATED_TOOLS).
"""
from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation import claim_review, decision_state_service
from ...domain import decision_state as ds
from ...domain.errors import InvalidRequest
from ..registry import ToolContext, ToolDefinition, ToolOutput

CLAIM_REVIEW_TOOL_NAME: Final[str] = "submit_claim_review"


class ClaimVerdictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Literal["supported", "partially_supported", "not_supported", "cannot_judge"]
    reason: str = Field(min_length=1, max_length=500)


class SubmitClaimReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[ClaimVerdictInput] = Field(min_length=1, max_length=64)


def build(database) -> list[ToolDefinition]:
    async def submit(context: ToolContext, payload: SubmitClaimReviewInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            answer = await uow.answers.get_for_run(context.run_id)
        if answer is None:
            raise InvalidRequest("this run has no accepted answer to review")
        expected = {
            c.claim_id for c in answer.claims
            if getattr(c.kind, "value", c.kind) in claim_review.REVIEWED_KINDS
        }
        given = [item.claim_id for item in payload.reviews]
        unknown = sorted(set(given) - expected)
        missing = sorted(expected - set(given))
        repeated = sorted({cid for cid in given if given.count(cid) > 1})
        if unknown or missing or repeated:
            raise InvalidRequest(
                "give exactly one verdict for every claim listed: "
                f"unknown {unknown}, missing {missing}, repeated {repeated}"
            )
        reviews = [item.model_dump() for item in payload.reviews]
        record = {"status": "completed", "answer_id": answer.id, "reviews": reviews,
                  "counts": claim_review.summarize(reviews)}
        await decision_state_service.advance(
            database, context.run_id, lambda state: ds.record_claim_review(state, record),
        )
        view = {"accepted": True, "counts": record["counts"]}
        return ToolOutput(canonical=record, model_view=view, ui_view=record,
                          provenance={"answer_id": answer.id})

    return [
        ToolDefinition(
            name=CLAIM_REVIEW_TOOL_NAME,
            title="Submit an independent claim-support review",
            description=(
                "Your only action: one verdict (supported, partially_supported, not_supported, "
                "cannot_judge) with a one-sentence reason for every claim listed in the prompt, "
                "judged only from the sources listed with it."
            ),
            input_model=SubmitClaimReviewInput,
            handler=submit,
            profiles=frozenset({"claim_review"}),
            soft_timeout_s=5.0,
            hard_timeout_s=10.0,
            idempotent=False,
            cost_class="cheap",
        )
    ]
