"""W9-02: why the case arm's first-pass acceptance fell, and the fixes.

The W8-02 paired run (C 0.54, D 0.26 first-pass) was diagnosed from the
rejection events it stored: under grounded-answer v2 the most frequent
first-draft refusal was ``unclaimed_numeric_value`` — the model copied the
value a tool printed (``0.7999394536018372``) into its prose, while the server
renders an identity claim to twelve places (``0.799939453602``). The model
cannot know that string, so the draft could not have passed. The second defect:
``explainer_validation`` is shown beside every attribution but did not resolve
as a field, so a claim citing it was refused twice and the turn fell back.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.conversation.submit_answer import SubmitAnswer
from toxagent.config import PolicySettings
from toxagent.domain import explainer_validation as ev
from toxagent.domain.errors import AnswerValidationFailed
from toxagent.domain.observation import Observation, ObservationKind, Producer
from toxagent.validation.answer.candidate_wire import LimitationCandidate
from toxagent.validation.answer.draft_wire import ClaimCandidateV2, GroundedAnswerDraftV2
from tests.integration.test_evidence_relations_wiring import rig

pytestmark = pytest.mark.anyio

#: tests/support/predictor.py's default hERG probability.
HERG = 0.73064


def _numeric(observation, **overrides) -> ClaimCandidateV2:
    fields = dict(
        local_ref="herg_p", kind="numeric", text="Predicted hERG blocker probability.",
        observation_id=observation.id, field_path="predictions.herg.probability_blocker",
    )
    fields.update(overrides)
    return ClaimCandidateV2(**fields)


def _draft(observation, markdown: str, **claim_overrides) -> GroundedAnswerDraftV2:
    return GroundedAnswerDraftV2(
        answer_markdown=markdown,
        claims=[_numeric(observation, **claim_overrides)],
        limitations=[LimitationCandidate(code="uncalibrated_probability", text="")],
    )


async def _submit(db, session, run, draft, *, language="en"):
    return await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
        session_id=session.id, run_id=run.id, candidate=draft, language=language,
    )


def _codes(exc_info) -> set[str]:
    return {
        v["code"] if isinstance(v, dict) else v.code
        for v in exc_info.value.detail["violations"]
    }


async def test_the_full_value_a_tool_printed_is_accepted_in_the_prose(db):
    session, run, observation = await rig(db)
    outcome = await _submit(db, session, run, _draft(
        observation, f"The predicted hERG blocker probability is {HERG!r}.",
    ))
    assert outcome.is_fallback is False


@pytest.mark.parametrize("written", ["0.731", "0.73", "73.1%", "73%"])
async def test_a_faithful_rounding_of_a_claimed_value_is_accepted(db, written):
    session, run, observation = await rig(db)
    outcome = await _submit(db, session, run, _draft(
        observation, f"The predicted hERG blocker probability is {written}.",
    ))
    assert outcome.is_fallback is False


async def test_a_vietnamese_decimal_comma_is_the_same_number(db):
    session, run, observation = await rig(db)
    outcome = await _submit(db, session, run, _draft(
        observation, "Xác suất chặn hERG dự đoán là 0,731.",
    ), language="vi")
    assert outcome.is_fallback is False


@pytest.mark.parametrize("written", ["0.74", "0.9", "81%"])
async def test_a_number_that_is_no_claims_value_is_still_refused(db, written):
    session, run, observation = await rig(db)
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await _submit(db, session, run, _draft(
            observation, f"The predicted hERG blocker probability is {written}.",
        ))
    assert "unclaimed_numeric_value" in _codes(exc_info)


async def test_a_placeholder_is_filled_with_the_server_rendered_value(db):
    session, run, observation = await rig(db)
    outcome = await _submit(db, session, run, _draft(
        observation, "The predicted hERG blocker probability is {{herg_p}}.",
        transform="round:3",
    ))
    assert outcome.answer.answer_markdown == "The predicted hERG blocker probability is 0.731."


async def test_a_placeholder_uses_the_locale_rendering(db):
    session, run, observation = await rig(db)
    outcome = await _submit(db, session, run, _draft(
        observation, "Xác suất chặn hERG là {{herg_p}}.", transform="percent:1",
    ), language="vi")
    assert outcome.answer.answer_markdown == "Xác suất chặn hERG là 73,1%."


async def test_a_placeholder_naming_no_claim_is_a_correctable_violation(db):
    session, run, observation = await rig(db)
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await _submit(db, session, run, _draft(observation, "The value is {{clintox_p}}."))
    assert "answer_placeholder_unknown" in _codes(exc_info)


async def test_a_placeholder_for_a_claim_without_a_value_inserts_its_text(db):
    """W9-A: models mark scientific claims with placeholders too; refusing them
    cost 26 first drafts in the C + answer_draft_v2 run."""
    session, run, observation = await rig(db)
    draft = GroundedAnswerDraftV2(
        answer_markdown="Note: {{note}}",
        claims=[ClaimCandidateV2(
            local_ref="note", kind="scientific", text="The score is a model output.",
            observation_id=observation.id, field_path="predictions.herg.probability_blocker",
        )],
    )
    outcome = await _submit(db, session, run, draft)
    assert outcome.answer.answer_markdown == "Note: The score is a model output."


async def test_the_explainer_verdict_shown_beside_an_attribution_is_citable(db):
    session, run, observation = await rig(db)
    attribution = Observation.create(
        session_id=session.id, run_id=run.id, producer=Producer.ATTRIBUTION,
        kind=ObservationKind.ATTRIBUTION, schema_version="toxpred-attribution-v1",
        canonical_payload={"tokens": [{"token": "Cl", "score": -0.1}]},
        model_projection={
            "endpoint": "herg", "task": None, "status": "completed",
            "method": "grad_x_input_v2", "model_id": "herg-tox21-chemberta-v1",
        },
        provenance={}, now=datetime.now(timezone.utc),
        required_limitations=("attribution_not_causality",),
    )
    async with db.unit_of_work() as uow:
        await uow.observations.add(attribution, analysis_id=observation.provenance.get("analysis_id"))
        await uow.commit()
    draft = GroundedAnswerDraftV2(
        answer_markdown="The attribution method was not more faithful than a random control.",
        claims=[ClaimCandidateV2(
            local_ref="xai", kind="classification",
            text="The attribution method was not more faithful than a random control.",
            observation_id=attribution.id,
            field_path="explainer_validation.faithfulness_vs_random_control",
        )],
        limitations=[LimitationCandidate(code="attribution_not_causality", text="")],
    )
    outcome = await _submit(db, session, run, draft)
    assert outcome.is_fallback is False
    assert outcome.answer.claims[0].source_value == ev.Faithfulness.NOT_BETTER_THAN_RANDOM
