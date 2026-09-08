from datetime import datetime, timezone

from toxagent.answer.compiler import AnswerCompiler, SemanticAnswerDraft, SemanticClaim
from toxagent.domain.ids import RUN, SESSION, new_id
from toxagent.domain.observation import Observation, ObservationKind, Producer


def test_compiler_owns_ids_values_rounding_and_percentages():
    observation = Observation.create(
        session_id=new_id(SESSION), run_id=new_id(RUN), producer=Producer.PREDICTOR,
        kind=ObservationKind.PREDICTION, schema_version="v1",
        canonical_payload={"predictions": {"herg": {"probability_blocker": 0.73146}}},
        model_projection={}, provenance={}, now=datetime.now(timezone.utc),
    )
    draft = SemanticAnswerDraft((
        ("Result", (SemanticClaim(
            "numeric", "hERG blocker probability is {value}.", observation.id,
            "predictions.herg.probability_blocker", "percent:2",
        ),)),
    ))
    candidate = AnswerCompiler().compile(draft, observations={observation.id: observation})
    claim = candidate.claims[0]
    assert claim.claim_id.startswith("clm_")
    assert claim.source_value == 0.73146
    assert claim.rendered_value == "73.15%"
    assert "73.15%" in candidate.answer_markdown
