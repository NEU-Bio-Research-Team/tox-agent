"""XAI-01: one explanation pipeline, one cache, one schema version.

There used to be two. ``create_analysis`` computed eager explanations, wrote them
under ``toxpred-explanation-v2`` and checkpointed them under
``explanation_checkpoint_key``; ``application/explanation/service.py`` wrote
``toxpred-explanation-v1`` and looked them up under a key built from every
model's artifact hashes. Neither could see the other's work, so a report builder
holding a finished explanation asked the predictor for it again — or recorded a
gap — with the answer already in the database.

These tests are about who calls the predictor, and how often. Every one of them
counts ``explain`` calls, because "the report reused it" and "the report
recomputed it and got the same answer" are indistinguishable from the report
itself and differ by one backward pass.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.explanation.service import (
    EXPLANATION_SCHEMA_VERSION,
    ExplanationUnavailable,
    GetOrCreateExplanation,
    classify_failure,
    has_numeric_attribution,
    is_explanation_schema,
    package_from_observation,
)
from toxagent.application.explanation.identity import (
    explanation_cache_key,
    model_artifact_fingerprint,
)
from toxagent.application.policy import Actor
from toxagent.platform.config import PolicySettings
from toxagent.domain.message import Message, Role
from toxagent.domain.observation import ObservationKind
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.persistence.object_store import InMemoryObjectStore
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="360">'
    '<ellipse cx="10" cy="10" rx="4" ry="4" style="fill:#D73027" /></svg>'
)


class _CountingPredictor:
    """A real client, wrapped to count ``explain`` calls."""

    def __init__(self, **stub: object) -> None:
        self._client = StubPredictor(**stub).client()
        self.explained: list[tuple[str, str | None]] = []

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def explain(self, smiles, endpoint, task=None, *, model_id=None):
        self.explained.append((endpoint, task))
        return await self._client.explain(smiles, endpoint, task, model_id=model_id)


async def _seed(db) -> tuple[str, str]:
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.DETERMINISTIC, Intent.ANALYSIS, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    return session.id, run.id


async def _analyse(db, predictor, session_id, run_id, *, targets=(("herg", None),)):
    return await CreateAnalysis(db, predictor, PolicySettings()).execute(
        actor=ACTOR, session_id=session_id, run_id=run_id, smiles="CCO",
        endpoints=("herg", "tox21"), explanation_mode="required",
        explanation_targets=targets,
    )


async def _attributions(db, analysis_id):
    async with db.unit_of_work() as uow:
        observations = await uow.observations.list_for_analysis(analysis_id)
    return [o for o in observations if o.kind is ObservationKind.ATTRIBUTION]


# --- one schema version, one key --------------------------------------------

async def test_the_analysis_pipeline_writes_the_shared_schema_version(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor()
    result = await _analyse(db, predictor, session_id, run_id)

    written = await _attributions(db, result.snapshot.id)
    assert [o.schema_version for o in written] == [EXPLANATION_SCHEMA_VERSION]


async def test_the_analysis_pipeline_records_the_cache_key_on_the_observation(db):
    """Without this, the observation is findable only by its target, and the
    report builder cannot tell whether the attribution in front of it was
    computed under the weights currently loaded."""
    session_id, run_id = await _seed(db)
    result = await _analyse(db, _CountingPredictor(), session_id, run_id)

    observation = (await _attributions(db, result.snapshot.id))[0]
    model_id = result.snapshot.model_for("herg")
    assert observation.provenance["cache_key"] == explanation_cache_key(
        canonical_smiles=result.snapshot.canonical_smiles,
        endpoint="herg", task=None, model_id=model_id,
        # Per-model, so retraining one admitted model leaves the other's
        # explanations valid rather than invalidating the whole cache.
        artifact_fingerprint=model_artifact_fingerprint(
            result.snapshot.provenance, model_id
        ),
    )
    assert observation.provenance["explanation_id"].startswith("xpl_")
    assert observation.provenance["alignment_version"]


async def test_every_schema_version_is_still_readable():
    """Reports already exist that cite v1 and v2 rows. Matching on one version
    string is what made the builder blind to a finished explanation."""
    assert is_explanation_schema("toxpred-explanation-v1")
    assert is_explanation_schema("toxpred-explanation-v2")
    assert is_explanation_schema(EXPLANATION_SCHEMA_VERSION)
    assert not is_explanation_schema("toxpred-prediction-v1")
    assert not is_explanation_schema(None)


# --- reuse ------------------------------------------------------------------

async def test_the_report_reuses_what_the_analysis_already_paid_for(db):
    """The XAI-01 headline. One backward pass, two consumers."""
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor()
    result = await _analyse(db, predictor, session_id, run_id)
    assert predictor.explained == [("herg", None)]

    outcome = await GetOrCreateExplanation(db, predictor).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )

    assert predictor.explained == [("herg", None)], "the explanation was recomputed"
    assert outcome.reused is True
    assert outcome.package.endpoint == "herg"
    assert outcome.package.highlights.positive_contributors


async def test_asking_twice_on_demand_calls_the_predictor_once(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor()
    result = await _analyse(db, predictor, session_id, run_id, targets=())
    assert predictor.explained == []

    service = GetOrCreateExplanation(db, predictor)
    first = await service.execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    second = await service.execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    assert predictor.explained == [("herg", None)]
    assert first.reused is False and second.reused is True
    assert second.observation.id == first.observation.id


async def test_an_explanation_is_never_reused_across_tox21_assays(db):
    """The twelve assays are independent measurements. Serving one's attribution
    for another would be a picture of the wrong question (SCI-09)."""
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor()
    result = await _analyse(db, predictor, session_id, run_id, targets=(("tox21", "NR-AR"),))

    outcome = await GetOrCreateExplanation(db, predictor).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="tox21", task="NR-AhR",
    )
    assert predictor.explained == [("tox21", "NR-AR"), ("tox21", "NR-AhR")]
    assert outcome.package.task == "NR-AhR"


async def test_new_weights_under_the_same_model_id_are_a_cache_miss(db):
    """K06: an id is a name a deployment chooses; a retrain replaces the weights
    behind it without the name moving."""
    keys = {
        weights: explanation_cache_key(
            canonical_smiles="CCO", endpoint="herg", task=None, model_id="m",
            artifact_fingerprint=(f"m:weights_sha256={weights}",),
        )
        for weights in ("aaa", "bbb")
    }
    assert keys["aaa"] != keys["bbb"]
    # And an unidentified artifact is its own key, never the pinned one's.
    assert explanation_cache_key(
        canonical_smiles="CCO", endpoint="herg", task=None, model_id="m",
    ) not in keys.values()


async def test_a_different_alignment_version_is_a_cache_miss():
    """Re-aligning the same attribution attaches different numbers to the same
    atom indices while leaving the payload byte-identical — drift a content hash
    cannot see."""
    common = dict(canonical_smiles="CCO", endpoint="herg", task=None, model_id="m")
    assert explanation_cache_key(**common, alignment_version="atom-alignment-v1") != (
        explanation_cache_key(**common, alignment_version="atom-alignment-v2")
    )


# --- the figure -------------------------------------------------------------

async def test_a_cached_payload_missing_its_figure_is_drawn_without_a_second_pass(db):
    """The expensive half is the backward pass. Repeating it to obtain a picture
    would also risk a figure that disagrees with the numbers it depicts."""
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(explain_depiction=SVG, signed_contributions=True)
    result = await _analyse(db, predictor, session_id, run_id)
    # The eager path has no object store, so it stored numbers and no figure.
    assert (await _attributions(db, result.snapshot.id))[0].provenance.get("figure") is None

    outcome = await GetOrCreateExplanation(
        db, predictor, InMemoryObjectStore()
    ).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )

    assert predictor.explained == [("herg", None)], "a figure cost a second backward pass"
    assert outcome.figure_only is True
    assert outcome.reused is True
    assert outcome.package.figure is not None
    assert outcome.package.figure.observation_id == outcome.observation.id
    # The numbers are the cached ones, not a recomputation's.
    assert outcome.observation.canonical_payload == (
        (await _attributions(db, result.snapshot.id))[0].canonical_payload
    )


async def test_the_figure_and_its_numbers_reach_the_report_tables_together(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(explain_depiction=SVG, signed_contributions=True)
    result = await _analyse(db, predictor, session_id, run_id, targets=())

    outcome = await GetOrCreateExplanation(
        db, predictor, InMemoryObjectStore()
    ).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    figure = outcome.package.figure
    assert figure is not None
    async with db.unit_of_work() as uow:
        row = await uow.reports.get_figure(figure.figure_id, session_id=session_id)
        attachment = await uow.attachments.get(
            figure.attachment_id, owner_id=ACTOR.subject_id
        )
    assert row is not None, "the figure metadata row was not written"
    assert attachment is not None, "the figure bytes were not attached"
    assert attachment.sha256 == figure.content_sha256


async def test_a_figure_failure_does_not_cost_the_numeric_contributors(db):
    """A refused depiction is a gap, never a reason to lose an explanation that
    is otherwise complete."""
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(
        explain_depiction="<not-svg>{{", signed_contributions=True
    )
    result = await _analyse(db, predictor, session_id, run_id, targets=())

    outcome = await GetOrCreateExplanation(
        db, predictor, InMemoryObjectStore()
    ).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    assert outcome.package.figure is None
    assert outcome.package.highlights.positive_contributors
    assert outcome.package.highlights.negative_contributors
    assert ExplanationUnavailable.FIGURE_STORE_FAILED in (
        outcome.package.failure_reason or ""
    )


async def test_no_object_store_is_a_named_absence_not_a_silent_one(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(explain_depiction=SVG)
    result = await _analyse(db, predictor, session_id, run_id, targets=())

    outcome = await GetOrCreateExplanation(db, predictor).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    assert outcome.package.figure is None
    assert outcome.package.failure_reason == ExplanationUnavailable.NO_OBJECT_STORE


# --- signs and failure classification ---------------------------------------

async def test_both_directions_survive_into_the_package(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(signed_contributions=True)
    result = await _analyse(db, predictor, session_id, run_id, targets=())

    outcome = await GetOrCreateExplanation(db, predictor).execute(
        owner_id=ACTOR.subject_id, session_id=session_id, run_id=run_id,
        analysis_id=result.snapshot.id, endpoint="herg",
    )
    positive = outcome.package.highlights.positive_contributors
    negative = outcome.package.highlights.negative_contributors
    assert positive and negative
    assert all(c["signed_contribution"] > 0 for c in positive)
    assert all(c["signed_contribution"] < 0 for c in negative)
    # Unmapped mass is carried verbatim, including a real zero.
    assert outcome.package.highlights.unmapped_importance == pytest.approx(0.35)


@pytest.mark.parametrize("reported,expected", [
    ("explanation_budget_exceeded", ExplanationUnavailable.BUDGET_EXCEEDED),
    ("TimeoutError", ExplanationUnavailable.BUDGET_EXCEEDED),
    ("token_attribution_unsupported", ExplanationUnavailable.ATTRIBUTION_UNSUPPORTED),
    ("ModelNotLoaded", ExplanationUnavailable.MODEL_UNAVAILABLE),
    ("token_offset_alignment_failed", ExplanationUnavailable.ALIGNMENT_FAILED),
])
def test_a_failure_reports_which_failure_it_was(reported, expected):
    """Five different causes lead to five different actions, and a single
    ``explanation_failed`` collapses them into a shrug."""
    assert classify_failure({"metadata": {"error": reported}}) == expected


def test_an_unrecognised_failure_is_not_given_an_invented_category():
    assert classify_failure({"metadata": {"error": "SomethingNew"}}) is None


def test_only_a_payload_with_real_numbers_counts_as_attribution():
    """A ``partial`` explanation with numbers is reusable. One with none is a
    failure wearing a softer word, and caching it would pin it in place."""
    assert has_numeric_attribution({"atoms": [{"signed_contribution": 0.1}]})
    assert has_numeric_attribution({"bonds": [{"relative_importance": 0.0}]})
    assert not has_numeric_attribution({"atoms": [], "bonds": []})
    assert not has_numeric_attribution({"atoms": [{"symbol": "C"}]})


async def test_a_failed_explanation_keeps_its_reason_through_a_rebuild(db):
    """``package_from_observation`` is the report validator's rebuild path, so a
    reason code that does not survive it is a gap the report cannot explain."""
    session_id, run_id = await _seed(db)
    result = await _analyse(db, _FailingPredictor(), session_id, run_id)

    observation = (await _attributions(db, result.snapshot.id))[0]
    package = package_from_observation(observation)
    assert package.status.value == "failed"
    assert package.figure is None
    assert package.endpoint == "herg"
    assert package.failure_reason


class _FailingPredictor(_CountingPredictor):
    async def explain(self, smiles, endpoint, task=None, *, model_id=None):
        self.explained.append((endpoint, task))
        raise RuntimeError("the explainer is unavailable")
