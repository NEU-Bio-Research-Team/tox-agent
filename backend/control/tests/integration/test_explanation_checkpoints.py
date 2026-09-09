"""A bundle must not pay twice for work it already finished.

I19's trigger is a crash partway through a multi-target explanation bundle.
Everything was computed before anything was committed, so five completed
targets out of eight were discarded along with the three that had not run, and
the retry recomputed all eight. Separately, each predictor call had a timeout
and the sequence of them had none, so a bundle could outlast the run deadline
while every individual call looked healthy.

The crash is modelled by a predictor that raises on a chosen target — the
in-process equivalent of the process going away mid-bundle, since either way
the snapshot transaction never commits.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")
TARGETS = (("herg", None), ("tox21", "NR-AR"), ("tox21", "NR-AhR"))


class _CountingPredictor:
    """A real client, wrapped to record explain calls and disturb one target.

    Delegating everything else keeps the prediction path genuine — the
    canonical SMILES and the resolved model that the checkpoint key is built
    from come from the same code the product runs.
    """

    def __init__(
        self, *, fail_on: str | None = None, stall_on: str | None = None,
        weights_sha256: str | None = None,
    ) -> None:
        self._client = StubPredictor(weights_sha256=weights_sha256).client()
        self.explained: list[tuple[str, str | None]] = []
        self._fail_on = fail_on
        self._stall_on = stall_on

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def explain(self, smiles, endpoint, task=None, *, model_id=None):
        self.explained.append((endpoint, task))
        if self._fail_on == (task or endpoint):
            raise RuntimeError("the predictor went away mid-bundle")
        if self._stall_on == (task or endpoint):
            await asyncio.sleep(30)
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


async def _run(db, predictor, session_id, run_id, *, budget_s: float = 120.0):
    settings = PolicySettings(explanation_budget_s=budget_s)
    return await CreateAnalysis(db, predictor, settings).execute(
        actor=ACTOR, session_id=session_id, run_id=run_id, smiles="CCO",
        endpoints=("herg", "tox21"), explanation_mode="required",
        explanation_targets=TARGETS,
    )


async def test_a_retry_after_a_mid_bundle_failure_only_does_the_missing_target(db):
    session_id, run_id = await _seed(db)

    first = _CountingPredictor(fail_on="NR-AhR")
    await _run(db, first, session_id, run_id)
    assert first.explained == [("herg", None), ("tox21", "NR-AR"), ("tox21", "NR-AhR")]

    # The retry: a fresh predictor, so what it is asked for is exactly what
    # was not already checkpointed.
    session_id_2, run_id_2 = await _seed(db)
    retry = _CountingPredictor()
    result = await _run(db, retry, session_id_2, run_id_2)

    assert retry.explained == [("tox21", "NR-AhR")], (
        "the two targets that succeeded were recomputed"
    )
    async with db.unit_of_work() as uow:
        observations = await uow.observations.list_for_analysis(result.snapshot.id)
    attributions = [o for o in observations if o.kind.value == "attribution"]
    assert len(attributions) == 3, "the reused targets must still reach the bundle"


async def test_a_failed_target_is_labelled_not_omitted_and_not_checkpointed(db):
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(fail_on="NR-AhR")

    result = await _run(db, predictor, session_id, run_id)

    async with db.unit_of_work() as uow:
        observations = await uow.observations.list_for_analysis(result.snapshot.id)
    attributions = [o for o in observations if o.kind.value == "attribution"]
    assert len(attributions) == 3, "a failed target must be recorded, not dropped"
    failed = [o for o in attributions if o.canonical_payload.get("status") == "failed"]
    assert len(failed) == 1
    assert failed[0].canonical_payload["metadata"]["error"] == "RuntimeError"
    assert failed[0].canonical_payload["atoms"] == [], (
        "a failed explanation must carry no attribution"
    )


async def test_the_whole_bundle_is_bounded_even_when_each_call_looks_healthy(db):
    """One stalled target must not let the bundle run past its budget."""
    session_id, run_id = await _seed(db)
    predictor = _CountingPredictor(stall_on="NR-AR")

    started = asyncio.get_running_loop().time()
    result = await _run(db, predictor, session_id, run_id, budget_s=0.4)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 10, f"the bundle ran for {elapsed:.1f}s against a 0.4s budget"
    async with db.unit_of_work() as uow:
        observations = await uow.observations.list_for_analysis(result.snapshot.id)
    attributions = [o for o in observations if o.kind.value == "attribution"]
    reasons = [
        o.canonical_payload.get("metadata", {}).get("error") for o in attributions
    ]
    assert reasons.count("explanation_budget_exceeded") >= 1, reasons
    # And the target that ran before the budget was spent is still good.
    assert any(o.canonical_payload.get("status") != "failed" for o in attributions)


async def test_a_checkpoint_is_not_reused_across_a_different_model(db):
    """An explanation is of a molecule *by a model*; the other model's
    attribution is not an answer about this one (I11)."""
    from toxagent.application.create_analysis import explanation_checkpoint_key

    a = explanation_checkpoint_key(
        canonical_smiles="CCO", endpoint="herg", task=None, model_id="model-a"
    )
    b = explanation_checkpoint_key(
        canonical_smiles="CCO", endpoint="herg", task=None, model_id="model-b"
    )
    assert a != b


def test_a_checkpoint_is_not_reused_across_different_weights_under_one_id():
    """K06 asks for the pin to be the model *hash*, not the model name.

    A model id is a name the deployment chooses; a retrain, a re-download or a
    corrected checkpoint replaces the weights behind it without the name
    moving. Keyed on the name alone, the bundle would serve the old weights'
    attribution as the new model's — the I11 mismatch again, arriving through
    time instead of through a second model.
    """
    from toxagent.application.create_analysis import explanation_checkpoint_key

    def key(*fingerprint: str) -> str:
        return explanation_checkpoint_key(
            canonical_smiles="CCO", endpoint="herg", task=None,
            model_id="herg-tox21-chemberta-v1", artifact_fingerprint=fingerprint,
        )

    retrained = key("herg-tox21-chemberta-v1:weights_sha256=aaa")
    assert retrained != key("herg-tox21-chemberta-v1:weights_sha256=bbb")
    # A tokenizer swap changes what the tokens mean, so it changes the
    # explanation even when the weights are byte-identical.
    assert retrained != key(
        "herg-tox21-chemberta-v1:weights_sha256=aaa",
        "herg-tox21-chemberta-v1:tokenizer_sha256=ccc",
    )
    # And an unidentified artifact is its own key, never the pinned one's:
    # a checkpoint written when the weights were unknown must not be handed
    # back as though it had been pinned to them.
    assert retrained != key()


def test_the_fingerprint_selects_only_the_model_that_answered():
    """Per-model, so retraining one admitted model leaves the other's
    explanations valid rather than invalidating the whole cache."""
    from toxagent.application.create_analysis import model_artifact_fingerprint

    class _Provenance:
        artifact_hashes = (
            "model-b:weights_sha256=bbb",
            "model-a:weights_sha256=aaa",
            "model-a:tokenizer_sha256=ttt",
        )

    assert model_artifact_fingerprint(_Provenance(), "model-a") == (
        "model-a:tokenizer_sha256=ttt", "model-a:weights_sha256=aaa",
    )
    assert model_artifact_fingerprint(_Provenance(), "model-b") == (
        "model-b:weights_sha256=bbb",
    )
    assert model_artifact_fingerprint(_Provenance(), None) == ()


async def test_new_weights_under_the_same_id_are_explained_again(db):
    """The end-to-end version: same molecule, same model id, new checksum.

    The stub reports provenance the way ToxPred does — per model, with the id
    the prediction sections carry — so this exercises the same correlation the
    product does rather than a hand-built key.
    """
    session_id, run_id = await _seed(db)

    first = _CountingPredictor()
    await _run(db, first, session_id, run_id)
    assert len(first.explained) == len(TARGETS)

    # Nothing about the molecule or the request changed, so a rerun must reuse.
    reused = _CountingPredictor()
    await _run(db, reused, session_id, run_id)
    assert reused.explained == []

    # Now the weights behind that id are replaced.
    retrained = _CountingPredictor(weights_sha256="f" * 64)
    await _run(db, retrained, session_id, run_id)
    assert retrained.explained == list(TARGETS)
