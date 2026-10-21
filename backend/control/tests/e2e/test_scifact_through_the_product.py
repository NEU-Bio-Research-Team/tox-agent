"""W7-04 end to end: a claim with no molecule, a pinned corpus, abstract labels.

The transfer study (`evals/external/scifact/product.py`) needs three things
from the product at once: the corpus provider must serve a ranked local corpus,
a question naming no molecule must reach the evidence tools
(`subjectless_research_v1`), and the relations of the accepted answer must be
readable so an abstract label can be derived. Any one of them failing in a live
run would look like the model doing badly, so the three are tested together on
the scripted runtime.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from evals.external.scifact.product import (
    corpus_doc_ids,
    predictions_for_claim,
    question_for,
    write_corpus,
)
from evals.external.scifact.data import Document
from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor
from toxagent.platform.config import ResearchSettings
from toxagent.research.providers.corpus import CorpusResearchProvider

pytestmark = pytest.mark.anyio

CLAIM = "Aspirin reduces platelet aggregation in healthy volunteers."

CORPUS = {
    11: Document(11, "Aspirin and platelet aggregation in healthy volunteers",
                 ("Aspirin reduced platelet aggregation in healthy volunteers.",
                  "The effect persisted for 24 hours.")),
    12: Document(12, "No effect of aspirin on platelet aggregation",
                 ("Aspirin did not reduce platelet aggregation in this cohort.",)),
    13: Document(13, "Zebrafish cardiac development",
                 ("Cardiac development in zebrafish follows a fixed programme.",)),
}


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SUBJECTLESS_RESEARCH_V1", "1")


def _corpus_provider(tmp_path):
    digest = write_corpus(CORPUS, tmp_path / "corpus.jsonl")
    return CorpusResearchProvider(tmp_path / "corpus.jsonl", sha256=digest), digest


#: What each source itself states, and nothing more. Doc 11 names its
#: population; doc 12 says only "this cohort", so its relation leaves the
#: population out rather than borrowing doc 11's (a scope filled in to match
#: is how a false contradiction gets through).
SUPPORTING_SCOPE = {"endpoint": "platelet aggregation", "ligand_id": "aspirin",
                    "experimental_context": "healthy volunteers"}
CONTRADICTING_SCOPE = {"endpoint": "platelet aggregation", "ligand_id": "aspirin"}


def _draft(supporting: str, contradicting: str, *, supporting_scope: dict,
           contradicting_scope: dict) -> dict:
    return {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": (
            "One retrieved study reports reduced platelet aggregation; another reports "
            "no reduction in its cohort."
        ),
        "claims": [{
            "local_ref": "lit", "kind": "scientific",
            "text": "Retrieved studies disagree on whether aspirin reduced platelet "
                    "aggregation.",
            "citation_ids": [supporting, contradicting],
        }],
        "limitations": [{"code": "evidence_scope_limited", "text": ""}],
        "evidence_relations": [
            {"proposition": CLAIM, "source_class": "external_experimental",
             "source_id": supporting, "relation": "supports",
             "reason_codes": ["same_population"], **supporting_scope},
            {"proposition": CLAIM, "source_class": "external_experimental",
             "source_id": contradicting, "relation": "contradicts",
             "reason_codes": ["opposite_finding"], **contradicting_scope},
        ],
    }


def _script(seen: dict, attempts: list[tuple[dict, dict]]):
    """Search, read every hit, then submit one draft per (supporting scope,
    contradicting scope) in ``attempts``, keeping each submission's result."""

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        # No analysis_id: the question names no molecule of this session.
        found = await turn.call_tool("search_toxicology_evidence", {
            "query": "aspirin platelet aggregation healthy volunteers", "limit": 5,
        })
        seen["results"].append(found)
        results = found["model_view"]["results"]
        by_record = {}
        for result in results:
            record = await turn.call_tool("get_evidence_record",
                                          {"evidence_id": result["evidence_id"]})
            by_record[record["model_view"]["normalized_facts"]["corpus_record_id"]] = (
                result["evidence_id"]
            )
        seen["by_record"] = by_record
        for supporting_scope, contradicting_scope in attempts:
            submitted = await turn.call_tool("submit_grounded_answer", _draft(
                by_record["11"], by_record["12"],
                supporting_scope=supporting_scope, contradicting_scope=contradicting_scope,
            ))
            seen["results"].append(submitted)

    return script


async def _ask(client, db, script) -> tuple[str, dict]:
    await _install_scripted_runtime(client, script)
    session_id = await _new_session(client)
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": "auto",
              "content": [{"type": "text", "text": question_for(CLAIM)}]},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    run = await wait_for_run(client, session_id, submitted.json()["run_id"])
    return session_id, run


def _config(tmp_path, digest):
    return settings(research=ResearchSettings(
        provider="corpus", corpus_path=str(tmp_path / "corpus.jsonl"), corpus_sha256=digest,
    ))


async def test_a_claim_becomes_abstract_labels_the_study_can_score(db, flags_on, tmp_path):
    provider, digest = _corpus_provider(tmp_path)
    seen: dict = {"prompt": "", "results": []}
    script = _script(seen, [(SUPPORTING_SCOPE, CONTRADICTING_SCOPE)])

    async with api_client(db, StubPredictor(), config=_config(tmp_path, digest),
                          research_provider=provider) as client:
        session_id, run = await _ask(client, db, script)
        assert run["status"] == "completed", run
        # The claim named no molecule and was still answered from the literature.
        assert run["intent"] == "decision_support"
        assert "no prediction to read" in seen["prompt"]
        assert seen["results"][-1]["status"] == "completed", seen["results"][-1]

        relations = await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/evidence-relations", headers=AUTH
        )
        assert relations.status_code == 200, relations.text
        payload = relations.json()["evidence_relations"]
        assert len(payload) == 2
        # One proposition text, one server-minted proposition id.
        assert len({r["proposition_id"] for r in payload}) == 1
        # Each scope is what its source stated: doc 12's population stays unstated.
        scopes = {r["relation"]: r["scope"] for r in payload}
        assert scopes["supports"]["experimental_context"] == "healthy volunteers"
        assert scopes["contradicts"]["experimental_context"] is None

        evidence = await client.get(f"/v1/sessions/{session_id}/evidence",
                                    params={"status": "all", "limit": 200}, headers=AUTH)
        doc_ids = corpus_doc_ids(evidence.json()["evidence"])
        # Every corpus record the search returned is traceable to its doc id,
        # including the irrelevant one the ranker also returned.
        assert set(doc_ids.values()) >= {11, 12}
        predictions, skipped = predictions_for_claim(payload, doc_ids)
        assert predictions == {11: {"label": "SUPPORT", "sentences": []},
                               12: {"label": "CONTRADICT", "sentences": []}}
        assert skipped == []

        # The deployment describes itself as serving that corpus, so a recorded
        # number cannot be read as a live literature run.
        product = await client.get("/v1/system/effective-product", headers=AUTH)
        corpus = (product.json()["providers"])["research_corpus"]
        assert corpus == {"path_name": "corpus.jsonl", "sha256": digest}


async def test_a_contradiction_with_no_scope_is_refused_until_the_sources_scope_is_stated(
    db, flags_on, tmp_path,
):
    """The scope guard through the product: a supports and a contradicts with
    nothing stated about either source is not yet a disagreement, so it is a
    typed, correctable violation naming what to state -- not a failed run, and
    not a silent downgrade. Stating what the sources say gets it accepted."""
    provider, digest = _corpus_provider(tmp_path)
    seen: dict = {"prompt": "", "results": []}
    script = _script(seen, [({}, {}), (SUPPORTING_SCOPE, CONTRADICTING_SCOPE)])

    async with api_client(db, StubPredictor(), config=_config(tmp_path, digest),
                          research_provider=provider) as client:
        session_id, run = await _ask(client, db, script)

        first, retried = seen["results"][-2:]
        assert first["status"] == "error", first
        assert first["error"]["code"] == "answer_validation_failed"
        violations = first["error"]["details"]["violations"]
        assert [v["code"] for v in violations] == [
            "evidence_relation_contradiction_scope_not_assessed"
        ]
        assert violations[0]["path"] == "evidence_relations[1].relation"
        assert "in which molecule or system" in violations[0]["message"]
        assert "do not fill a scope in" in violations[0]["message"]

        assert retried["status"] == "completed", retried
        assert retried["model_view"]["is_fallback"] is False
        assert run["status"] == "completed", run

        relations = await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/evidence-relations", headers=AUTH
        )
        assert sorted(r["relation"] for r in relations.json()["evidence_relations"]) == [
            "contradicts", "supports",
        ]


async def test_the_corpus_the_deployment_serves_is_the_file_it_is_pinned_to(tmp_path):
    """The study compares the deployment's reported hash with the one it built;
    editing the corpus after pinning it must not go unnoticed."""
    _, digest = _corpus_provider(tmp_path)
    path = tmp_path / "corpus.jsonl"
    path.write_bytes(path.read_bytes() + json.dumps(
        {"record_id": "14", "title": "Added later", "sentences": ["Not in the pin."]}
    ).encode("utf-8") + b"\n")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != digest
    with pytest.raises(ValueError, match="not the pinned"):
        CorpusResearchProvider(path, sha256=digest)
