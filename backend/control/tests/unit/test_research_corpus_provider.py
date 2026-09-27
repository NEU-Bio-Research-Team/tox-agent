"""The pinned-corpus research provider, and SciFact through the product (W7-04).

Two things are tested here that a live run cannot tell apart from a bad model:
the ranking is deterministic and the corpus is the recorded one, and a relation
the product wrote becomes an abstract label only when it took a side.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from evals.external.scifact.data import Document
from evals.external.scifact.product import (
    abstract_label_only,
    corpus_doc_ids,
    corpus_lines,
    predictions_for_claim,
    product_problems,
    question_for,
    write_corpus,
)
from toxagent.config import ResearchSettings
from toxagent.research.providers import build_provider
from toxagent.research.providers.corpus import CorpusResearchProvider

pytestmark = pytest.mark.anyio

RECORDS = [
    {"record_id": "1", "title": "hERG block by terfenadine",
     "sentences": ["Terfenadine blocked the hERG channel.", "The IC50 was 56 nM."]},
    {"record_id": "2", "title": "Aspirin and platelet aggregation",
     "sentences": ["Aspirin inhibited platelet aggregation."]},
    {"record_id": "3", "title": "A review of potassium channels",
     "sentences": ["Potassium channels matter.", "hERG is one of them."]},
]


def _corpus(tmp_path, records=RECORDS, name="corpus.jsonl"):
    path = tmp_path / name
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_the_corpus_must_be_the_pinned_one(tmp_path):
    path, digest = _corpus(tmp_path)
    CorpusResearchProvider(path, sha256=digest)  # the pin it was built with
    with pytest.raises(ValueError, match="not the pinned"):
        CorpusResearchProvider(path, sha256="0" * 64)


def test_a_corpus_without_a_pin_is_refused(tmp_path):
    path, _ = _corpus(tmp_path)
    with pytest.raises(ValueError, match="SHA256 is required"):
        CorpusResearchProvider(path, sha256="  ")


def test_a_typo_in_the_builder_is_refused_not_ignored(tmp_path):
    path, digest = _corpus(tmp_path, [{"record_id": "1", "title": "t", "sentance": ["oops"]}])
    with pytest.raises(ValueError, match="unknown field"):
        CorpusResearchProvider(path, sha256=digest)


def test_a_duplicate_record_id_is_refused(tmp_path):
    path, digest = _corpus(tmp_path, [RECORDS[0], RECORDS[0]])
    with pytest.raises(ValueError, match="duplicate record_id"):
        CorpusResearchProvider(path, sha256=digest)


async def test_ranking_puts_the_matching_abstract_first_and_bounds_the_result(tmp_path):
    path, digest = _corpus(tmp_path)
    provider = CorpusResearchProvider(path, sha256=digest)
    hits = await provider.search(query="terfenadine hERG IC50", source_types=None,
                                 date_from=None, limit=2)
    assert [hit.provider_record_id for hit in hits] == ["1", "3"]
    assert hits[0].normalized_facts["rank"] == 1
    assert hits[0].normalized_facts["corpus_record_id"] == "1"
    # Every hit carries the pin, so an evidence record can be traced to the
    # corpus that served it long after the run.
    assert hits[0].raw["corpus_sha256"] == digest
    assert "IC50 was 56 nM" in (hits[0].abstract_or_excerpt or "")


async def test_a_query_no_record_matches_returns_nothing(tmp_path):
    path, digest = _corpus(tmp_path)
    provider = CorpusResearchProvider(path, sha256=digest)
    assert await provider.search(query="zebrafish", source_types=None,
                                 date_from=None, limit=5) == []


async def test_ties_are_broken_by_record_id_so_a_rerun_ranks_the_same(tmp_path):
    same = [{"record_id": rid, "title": "identical", "sentences": ["same text"]}
            for rid in ("9", "3", "11")]
    path, digest = _corpus(tmp_path, same)
    provider = CorpusResearchProvider(path, sha256=digest)
    first = await provider.search(query="identical same text", source_types=None,
                                  date_from=None, limit=3)
    second = await provider.search(query="identical same text", source_types=None,
                                   date_from=None, limit=3)
    ids = [hit.provider_record_id for hit in first]
    assert ids == [hit.provider_record_id for hit in second] == ["11", "3", "9"]


def test_the_factory_and_the_effective_product_name_the_corpus(tmp_path):
    from toxagent.application.effective_product import describe_effective_product
    from toxagent.config import (
        CompoundSettings, OcrSettings, PolicySettings, PredictorSettings, PredictSettings,
        RuntimeSettings, SecuritySettings, Settings,
    )

    path, digest = _corpus(tmp_path)
    research = ResearchSettings(provider="corpus", corpus_path=str(path), corpus_sha256=digest)
    provider = build_provider(research)
    assert isinstance(provider, CorpusResearchProvider)
    assert provider.corpus_size == len(RECORDS)
    settings = Settings(
        database_url="sqlite+aiosqlite:///x.db", predictor=PredictorSettings(),
        policy=PolicySettings(), predict=PredictSettings(), runtime=RuntimeSettings(),
        research=research, compound=CompoundSettings(), ocr=OcrSettings(),
        security=SecuritySettings(),
    )
    providers = describe_effective_product(settings)["providers"]
    assert providers["research_provider"] == "corpus"
    assert providers["research_corpus"] == {"path_name": path.name, "sha256": digest}


# --- SciFact through the product ---------------------------------------------


def _relation(source_id: str, relation: str, source_class: str = "external_experimental"):
    return {"source_ref": {"source_class": source_class, "source_id": source_id},
            "relation": relation}


def test_only_a_stance_becomes_an_abstract_label():
    doc_ids = {"evd_a": 11, "evd_b": 12, "evd_c": 13}
    relations = [
        _relation("evd_a", "supports"),
        _relation("evd_b", "contradicts"),
        _relation("evd_c", "contextual"),
    ]
    predictions, skipped = predictions_for_claim(relations, doc_ids)
    assert predictions == {11: {"label": "SUPPORT", "sentences": []},
                           12: {"label": "CONTRADICT", "sentences": []}}
    assert skipped == [{"reason": "no_stance", "doc_id": 13, "relation": "contextual"}]


def test_a_predictor_fact_is_not_a_prediction_about_an_abstract():
    predictions, skipped = predictions_for_claim(
        [_relation("obs_1", "supports", "predictor_fact")], {"evd_a": 11}
    )
    assert predictions == {} and skipped == []


def test_a_source_outside_the_corpus_is_recorded_not_labelled():
    predictions, skipped = predictions_for_claim([_relation("evd_z", "supports")], {"evd_a": 11})
    assert predictions == {}
    assert skipped == [{"reason": "source_not_in_corpus", "source_id": "evd_z",
                        "relation": "supports"}]


def test_two_propositions_disagreeing_about_one_abstract_stay_unlabelled():
    relations = [_relation("evd_a", "supports"), _relation("evd_a", "contradicts")]
    predictions, skipped = predictions_for_claim(relations, {"evd_a": 11})
    assert predictions == {}
    assert skipped == [{"reason": "conflicting_stances", "doc_id": 11,
                        "labels": ["CONTRADICT", "SUPPORT"]}]


def test_the_same_stance_twice_is_one_label():
    relations = [_relation("evd_a", "supports"), _relation("evd_a", "supports")]
    predictions, skipped = predictions_for_claim(relations, {"evd_a": 11})
    assert predictions == {11: {"label": "SUPPORT", "sentences": []}} and skipped == []


def test_evidence_records_map_back_to_corpus_doc_ids():
    evidence = [
        {"evidence_id": "evd_a", "normalized_facts": {"corpus_record_id": "11"}},
        {"evidence_id": "evd_b", "normalized_facts": {"corpus": "x"}},
        {"evidence_id": "evd_c", "normalized_facts": {"corpus_record_id": "not-a-number"}},
    ]
    assert corpus_doc_ids(evidence) == {"evd_a": 11}


def test_only_the_defined_metric_is_computed():
    gold = {1: {11: {"label": "SUPPORT", "rationales": [[0]]}},
            2: {12: {"label": "CONTRADICT", "rationales": [[3]]}}}
    predictions = {1: {11: {"label": "SUPPORT", "sentences": []}}, 2: {}}
    result = abstract_label_only(predictions, gold)
    assert result["abstract_label_only"]["precision"] == 1.0
    assert result["abstract_label_only"]["recall"] == 0.5
    assert set(result["not_computed"]) == {
        "sentence_selection", "sentence_label", "abstract_rationalized"
    }
    assert "abstract_rationalized" not in result


def test_a_claim_with_no_prediction_still_counts_against_recall():
    gold = {1: {11: {"label": "SUPPORT", "rationales": [[0]]}}}
    with pytest.raises(ValueError, match="must cover exactly"):
        abstract_label_only({}, gold)


def test_the_question_adds_nothing_the_product_could_copy():
    question = question_for("Terfenadine blocks hERG")
    assert "Terfenadine blocks hERG" in question
    for leak in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO", "corpus"):
        assert leak not in question


def test_a_deployment_that_cannot_measure_this_is_refused_before_any_claim_runs():
    live = {"providers": {"research_provider": "europepmc"},
            "flags": {"answer_draft_v2": {"enabled": False},
                      "subjectless_research_v1": {"enabled": False}}}
    problems = product_problems(live, corpus_sha256="a" * 64)
    assert len(problems) == 3
    assert any("europepmc" in p for p in problems)
    assert any("answer_draft_v2" in p for p in problems)
    assert any("subjectless_research_v1" in p for p in problems)

    wrong_corpus = {
        "providers": {"research_provider": "corpus", "research_corpus": {"sha256": "b" * 64}},
        "flags": {"answer_draft_v2": {"enabled": True},
                  "subjectless_research_v1": {"enabled": True}},
    }
    assert [p for p in product_problems(wrong_corpus, corpus_sha256="a" * 64) if "sha256" in p]

    ready = {
        "providers": {"research_provider": "corpus", "research_corpus": {"sha256": "a" * 64}},
        "flags": {"answer_draft_v2": {"enabled": True},
                  "subjectless_research_v1": {"enabled": True}},
    }
    assert product_problems(ready, corpus_sha256="a" * 64) == []


def test_the_corpus_file_depends_only_on_the_release(tmp_path):
    corpus = {
        12: Document(12, "Second", ("b",)),
        11: Document(11, "First", ("a", "c")),
    }
    digest = write_corpus(corpus, tmp_path / "corpus.jsonl")
    lines = (tmp_path / "corpus.jsonl").read_text().splitlines()
    assert [json.loads(line)["record_id"] for line in lines] == ["11", "12"]
    assert digest == hashlib.sha256(
        ("\n".join(corpus_lines(corpus)) + "\n").encode("utf-8")
    ).hexdigest()
    # The file the provider is pinned to is exactly the file it can load.
    provider = CorpusResearchProvider(tmp_path / "corpus.jsonl", sha256=digest)
    assert provider.corpus_size == 2
