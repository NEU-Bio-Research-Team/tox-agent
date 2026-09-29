"""SciFact adapter: the official metrics port, retrieval, judge parsing, runner (Wave 7)."""
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from evals.external.scifact import judges, retrieval
from evals.external.scifact.data import Claim, Document, gold_of
from evals.external.scifact.metrics import compute_metrics
from evals.external.scifact.run import run, select

pytestmark = pytest.mark.anyio


def _close(value: float, fraction: Fraction) -> bool:
    return abs(value - float(fraction)) < 1e-12


def test_the_official_worked_example_scores_as_documented():
    """doc/evaluation.md, 'Detailed scoring example' (claim 52)."""
    gold = {52: {11: {"label": "SUPPORT", "rationales": [[0, 1], [11]]},
                 15: {"label": "SUPPORT", "rationales": [[4]]}}}
    predictions = {52: {11: {"label": "SUPPORT", "sentences": [1, 11, 13]},
                        16: {"label": "CONTRADICT", "sentences": [18, 20]}}}
    m = compute_metrics(predictions, gold)
    for key in ("precision", "recall", "f1"):
        assert _close(m["abstract_rationalized"][key], Fraction(1, 2))
    assert _close(m["sentence_selection"]["precision"], Fraction(1, 5))
    assert _close(m["sentence_selection"]["recall"], Fraction(1, 4))
    assert _close(m["sentence_selection"]["f1"], Fraction(2, 9))


def test_only_the_first_three_sentences_count_at_abstract_level():
    gold = {1: {7: {"label": "SUPPORT", "rationales": [[9]]}}}
    late = {1: {7: {"label": "SUPPORT", "sentences": [0, 1, 2, 9]}}}
    m = compute_metrics(late, gold)
    assert m["abstract_label_only"]["f1"] == 1.0
    assert m["abstract_rationalized"]["f1"] == 0.0
    # Sentence level counts every predicted sentence.
    assert _close(m["sentence_selection"]["precision"], Fraction(1, 4))


def test_a_wrong_label_scores_nothing_at_either_level_that_needs_the_label():
    gold = {1: {7: {"label": "SUPPORT", "rationales": [[2]]}}}
    wrong = {1: {7: {"label": "CONTRADICT", "sentences": [2]}}}
    m = compute_metrics(wrong, gold)
    assert m["abstract_label_only"]["f1"] == 0.0
    assert m["sentence_selection"]["f1"] == 1.0  # selection ignores the label
    assert m["sentence_label"]["f1"] == 0.0


def test_nei_predictions_are_ignored_and_empty_claims_still_count_in_recall():
    gold = {1: {7: {"label": "SUPPORT", "rationales": [[2]]}}, 2: {}}
    predictions = {1: {7: {"label": "NOT_ENOUGH_INFO", "sentences": [2]}}, 2: {}}
    m = compute_metrics(predictions, gold)
    assert m["counts"]["abstract"].get("retrieved", 0) == 0
    assert m["abstract_label_only"]["recall"] == 0.0
    assert m["counts"]["abstract"]["relevant"] == 1


def test_every_claim_needs_a_prediction_line():
    gold = {1: {}, 2: {}}
    with pytest.raises(ValueError, match="missing"):
        compute_metrics({1: {}}, gold)


def test_oracle_retrieval_matches_the_official_script():
    claims = [
        Claim(1, "c", {7: {"label": "SUPPORT", "rationales": [[0]]}}, (7, 8)),
        Claim(2, "nei", {}, (9, 10)),
    ]
    assert retrieval.oracle(claims) == {1: [7], 2: [9]}
    assert retrieval.oracle(claims, include_nei=False) == {1: [7], 2: []}


def test_tfidf_ranks_the_matching_abstract_first():
    # The official retriever is scikit-learn's, an eval-time import the control
    # plane itself does not depend on.
    pytest.importorskip("sklearn")
    corpus = {1: Document(1, "Aspirin and platelets", ("Aspirin inhibits platelet aggregation.",)),
              2: Document(2, "Soil bacteria", ("Nitrogen fixation in legumes.",))}
    claims = [Claim(1, "Aspirin inhibits platelet aggregation", {}, ())]
    assert retrieval.tfidf(claims, corpus, k=1) == {1: [1]}


@pytest.mark.parametrize("text, expected", [
    ('{"label": "SUPPORT", "sentences": [1, 2, 99, "x", 1]}', ("SUPPORT", [1, 2], None)),
    ('Here you go: {"label": "contradict", "sentences": []} done', ("CONTRADICT", [], None)),
    ("no json", ("NOT_ENOUGH_INFO", [], "no JSON object in the output")),
    ('{"label": "MAYBE"}', ("NOT_ENOUGH_INFO", [], "unknown label 'MAYBE'")),
])
def test_judge_output_is_parsed_strictly(text, expected):
    assert judges.parse(text, 5) == expected


def test_the_prompt_numbers_sentences_from_zero():
    text = judges.render(Claim(1, "A claim.", {}, ()), Document(3, "T", ("First.", "Second.")))
    assert "[0] First.\n[1] Second." in text and "Claim: A claim." in text


def test_a_subset_is_deterministic_for_a_seed():
    claims = [Claim(i, "c", {}, ()) for i in range(20)]
    assert [c.id for c in select(claims, limit=5, seed=1)] == [c.id for c in select(claims, limit=5, seed=1)]
    assert len(select(claims, limit=None, seed=1)) == 20


class FakeJudge:
    name = "fake:judge"

    def describe(self):
        return {"judge": "fake"}

    async def judge(self, claim, document):
        if document.doc_id == 99:
            raise RuntimeError("provider down")
        label = "SUPPORT" if claim.id == 1 else "NOT_ENOUGH_INFO"
        return judges.Judgment(label, [0], '{"label": "%s"}' % label, model={"model_id_resolved": "fake-1"})


async def test_the_runner_writes_the_official_format_and_counts_failures(tmp_path):
    corpus = {7: Document(7, "t", ("s0", "s1")), 8: Document(8, "t", ("s0",)),
              99: Document(99, "t", ("s0",))}
    claims = [Claim(1, "c1", {7: {"label": "SUPPORT", "rationales": [[0]]}}, (7,)),
              Claim(2, "c2", {}, (8,)), Claim(3, "c3", {}, (99,))]
    result = await run(claims=claims, corpus=corpus, retrieved=retrieval.oracle(claims),
                       judge=FakeJudge(), out=tmp_path, parallel=2)
    lines = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert lines == [{"id": 1, "evidence": {"7": {"label": "SUPPORT", "sentences": [0]}}},
                     {"id": 2, "evidence": {}}, {"id": 3, "evidence": {}}]
    statuses = sorted(j["status"] for j in result["judgments"])
    assert statuses == ["error", "ok", "ok"]
    metrics = compute_metrics(result["predictions"], gold_of(claims))
    assert metrics["abstract_rationalized"]["f1"] == 1.0
