"""SciFact's official metrics, ported without pandas.

Port of ``verisci/evaluate/lib/metrics.py`` from https://github.com/allenai/scifact
at commit 68b98a56d93e0f9da0d2aab4e6c3294699a0f72e (Apache License 2.0,
Copyright the SciFact authors). The logic is unchanged: the same four metrics,
the same three-sentence cap at abstract level, NEI predictions ignored, and a
sentence credited only when its whole gold rationale was selected.
``tests/unit/test_scifact.py`` checks the port against hand-worked cases for
every branch; the port was also checked against the official module on random
predictions over the dev split when it was written (recorded in the README).

Data shapes (the official files, parsed):

* gold: ``{claim_id: {doc_id: {"label": "SUPPORT"|"CONTRADICT", "rationales": [[sent, ...], ...]}}}``
* predictions: ``{claim_id: {doc_id: {"label": "SUPPORT"|"CONTRADICT", "sentences": [sent, ...]}}}``
  — the official prediction format has no NEI; an abstract judged NEI is
  simply absent, and an explicit ``NOT_ENOUGH_INFO`` is skipped the same way.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

MAX_ABSTRACT_SENTS = 3
NEI = "NOT_ENOUGH_INFO"
LABELS = ("SUPPORT", "CONTRADICT")

Gold = Mapping[int, Mapping[int, Mapping[str, Any]]]
Predictions = Mapping[int, Mapping[int, Mapping[str, Any]]]


def safe_divide(num: float, denom: float) -> float:
    return 0 if denom == 0 else num / denom


def compute_f1(counts: Mapping[str, int], difficulty: str | None = None) -> dict[str, float]:
    correct_key = "correct" if difficulty is None else f"correct_{difficulty}"
    precision = safe_divide(counts[correct_key], counts["retrieved"])
    recall = safe_divide(counts[correct_key], counts["relevant"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def contains_evidence(predicted: set[int], gold: list[set[int]]) -> bool:
    return any(gold_rationale.issubset(predicted) for gold_rationale in gold)


def _abstract_correct(doc_id: int, doc_pred: Mapping[str, Any],
                      gold_claim: Mapping[int, Mapping[str, Any]]) -> tuple[bool, bool]:
    pred_rationales = list(doc_pred["sentences"])[:MAX_ABSTRACT_SENTS]
    if doc_id not in gold_claim:
        return False, False
    if doc_pred["label"] != gold_claim[doc_id]["label"]:
        return False, False
    gold_rationales = [set(r) for r in gold_claim[doc_id]["rationales"]]
    return True, contains_evidence(set(pred_rationales), gold_rationales)


def _count_rationale_sents(predicted: set[int], gold: list[set[int]]) -> int:
    n_correct = 0
    for index in predicted:
        gold_sets = [entry for entry in gold if index in entry]
        assert len(gold_sets) < 2  # a sentence cannot be in two rationales
        if not gold_sets:
            continue
        if gold_sets[0].issubset(predicted):
            n_correct += 1
    return n_correct


def compute_metrics(predictions: Predictions, gold: Gold) -> dict[str, dict[str, float]]:
    """The official four metrics over every claim in ``gold``.

    The official evaluator iterates the *prediction* file, so a claim left out
    of it silently drops out of recall. Every gold claim must therefore have a
    prediction entry (an empty one for "no evidence"), which makes iterating
    the gold claims here identical to the official loop.
    """
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing or extra:
        raise ValueError(
            f"predictions must cover exactly the gold claims; missing {missing[:5]}, extra {extra[:5]}"
        )
    abstract: Counter = Counter()
    sentence: Counter = Counter()
    for claim_id, gold_claim in gold.items():
        abstract["relevant"] += len(gold_claim)
        for gold_doc in gold_claim.values():
            sentence["relevant"] += sum(len(r) for r in gold_doc["rationales"])
        for doc_id, doc_pred in predictions[claim_id].items():
            if doc_pred["label"] == NEI:
                continue
            abstract["retrieved"] += 1
            label_only, rationalized = _abstract_correct(doc_id, doc_pred, gold_claim)
            abstract["correct_label_only"] += int(label_only)
            abstract["correct_rationalized"] += int(rationalized)

            sentence["retrieved"] += len(doc_pred["sentences"])
            if doc_id in gold_claim:
                gold_rationales = [set(r) for r in gold_claim[doc_id]["rationales"]]
                n_correct = _count_rationale_sents(set(doc_pred["sentences"]), gold_rationales)
                sentence["correct_selection"] += n_correct
                sentence["correct_label"] += n_correct * int(doc_pred["label"] == gold_claim[doc_id]["label"])
    return {
        "sentence_selection": compute_f1(sentence, "selection"),
        "sentence_label": compute_f1(sentence, "label"),
        "abstract_label_only": compute_f1(abstract, "label_only"),
        "abstract_rationalized": compute_f1(abstract, "rationalized"),
        "counts": {"abstract": dict(abstract), "sentence": dict(sentence)},
    }
