"""Abstract retrieval as the SciFact repository defines it.

* ``oracle`` — ``verisci/inference/abstract_retrieval/oracle.py``: the claim's
  gold evidence abstracts; with ``include_nei`` a claim with none gets its first
  cited abstract, so the judge also meets abstracts that should be NEI.
* ``tfidf`` — ``verisci/inference/abstract_retrieval/tfidf.py``: scikit-learn
  TF-IDF over title + abstract, English stop words, n-grams (1, 2) by default,
  top ``k`` (the paper used k = 3).

The retrieval setting is part of the result's label: an oracle number is never
reported as an open-retrieval one.
"""
from __future__ import annotations

from evals.external.scifact.data import Claim, Document


def oracle(claims: list[Claim], *, include_nei: bool = True) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for claim in claims:
        doc_ids = list(claim.evidence)
        if not doc_ids and include_nei and claim.cited_doc_ids:
            doc_ids = [claim.cited_doc_ids[0]]
        out[claim.id] = doc_ids
    return out


def tfidf(claims: list[Claim], corpus: dict[int, Document], *, k: int = 3,
          min_gram: int = 1, max_gram: int = 2) -> dict[int, list[int]]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = list(corpus.values())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(min_gram, max_gram))
    # The official script concatenates title and abstract with no separator
    # after the title; kept byte-for-byte so the ranking is the same.
    doc_vectors = vectorizer.fit_transform([d.title + " ".join(d.sentences) for d in documents])
    out: dict[int, list[int]] = {}
    for claim in claims:
        claim_vector = vectorizer.transform([claim.claim]).todense()
        scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
        ranked = scores.argsort()[::-1].tolist()
        out[claim.id] = [documents[i].doc_id for i in ranked[:k]]
    return out
