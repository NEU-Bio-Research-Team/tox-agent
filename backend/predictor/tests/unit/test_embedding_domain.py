import numpy as np

from toxpred.scientific.embedding_domain import EmbeddingDomain


def test_embedding_domain_is_frozen_and_scores_without_fake_confidence():
    domain = EmbeddingDomain(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), k=2)
    score = domain.score(np.array([0.4, 0.4]))
    assert score.knn_distance >= 0
    assert score.mahalanobis_distance >= 0
    assert 0 < score.density_proxy <= 1
    assert not hasattr(score, "confidence")
