from __future__ import annotations

import numpy as np
import pytest

from tasnif.clustering import fit_kmeans, predict_kmeans


def test_fit_kmeans_recovers_clusters(fake_embeddings: np.ndarray) -> None:
    fit = fit_kmeans(fake_embeddings, n_clusters=4)
    assert fit.labels.shape == (20,)
    assert fit.centroids.shape == (4, 32)
    assert fit.counts.sum() == 20
    assert len(set(fit.labels.tolist())) == 4


def test_fit_kmeans_silhouette_optional(fake_embeddings: np.ndarray) -> None:
    fit = fit_kmeans(fake_embeddings, n_clusters=4, compute_silhouette=True)
    assert fit.silhouette is not None
    assert fit.silhouette > 0.5

    no_sil = fit_kmeans(fake_embeddings, n_clusters=4, compute_silhouette=False)
    assert no_sil.silhouette is None


def test_fit_kmeans_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_kmeans(np.zeros((5,), dtype=np.float32), n_clusters=2)
    with pytest.raises(ValueError, match=">= 1"):
        fit_kmeans(np.zeros((10, 4), dtype=np.float32), n_clusters=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_kmeans(np.zeros((3, 4), dtype=np.float32), n_clusters=5)


def test_predict_kmeans(fake_embeddings: np.ndarray) -> None:
    fit = fit_kmeans(fake_embeddings, n_clusters=4)
    pred = predict_kmeans(fit.model, fake_embeddings)
    np.testing.assert_array_equal(pred, fit.labels)


def test_fit_kmeans_deterministic(fake_embeddings: np.ndarray) -> None:
    a = fit_kmeans(fake_embeddings, n_clusters=4, random_state=1)
    b = fit_kmeans(fake_embeddings, n_clusters=4, random_state=1)
    np.testing.assert_array_equal(a.labels, b.labels)
