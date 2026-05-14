from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .logging import get_logger

_log = get_logger("clustering")


@dataclass(frozen=True, slots=True)
class KMeansFit:
    labels: np.ndarray
    centroids: np.ndarray
    counts: np.ndarray
    inertia: float
    silhouette: float | None
    model: KMeans


def fit_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    n_init: int | str = "auto",
    max_iter: int = 300,
    random_state: int | None = 42,
    compute_silhouette: bool = False,
) -> KMeansFit:
    """Cluster ``features`` with K-Means.

    Returns labels, centroids, per-cluster counts, inertia, and (optionally) the
    silhouette score. Set ``compute_silhouette=True`` for evaluation runs - it
    is O(N^2) and you usually don't want it on large datasets.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if n_clusters > len(features):
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed number of samples ({len(features)})"
        )

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = model.fit_predict(features)
    counts = np.bincount(labels, minlength=n_clusters)

    silhouette: float | None = None
    if compute_silhouette and n_clusters > 1 and len(features) > n_clusters:
        try:
            silhouette = float(silhouette_score(features, labels))
        except ValueError as exc:
            _log.warning("Silhouette score failed: %s", exc)

    _log.info(
        "KMeans: k=%d, inertia=%.3f, counts=%s%s",
        n_clusters,
        float(model.inertia_),
        counts.tolist(),
        f", silhouette={silhouette:.3f}" if silhouette is not None else "",
    )
    return KMeansFit(
        labels=labels.astype(np.int32, copy=False),
        centroids=model.cluster_centers_.astype(np.float32, copy=False),
        counts=counts.astype(np.int64, copy=False),
        inertia=float(model.inertia_),
        silhouette=silhouette,
        model=model,
    )


def predict_kmeans(model: KMeans, features: np.ndarray) -> np.ndarray:
    """Assign new samples to clusters of an already-fitted KMeans model."""
    out: np.ndarray = model.predict(features).astype(np.int32, copy=False)
    return out
