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
    n_init: int | str = 10,
    max_iter: int = 300,
    random_state: int | None = 42,
    compute_silhouette: bool = False,
) -> KMeansFit:
    """Cluster ``features`` with K-Means.

    Returns labels, centroids, per-cluster counts, inertia, and (optionally) the
    silhouette score. Ten initializations by default: a single k-means++ start
    moves NMI by about 0.02 from seed to seed on real embeddings. Set
    ``compute_silhouette=True`` for evaluation runs - it is O(N^2) and you
    usually don't want it on large datasets.
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


def choose_n_clusters(
    features: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 30,
    random_state: int | None = 42,
    sample_size: int = 5000,
) -> tuple[int, dict[int, float]]:
    """Pick the number of clusters with the best silhouette score.

    Runs K-Means for every ``k`` in ``[k_min, min(k_max, n_samples - 1)]`` and
    returns the best ``k`` with the score of each. Above ``sample_size``
    samples the silhouette is computed on a random subset, which keeps the
    search linear in the number of images.
    """
    n_samples = len(features)
    upper = min(k_max, n_samples - 1)
    if upper < k_min:
        raise ValueError(
            f"Choosing the number of clusters needs at least {k_min + 1} samples, got {n_samples}"
        )

    scores: dict[int, float] = {}
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(
            features
        )
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = float(
            silhouette_score(
                features,
                labels,
                sample_size=sample_size if n_samples > sample_size else None,
                random_state=random_state,
            )
        )
    if not scores:
        raise ValueError("Could not split the samples into two or more clusters")
    best = max(scores, key=lambda k: scores[k])
    _log.info("Chose k=%d by silhouette (%.3f) from k=%d..%d", best, scores[best], k_min, upper)
    return best, scores


def predict_kmeans(model: KMeans, features: np.ndarray) -> np.ndarray:
    """Assign new samples to clusters of an already-fitted KMeans model."""
    out: np.ndarray = model.predict(features).astype(np.int32, copy=False)
    return out
