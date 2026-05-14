from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .logging import get_logger

_log = get_logger("reduction")


def reduce_pca(
    embeddings: np.ndarray,
    *,
    n_components: int = 16,
    random_state: int | None = 42,
) -> np.ndarray:
    """Reduce ``embeddings`` to ``n_components`` dimensions with PCA.

    ``n_components`` is clamped to ``min(n_samples, n_features, n_components)``.
    """
    if embeddings.ndim != 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    n_samples, n_features = embeddings.shape
    effective = min(n_components, n_samples, n_features)
    if effective < n_components:
        _log.info(
            "Clamped PCA components from %d to %d (samples=%d, features=%d)",
            n_components,
            effective,
            n_samples,
            n_features,
        )

    pca = PCA(n_components=effective, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    _log.info(
        "PCA: %d -> %d (explained variance=%.3f)",
        n_features,
        effective,
        float(pca.explained_variance_ratio_.sum()),
    )
    out: np.ndarray = reduced.astype(np.float32, copy=False)
    return out
