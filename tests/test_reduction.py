from __future__ import annotations

import numpy as np

from tasnif.reduction import reduce_pca


def test_reduce_pca_target_dim() -> None:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(50, 128)).astype(np.float32)
    reduced = reduce_pca(embs, n_components=16)
    assert reduced.shape == (50, 16)


def test_reduce_pca_clamps_when_components_exceed_samples() -> None:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(4, 64)).astype(np.float32)
    reduced = reduce_pca(embs, n_components=16)
    assert reduced.shape[1] == 4


def test_reduce_pca_handles_3d_input() -> None:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(10, 4, 8)).astype(np.float32)
    reduced = reduce_pca(embs, n_components=4)
    assert reduced.shape == (10, 4)


def test_reduce_pca_is_deterministic_with_seed() -> None:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(30, 64)).astype(np.float32)
    a = reduce_pca(embs, n_components=8, random_state=7)
    b = reduce_pca(embs, n_components=8, random_state=7)
    np.testing.assert_array_equal(a, b)
