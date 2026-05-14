from __future__ import annotations

from pathlib import Path

import pytest

from tasnif import TasnifClusterer
from tasnif.exceptions import NotFittedError


def test_clusterer_fit_predict_with_dummy(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(
        n_clusters=4,
        embedder=dummy_embedder,
        pca_dim=8,
        show_progress=False,
        random_state=0,
    )
    result = clf.fit_predict(tmp_image_dir)

    assert result.n_clusters == 4
    assert result.n_samples == 8
    assert len(result.paths) == 8
    assert result.labels.shape == (8,)
    # Two images per color → KMeans should recover 4 clusters of size 2
    assert sorted(result.counts.tolist()) == [2, 2, 2, 2]


def test_clusterer_result_dict_view(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)
    result = clf.fit_predict(tmp_image_dir)

    mapping = result.as_dict()
    assert len(mapping) == 8
    by_cluster = result.by_cluster()
    assert set(by_cluster.keys()) == {0, 1, 2, 3}


def test_clusterer_transform_does_not_fit(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, show_progress=False)
    embs = clf.transform(tmp_image_dir)
    assert embs.shape == (8, 32)
    with pytest.raises(NotFittedError):
        _ = clf.result_


def test_clusterer_predict_after_fit(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(
        n_clusters=4,
        embedder=dummy_embedder,
        pca_dim=8,
        show_progress=False,
        random_state=0,
    )
    clf.fit(tmp_image_dir)
    preds = clf.predict(tmp_image_dir)
    assert preds.shape == (8,)


def test_clusterer_export(tmp_image_dir: Path, tmp_path: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)
    clf.fit(tmp_image_dir)
    manifest = clf.export(tmp_path / "out", mode="copy", write_grid=False)
    assert manifest.n_files == 8
    assert manifest.embeddings_path is not None


def test_clusterer_without_pca(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(
        n_clusters=4, embedder=dummy_embedder, pca_dim=None, show_progress=False, random_state=0
    )
    result = clf.fit_predict(tmp_image_dir)
    assert result.pca_dim is None
    assert result.centroids.shape == (4, 32)
