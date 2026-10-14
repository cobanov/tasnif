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


def test_predict_reuses_the_fitted_pca(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)
    clf.fit(tmp_image_dir)
    paths = sorted(tmp_image_dir.iterdir())

    assert clf.predict(paths).tolist() == clf.labels_.tolist()
    # A single image used to fail: PCA was refitted on it, down to one component.
    assert clf.predict([paths[5]]).tolist() == [clf.labels_[5]]


def test_auto_n_clusters(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(
        n_clusters="auto", embedder=dummy_embedder, pca_dim=8, show_progress=False, random_state=0
    )
    result = clf.fit_predict(tmp_image_dir)

    assert clf.n_clusters_ == 4
    assert result.n_clusters == 4
    assert clf.k_scores_ is not None
    assert set(clf.k_scores_) == set(range(2, 8))


def test_invalid_n_clusters() -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        TasnifClusterer(n_clusters=0)
    with pytest.raises(ValueError, match="n_clusters"):
        TasnifClusterer(n_clusters="many")  # type: ignore[arg-type]


def test_fit_skips_unreadable_images(tmp_image_dir: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    broken = tmp_image_dir / "zz_broken.jpg"
    broken.write_bytes(b"\xff\xd8 truncated")
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)

    result = clf.fit_predict(tmp_image_dir)

    assert result.n_samples == 8
    assert broken not in result.paths
    assert clf.skipped_ == (broken,)


def test_fit_accepts_pil_images_and_exports_their_grids(
    pil_images,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    dummy_embedder,  # type: ignore[no-untyped-def]
) -> None:
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)
    result = clf.fit_predict(pil_images)
    assert result.n_samples == 8

    clf.predict(pil_images[:2])  # must not drop the images kept for the grids
    clf.transform(pil_images[:1])
    manifest = clf.export(tmp_path / "out", mode="none")
    assert len(manifest.grid_paths) == 4


def test_export_grids_load_from_disk(tmp_image_dir: Path, tmp_path: Path, dummy_embedder) -> None:  # type: ignore[no-untyped-def]
    clf = TasnifClusterer(n_clusters=4, embedder=dummy_embedder, pca_dim=8, show_progress=False)
    clf.fit(tmp_image_dir)
    manifest = clf.export(tmp_path / "out", mode="none")
    assert len(manifest.grid_paths) == 4
