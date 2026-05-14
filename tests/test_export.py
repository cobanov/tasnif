from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tasnif.export import export_clusters
from tasnif.types import ClusterResult


def _make_result(tmp_image_dir: Path, n_clusters: int = 2) -> ClusterResult:
    paths = sorted(tmp_image_dir.iterdir())
    labels = np.array([i % n_clusters for i in range(len(paths))], dtype=np.int32)
    counts = np.bincount(labels, minlength=n_clusters)
    return ClusterResult(
        labels=labels,
        centroids=np.zeros((n_clusters, 4), dtype=np.float32),
        counts=counts,
        paths=tuple(paths),
        n_clusters=n_clusters,
        inertia=1.0,
        silhouette=None,
        embedder="dummy",
        pca_dim=4,
    )


def test_export_copy_creates_cluster_dirs(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    manifest = export_clusters(result, out, mode="copy", write_grid=False)

    assert manifest.output_dir == out
    assert (out / "cluster_0").is_dir()
    assert (out / "cluster_1").is_dir()
    assert manifest.n_files == 8
    assert manifest.manifest_csv is not None and manifest.manifest_csv.exists()


def test_export_manifest_csv_contents(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    manifest = export_clusters(result, out, mode="copy", write_grid=False)

    assert manifest.manifest_csv is not None
    with manifest.manifest_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8
    assert {"path", "cluster", "filename"} == set(rows[0].keys())


def test_export_summary_json(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    manifest = export_clusters(result, out, mode="copy", write_grid=False)

    assert manifest.summary_json is not None
    data = json.loads(manifest.summary_json.read_text())
    assert data["n_clusters"] == 2
    assert data["n_samples"] == 8


def test_export_symlink_mode(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    export_clusters(result, out, mode="symlink", write_grid=False)
    links = list((out / "cluster_0").iterdir())
    assert links
    assert all(p.is_symlink() for p in links)


def test_export_with_grids(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    images = [Image.open(p).convert("RGB") for p in result.paths]
    out = tmp_path / "out"
    manifest = export_clusters(result, out, mode="copy", images=images, write_grid=True)
    assert manifest.grid_paths
    for p in manifest.grid_paths:
        assert p.exists()


def test_export_with_embeddings(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    embs = np.zeros((len(result.paths), 8), dtype=np.float32)
    out = tmp_path / "out"
    manifest = export_clusters(result, out, mode="copy", embeddings=embs, write_grid=False)
    assert manifest.embeddings_path is not None
    assert manifest.embeddings_path.exists()
    loaded = np.load(manifest.embeddings_path)
    assert loaded.shape == embs.shape


def test_export_refuses_non_empty_dir(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        export_clusters(result, out, mode="copy", write_grid=False)


def test_export_overwrite_flag(tmp_image_dir: Path, tmp_path: Path) -> None:
    result = _make_result(tmp_image_dir, n_clusters=2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing").write_text("x")
    export_clusters(result, out, mode="copy", write_grid=False, overwrite=True)
    assert (out / "cluster_0").is_dir()
