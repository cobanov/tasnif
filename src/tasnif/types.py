from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Outcome of a clustering run.

    ``labels[i]`` is the cluster index for ``paths[i]``. ``centroids`` are in the
    same feature space that was passed to the clusterer (post-PCA when PCA was
    applied).
    """

    labels: np.ndarray
    centroids: np.ndarray
    counts: np.ndarray
    paths: tuple[Path, ...]
    n_clusters: int
    inertia: float | None = None
    silhouette: float | None = None
    embedder: str | None = None
    pca_dim: int | None = None

    @property
    def n_samples(self) -> int:
        return len(self.paths)

    def as_dict(self) -> dict[Path, int]:
        return {path: int(label) for path, label in zip(self.paths, self.labels, strict=True)}

    def by_cluster(self) -> dict[int, list[Path]]:
        buckets: dict[int, list[Path]] = {i: [] for i in range(self.n_clusters)}
        for path, label in zip(self.paths, self.labels, strict=True):
            buckets[int(label)].append(path)
        return buckets


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Summary of a completed export operation."""

    output_dir: Path
    mode: str
    n_clusters: int
    n_files: int
    files_per_cluster: dict[int, int] = field(default_factory=dict)
    embeddings_path: Path | None = None
    manifest_csv: Path | None = None
    summary_json: Path | None = None
    grid_paths: tuple[Path, ...] = ()
