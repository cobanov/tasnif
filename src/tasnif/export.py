from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .logging import get_logger
from .types import ClusterResult, ExportManifest

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("export")

ExportMode = Literal["copy", "symlink", "move", "none"]


def export_clusters(
    result: ClusterResult,
    output_dir: str | Path,
    *,
    mode: ExportMode = "copy",
    images: Sequence[Image] | None = None,
    embeddings: np.ndarray | None = None,
    write_grid: bool = True,
    write_manifest: bool = True,
    grid_max_images: int = 9,
    overwrite: bool = False,
    extra_metadata: dict[str, object] | None = None,
) -> ExportManifest:
    """Materialize a :class:`ClusterResult` to disk.

    Layout::

        output_dir/
          cluster_0/   image001.jpg ...
          cluster_1/   ...
          clusters.csv          (path, cluster, filename)
          summary.json          (params, counts, metrics)
          embeddings.npy        (when ``embeddings`` is provided)
          grids/cluster_0.png   (when ``write_grid`` and ``images`` provided)
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {out} is not empty. Pass overwrite=True to merge into it."
        )
    out.mkdir(parents=True, exist_ok=True)

    by_cluster = result.by_cluster()
    files_per_cluster: dict[int, int] = {}

    if mode != "none":
        for cluster_id, paths in by_cluster.items():
            target = out / f"cluster_{cluster_id}"
            target.mkdir(parents=True, exist_ok=True)
            for src in paths:
                _materialize(src, target / src.name, mode)
            files_per_cluster[cluster_id] = len(paths)
    else:
        files_per_cluster = {k: len(v) for k, v in by_cluster.items()}

    manifest_csv: Path | None = None
    if write_manifest:
        manifest_csv = out / "clusters.csv"
        with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "cluster", "filename"])
            for path, label in zip(result.paths, result.labels, strict=True):
                writer.writerow([str(path), int(label), path.name])

    grid_paths: list[Path] = []
    if write_grid and images is not None:
        from .visualization import make_cluster_grid

        grid_dir = out / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        images_by_path = dict(zip(result.paths, images, strict=True))
        for cluster_id, paths in by_cluster.items():
            if not paths:
                continue
            cluster_imgs = [images_by_path[p] for p in paths[:grid_max_images]]
            grid_path = make_cluster_grid(
                cluster_imgs,
                grid_dir / f"cluster_{cluster_id}.png",
                title=f"Cluster {cluster_id} (n={len(paths)})",
                max_images=grid_max_images,
            )
            grid_paths.append(grid_path)

    embeddings_path: Path | None = None
    if embeddings is not None:
        embeddings_path = out / "embeddings.npy"
        np.save(embeddings_path, embeddings)

    summary_json = out / "summary.json"
    summary: dict[str, object] = {
        "n_clusters": result.n_clusters,
        "n_samples": result.n_samples,
        "counts": result.counts.tolist(),
        "embedder": result.embedder,
        "pca_dim": result.pca_dim,
        "inertia": result.inertia,
        "silhouette": result.silhouette,
        "mode": mode,
    }
    if extra_metadata:
        summary["extra"] = extra_metadata
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _log.info("Exported %d files to %s (mode=%s)", sum(files_per_cluster.values()), out, mode)

    return ExportManifest(
        output_dir=out,
        mode=mode,
        n_clusters=result.n_clusters,
        n_files=sum(files_per_cluster.values()),
        files_per_cluster=files_per_cluster,
        embeddings_path=embeddings_path,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        grid_paths=tuple(grid_paths),
    )


def _materialize(src: Path, dst: Path, mode: ExportMode) -> None:
    if dst.exists():
        dst.unlink()
    try:
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            dst.symlink_to(src.resolve())
        elif mode == "move":
            shutil.move(str(src), str(dst))
        else:
            raise ValueError(f"Unsupported export mode: {mode}")
    except OSError as exc:
        _log.error("Failed to %s %s -> %s: %s", mode, src, dst, exc)
        raise
