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
          grids/cluster_0.png   (when ``write_grid``)

    Existing files are never replaced: two images with the same name (from
    different folders, or from an earlier export with ``overwrite=True``) get a
    numeric suffix, ``photo-1.jpg``. ``clusters.csv`` records the name each one
    was given. Grid previews use ``images`` when given and otherwise load
    thumbnails from ``result.paths``.
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {out} is not empty. Pass overwrite=True to merge into it."
        )
    out.mkdir(parents=True, exist_ok=True)

    by_cluster = result.by_cluster()
    files_per_cluster = {k: len(v) for k, v in by_cluster.items()}
    exported: dict[Path, Path] = {}

    if mode != "none":
        for cluster_id, paths in by_cluster.items():
            target = out / f"cluster_{cluster_id}"
            target.mkdir(parents=True, exist_ok=True)
            taken = {child.name for child in target.iterdir()}
            for src in paths:
                name = _free_name(src.name, taken)
                taken.add(name)
                _materialize(src, target / name, mode)
                exported[src] = target / name

    manifest_csv: Path | None = None
    if write_manifest:
        manifest_csv = out / "clusters.csv"
        with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "cluster", "filename"])
            for path, label in zip(result.paths, result.labels, strict=True):
                writer.writerow([str(path), int(label), exported.get(path, path).name])

    grid_paths: list[Path] = []
    if write_grid:
        from .visualization import make_cluster_grid

        grid_dir = out / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        images_by_path = (
            dict(zip(result.paths, images, strict=True)) if images is not None else None
        )
        for cluster_id, paths in by_cluster.items():
            cluster_imgs = _grid_images(paths[:grid_max_images], images_by_path, exported)
            if not cluster_imgs:
                continue
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


def _free_name(name: str, taken: set[str]) -> str:
    """Return ``name``, or ``stem-N.ext`` with the smallest ``N`` not in ``taken``."""
    if name not in taken:
        return name
    path = Path(name)
    n = 1
    while f"{path.stem}-{n}{path.suffix}" in taken:
        n += 1
    return f"{path.stem}-{n}{path.suffix}"


def _grid_images(
    paths: Sequence[Path],
    images_by_path: dict[Path, Image] | None,
    exported: dict[Path, Path],
) -> list[Image]:
    if images_by_path is not None:
        return [images_by_path[p] for p in paths]

    from .io import load_image

    loaded: list[Image] = []
    for path in paths:
        try:
            # After a move the image only exists at its exported path.
            loaded.append(load_image(exported.get(path, path), max_size=256))
        except Exception as exc:
            _log.warning("Leaving %s out of the grid preview: %s", path, exc)
    return loaded


def _materialize(src: Path, dst: Path, mode: ExportMode) -> None:
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
