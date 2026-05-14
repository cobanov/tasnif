# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Unreleased

### Breaking
- **Rewritten public API.** The old `Tasnif().read().calculate().export()` flow has been removed. Use `TasnifClusterer` with scikit-learn-style `fit` / `predict` / `fit_predict` / `transform` instead. See the migration block in the README.
- **Dropped `img2vec_pytorch`** in favor of [`timm`](https://github.com/huggingface/pytorch-image-models). Any `timm` model is now usable as a backbone (`resnet50`, `convnext_base`, `vit_base_patch14_dinov2`, ...).
- **Minimum Python is 3.10.**
- **Package layout moved to `src/tasnif/`**; old `tasnif/` is gone.
- **`setup.py` / `setup.cfg` replaced with `pyproject.toml`** (PEP 621, Hatchling backend).

### Added
- New `TasnifClusterer` facade with explicit `fit`, `predict`, `fit_predict`, `transform`, and `export` methods.
- `ClusterResult` dataclass: `labels`, `centroids`, `counts`, `paths`, `inertia`, `silhouette`, `embedder`, `pca_dim` — inspect results without writing to disk.
- `ExportManifest` dataclass returned by `export_clusters` summarizing the run.
- Optional CLIP backend via `[clip]` extra (`open_clip`).
- `Embedder` Protocol + `register_embedder()` for plugging in custom backbones.
- First-class CLI: `tasnif cluster`, `tasnif embed`, `tasnif backends`.
- Export modes: `copy`, `symlink`, `move`, `none`.
- Auto-generated CSV manifest (`clusters.csv`) and JSON summary (`summary.json`).
- Optional silhouette score (`compute_silhouette=True`).
- Automatic device detection: `auto` -> `cuda` / `mps` / `cpu`.
- AMP / `torch.autocast` when running on CUDA.
- Recursive directory discovery, deterministic sorting, WebP / BMP / TIFF support.
- Ruff, Mypy (strict), Pre-commit, pytest with markers, GitHub Actions CI matrix (3.10–3.13), PyPI release workflow.

### Changed
- K-Means uses `sklearn.cluster.KMeans` with `random_state` for reproducibility (was `scipy.cluster.vq.kmeans2`).
- PCA uses `sklearn.decomposition.PCA` with `random_state` and clamps to `min(n_samples, n_features, n_components)`.
- Image grids are now produced by a single Matplotlib call instead of `savefig` inside a loop, fixing the figure leak and missing close.
- Logging no longer mutates the root logger on import. Apps must opt in by calling `configure_logging()` or configuring the `tasnif` logger directly.
- Image discovery now sorts paths alphabetically and traverses recursively by default.

### Removed
- `Tasnif` class, `tasnif.calculations`, `tasnif.utils`, `tasnif.logger` modules.
- `img2vec_pytorch` dependency.
- Global `warnings.filterwarnings("ignore")` call.

## [0.1.11] — Pre-rewrite

Last release of the legacy API. Frozen for reference.
