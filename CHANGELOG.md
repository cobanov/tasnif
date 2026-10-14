# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-14

### Changed
- **The default backbone is `timm:convnext_small.fb_in22k`** (ConvNeXt-S pretrained on ImageNet-22k) instead of `resnet50`. Across five 10-class sets (Imagenette, Imagewoof, Flowers, EuroSAT, DTD) its clusters reach a mean NMI of 0.84 against 0.71 for ResNet-50, which was trained on ImageNet-1k's classes and falls behind outside them; it embeds about 107 images per second on an M4 Pro against 193. The benchmark is in the README. Pass `embedder_kwargs={"model": "resnet50"}` (CLI: `-m resnet50`) for the previous behavior.
- **Images are streamed.** `fit`, `predict`, `transform` and the CLI used to decode every image at full resolution and keep all of them for the lifetime of the clusterer, about 36 MB per 12 MP photo, so a folder of 10,000 photos needed several hundred GB of RAM. Images are now decoded a few ahead on a thread pool (`workers`), scaled so their shorter side is at most `max_image_size` (default 512 px; JPEGs are decoded at reduced scale directly) and dropped once embedded. Grid previews load their thumbnails from disk at export time. `max_image_size=None` (CLI: `--max-size 0`) keeps full resolution.
- K-Means runs ten initializations instead of one. A single k-means++ start moved NMI by about 0.02 between seeds.
- Exports never replace an existing file. Images with the same name in one cluster get a numeric suffix (`IMG_0001-1.jpg`), and `clusters.csv` records the name each one was given.
- CI runs on Python 3.10 to 3.14 with current GitHub Actions; ruff, mypy and the pre-commit hooks are updated.

### Added
- `n_clusters="auto"` (CLI: `-k auto`) picks the number of clusters with the best silhouette score between `k_min` and `k_max`. The choice is `n_clusters_`, the score of each candidate `k_scores_`.
- Optional `face` backend via the `[face]` extra ([DeepFace](https://github.com/serengil/deepface)): embeds the largest face in each image with a face recognition model (default `Facenet512` with `retinaface`), and the whole image when no face is found, so photos group by person. Proposed by [@serengil](https://github.com/serengil) in [#2](https://github.com/cobanov/tasnif/issues/2).
- `ImageStream`, a lazy, threaded image loader that skips unreadable files; `fit_pca`, which returns the fitted PCA with the reduced matrix; `choose_n_clusters`.
- `TimmEmbedder(model_kwargs=...)`, passed to `timm.create_model` (for example `{"img_size": 224}`).
- `load_image(path, max_size=...)` and the CLI's `--max-size`.
- `skipped_` on the clusterer and `skipped` in `tasnif embed`'s metadata list files that could not be read.

### Fixed
- **Data loss in `move` exports.** Two images with the same file name from different folders (discovery is recursive) that landed in the same cluster overwrote each other, and in `move` mode the first photo was deleted. Re-exporting into a non-empty folder with `overwrite=True` could delete earlier files the same way.
- `predict` fitted a new PCA on the new images, so they were projected into a different space than the one the clusters were fitted in, and predicting a single image failed. It now reuses the PCA fitted in `fit`.
- Images are turned upright from their EXIF orientation. Phone photos were embedded and previewed sideways.
- One unreadable file stopped the whole run. It is now skipped with a warning.
- `embedder="clip"` without the `[clip]` extra raised a bare `ImportError` instead of saying how to install it.

### Removed
- The `umap` extra, which nothing used.

## [0.2.0] - 2026-05-14

### Breaking
- **Rewritten public API.** The old `Tasnif().read().calculate().export()` flow has been removed. Use `TasnifClusterer` with scikit-learn-style `fit` / `predict` / `fit_predict` / `transform` instead. See the migration block in the README.
- **Dropped `img2vec_pytorch`** in favor of [`timm`](https://github.com/huggingface/pytorch-image-models). Any `timm` model is now usable as a backbone (`resnet50`, `convnext_base`, `vit_base_patch14_dinov2`, ...).
- **Minimum Python is 3.10.**
- **Package layout moved to `src/tasnif/`**; old `tasnif/` is gone.
- **`setup.py` / `setup.cfg` replaced with `pyproject.toml`** (PEP 621, Hatchling backend).

### Added
- New `TasnifClusterer` facade with explicit `fit`, `predict`, `fit_predict`, `transform`, and `export` methods.
- `ClusterResult` dataclass: `labels`, `centroids`, `counts`, `paths`, `inertia`, `silhouette`, `embedder`, `pca_dim` - inspect results without writing to disk.
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

## [0.1.11] - Pre-rewrite

Last release of the legacy API. Frozen for reference.
