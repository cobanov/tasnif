<p align="center">
  <img src="assets/asdd.png" width="350" alt="Tasnif">
</p>

<p align="center">
  <strong>Unsupervised image clustering with modern deep embeddings.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/tasnif/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tasnif"></a>
  <a href="https://pypi.org/project/tasnif/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/tasnif"></a>
  <a href="https://github.com/cobanov/tasnif/actions"><img alt="CI" src="https://github.com/cobanov/tasnif/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

`tasnif` turns a folder of images into clusters you can browse on disk - no
labels required. It embeds each image with a pretrained vision backbone
(ConvNeXt pretrained on ImageNet-22k by default, any `timm` model, CLIP via
`open_clip`, or a face recognition model via `deepface`), reduces the
embeddings with PCA, and runs K-Means on top.

## Highlights

- **Modern backbones** - any [`timm`](https://github.com/huggingface/pytorch-image-models) model (ConvNeXt by default, DINOv2, ViT, ResNet, ...) or [CLIP](https://github.com/mlfoundations/open_clip) via the `[clip]` extra, picked by [benchmark](#use-a-different-model).
- **Group photos by person** - the `face` backend embeds the face in each image with [DeepFace](https://github.com/serengil/deepface) (`[face]` extra).
- **Picks the number of clusters** for you with `n_clusters="auto"` (best silhouette score).
- **Any folder size** - images are streamed and decoded a few at a time at reduced size; unreadable files are skipped, phone photos are turned upright from their EXIF orientation.
- **GPU / MPS / CPU** with automatic device detection.
- **scikit-learn-style API**: `fit`, `predict`, `fit_predict`, `transform`.
- **Rich export**: per-cluster folders, CSV manifest, JSON summary, preview grids, raw embeddings.
- **Multiple materialization modes**: `copy`, `symlink`, `move`, or `none` (metadata only). Existing files are never replaced; clashing names get a numeric suffix.
- **First-class CLI** powered by [Typer](https://typer.tiangolo.com/).
- **Pluggable embedders** - register your own backend.
- **Deterministic** with a `random_state` seed.

## Installation

`tasnif` is built and packaged with [uv](https://docs.astral.sh/uv/), but plain pip works too.

```bash
pip install tasnif                # core
pip install "tasnif[cli]"         # core + CLI
pip install "tasnif[clip]"        # core + CLIP backend
pip install "tasnif[face]"        # core + face backend (DeepFace)
pip install "tasnif[all]"         # everything
```

Development setup:

```bash
git clone https://github.com/cobanov/tasnif
cd tasnif
uv sync --extra dev
uv run pytest
```

## Quickstart - Python API

```python
from tasnif import TasnifClusterer

clf = TasnifClusterer(n_clusters="auto")  # or n_clusters=5
clf.fit("photos/")
result = clf.result_
print(clf.n_clusters_)          # the k that was chosen
print(clf.skipped_)             # files that could not be read

# Inspect without exporting
print(result.counts)            # e.g. [42, 31, 28, 19, 7]
print(result.silhouette)        # None unless compute_silhouette=True
mapping = result.as_dict()      # {Path('photos/a.jpg'): 2, ...}
buckets = result.by_cluster()   # {0: [Path(...), ...], 1: [...]}

# Export to disk
clf.export("output/", mode="copy")
```

## One-shot helper

```python
from tasnif import cluster_directory

cluster_directory("photos/", "output/", n_clusters=5, mode="symlink")
```

## CLI

```bash
# Cluster a directory with the default ConvNeXt backbone
tasnif cluster photos/ -k 5 -o output/

# Let tasnif pick the number of clusters
tasnif cluster photos/ -k auto -o output/

# Use CLIP and copy via symlink (fast, non-destructive)
tasnif cluster photos/ -k 8 --embedder clip --model ViT-B-32 --mode symlink

# Group portraits by person
tasnif cluster portraits/ -k 10 --embedder face

# Just compute embeddings, save .npy + .json
tasnif embed photos/ --embedder timm --model convnext_base -o embeddings.npy

# List available backends
tasnif backends
```

Run `tasnif --help` to see all options.

## Use a different model

The default is `timm:convnext_small.fb_in22k`, ConvNeXt-S pretrained on
ImageNet-22k. Pass any `timm` model name:

```python
clf = TasnifClusterer(
    n_clusters=8,
    embedder="timm",
    embedder_kwargs={"model": "convnext_tiny.fb_in22k", "device": "cuda"},
)
```

How backbones compare: K-Means (k=10, PCA 16) on 600 images from each of five
10-class sets, NMI averaged over three seeds (1.0 means the clusters match the
true classes exactly), and images per second on an Apple M4 Pro GPU:

| Backbone | Imagenette | Imagewoof | Flowers | EuroSAT | DTD | Mean | img/s |
|---|---|---|---|---|---|---|---|
| **`convnext_small.fb_in22k` (default)** | 0.99 | 0.86 | 0.99 | 0.68 | 0.65 | **0.84** | 107 |
| `convnext_tiny.fb_in22k` | 0.99 | 0.83 | 0.97 | 0.71 | 0.60 | 0.82 | 181 |
| DINOv2 ViT-S/14 reg4 at 224 px | 0.90 | 0.84 | 1.00 | 0.62 | 0.64 | 0.80 | 163 |
| CLIP ViT-B-16 | 0.99 | 0.70 | 0.95 | 0.68 | 0.61 | 0.79 | 79 |
| CLIP ViT-B-32 (`clip` default) | 0.96 | 0.65 | 0.93 | 0.70 | 0.61 | 0.77 | 213 |
| `resnet50` (default before 0.3.0) | 0.91 | 0.82 | 0.65 | 0.61 | 0.55 | 0.71 | 193 |

Imagewoof is ten dog breeds, Flowers ten species from Oxford Flowers-102,
EuroSAT satellite tiles and DTD textures. ResNet-50 learned the 1,000
ImageNet-1k classes and falls behind on anything else; the ImageNet-22k
ConvNeXt holds up across all five. With `n_clusters="auto"` it found exactly
10 clusters on three of the five sets and 11 on a fourth.
`convnext_tiny.fb_in22k` is as fast as ResNet-50 if speed matters more.

CLIP:

```python
clf = TasnifClusterer(
    n_clusters=8,
    embedder="clip",
    embedder_kwargs={"model": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
)
```

Faces (requires `tasnif[face]`):

```python
clf = TasnifClusterer(
    n_clusters=10,
    embedder="face",
    embedder_kwargs={"model": "Facenet512", "detector": "retinaface"},
)
```

The `face` backend represents each image by its largest face, so photos of the
same person cluster together whatever the background or framing. An image
without a detectable face is embedded whole by the same model. `model` takes
any [DeepFace](https://github.com/serengil/deepface) model (`Facenet512`,
`ArcFace`, `SFace`, ...) and `detector` any of its detector backends
(`retinaface`, `yunet`, `centerface`, ...). DeepFace runs on PyTorch when
TensorFlow is not installed and picks its own device, so `device` is ignored.
The models it wraps carry their own licenses; check them before production use.

## Large folders

Images given as a folder or a list of paths are never all in memory at once.
`TasnifClusterer` decodes them a few ahead on `workers` threads (default: up to
8), scales each so its shorter side is at most `max_image_size` (512 px by
default; JPEGs are decoded at reduced scale directly, so a 12 MP photo takes
about 1 MB instead of 36 MB) and drops it once embedded. Grid previews load
their thumbnails from disk at export time. Pass `max_image_size=None` (CLI:
`--max-size 0`) to keep full resolution.

## Custom backend

Anything implementing the [`Embedder`](src/tasnif/embeddings/base.py) protocol works:

```python
import numpy as np
from tasnif import TasnifClusterer, register_embedder

class MyEncoder:
    name = "my-encoder"
    @property
    def dim(self): return 128
    @property
    def device(self): return "cpu"
    def embed(self, images, *, batch_size=32, show_progress=True):
        return np.stack([extract(img) for img in images])

register_embedder("my-encoder", lambda: MyEncoder())

clf = TasnifClusterer(n_clusters=5, embedder="my-encoder")
```

## Building blocks

All pieces are independently usable:

```python
from tasnif import (
    discover_images, create_embedder, ImageStream,
    reduce_pca, fit_kmeans, export_clusters, ClusterResult,
)

paths = discover_images("photos/")
embedder = create_embedder("timm")
stream = ImageStream(paths, max_size=512)  # decodes lazily, skips unreadable files
embeddings = embedder.embed(stream)
paths = stream.loaded
reduced = reduce_pca(embeddings, n_components=16)
fit = fit_kmeans(reduced, n_clusters=5, compute_silhouette=True)

result = ClusterResult(
    labels=fit.labels, centroids=fit.centroids, counts=fit.counts,
    paths=tuple(paths), n_clusters=5, inertia=fit.inertia, silhouette=fit.silhouette,
    embedder=embedder.name, pca_dim=16,
)
export_clusters(result, "output/")
```

## Migrating from 0.1.x

The 0.1.x API (`Tasnif().read().calculate().export()`) was removed in 0.2.0.
Replace with:

```diff
- from tasnif import Tasnif
- c = Tasnif(num_classes=5, pca_dim=16, use_gpu=False)
- c.read("photos/")
- c.calculate()
- c.export("output/")
+ from tasnif import TasnifClusterer
+ clf = TasnifClusterer(n_clusters=5, pca_dim=16, embedder_kwargs={"device": "auto"})
+ clf.fit("photos/")
+ clf.export("output/")
```

See [CHANGELOG.md](CHANGELOG.md) for the full list of breaking changes.

## Contributing

Issues and PRs welcome. Before submitting, run:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

Pre-commit is configured - install hooks with `uv run pre-commit install`.

## License

MIT - see [LICENSE](LICENSE).
