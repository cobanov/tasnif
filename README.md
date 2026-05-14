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
labels required. It uses modern pretrained vision backbones (`timm` by default,
optionally CLIP via `open_clip`), reduces the embedding with PCA, and runs
K-Means on top.

## Highlights

- **Modern backbones** - any [`timm`](https://github.com/huggingface/pytorch-image-models) model (ResNet, ConvNeXt, ViT, DINOv2, ...) or [CLIP](https://github.com/mlfoundations/open_clip) via the `[clip]` extra.
- **GPU / MPS / CPU** with automatic device detection.
- **scikit-learn-style API**: `fit`, `predict`, `fit_predict`, `transform`.
- **Rich export**: per-cluster folders, CSV manifest, JSON summary, preview grids, raw embeddings.
- **Multiple materialization modes**: `copy`, `symlink`, `move`, or `none` (metadata only).
- **First-class CLI** powered by [Typer](https://typer.tiangolo.com/).
- **Pluggable embedders** - register your own backend.
- **Deterministic** with a `random_state` seed.

## Installation

`tasnif` is built and packaged with [uv](https://docs.astral.sh/uv/), but plain pip works too.

```bash
pip install tasnif                # core
pip install "tasnif[cli]"         # core + CLI
pip install "tasnif[clip]"        # core + CLIP backend
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

clf = TasnifClusterer(n_clusters=5, embedder="timm", pca_dim=16)
clf.fit("photos/")
result = clf.result_

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
# Cluster a directory with the default ResNet-50 (timm) backbone
tasnif cluster photos/ -k 5 -o output/

# Use CLIP and copy via symlink (fast, non-destructive)
tasnif cluster photos/ -k 8 --embedder clip --model ViT-B-32 --mode symlink

# Just compute embeddings, save .npy + .json
tasnif embed photos/ --embedder timm --model convnext_base -o embeddings.npy

# List available backends
tasnif backends
```

Run `tasnif --help` to see all options.

## Use a different model

Default is `timm:resnet50`. Pass any `timm` model name:

```python
clf = TasnifClusterer(
    n_clusters=8,
    embedder="timm",
    embedder_kwargs={"model": "vit_base_patch14_dinov2.lvd142m", "device": "cuda"},
)
```

CLIP:

```python
clf = TasnifClusterer(
    n_clusters=8,
    embedder="clip",
    embedder_kwargs={"model": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
)
```

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
    discover_images, create_embedder,
    reduce_pca, fit_kmeans, export_clusters, ClusterResult,
)

paths = discover_images("photos/")
embedder = create_embedder("timm", model="resnet50")
embeddings = embedder.embed([open_pil(p) for p in paths])
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
