"""Plug in a hand-rolled embedder.

Run with:

    uv run python examples/custom_embedder.py path/to/images
"""

from __future__ import annotations

import sys

import numpy as np

from tasnif import TasnifClusterer


class ColorHistogramEmbedder:
    """Trivial RGB histogram embedder — useful as a baseline."""

    name = "color-hist"

    def __init__(self, bins: int = 8) -> None:
        self.bins = bins
        self._dim = bins * 3

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return "cpu"

    def embed(
        self,
        images,  # type: ignore[no-untyped-def]
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        feats = []
        for img in images:
            arr = np.asarray(img.convert("RGB"))
            channels = [
                np.histogram(arr[..., c], bins=self.bins, range=(0, 256))[0] for c in range(3)
            ]
            vec = np.concatenate(channels).astype(np.float32)
            vec /= vec.sum() + 1e-8
            feats.append(vec)
        return np.stack(feats)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    clf = TasnifClusterer(
        n_clusters=5,
        embedder=ColorHistogramEmbedder(bins=8),
        pca_dim=None,
    )
    result = clf.fit_predict(sys.argv[1])
    print(f"Cluster sizes: {result.counts.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
