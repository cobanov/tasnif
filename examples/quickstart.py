"""Smallest possible end-to-end example.

Run with:

    uv run python examples/quickstart.py path/to/images output/
"""

from __future__ import annotations

import sys

from tasnif import TasnifClusterer, configure_logging


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    src, out = sys.argv[1], sys.argv[2]
    configure_logging()

    clf = TasnifClusterer(
        n_clusters=5,
        embedder="timm",
        embedder_kwargs={"model": "resnet50", "device": "auto"},
        pca_dim=16,
        compute_silhouette=True,
    )
    clf.fit(src)
    result = clf.result_
    print(f"\nClustered {result.n_samples} images into {result.n_clusters} clusters.")
    print(f"Counts: {result.counts.tolist()}")
    if result.silhouette is not None:
        print(f"Silhouette: {result.silhouette:.3f}")

    manifest = clf.export(out, mode="copy")
    print(f"Wrote {manifest.n_files} files to {manifest.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
