from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .logging import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("visualization")


def make_cluster_grid(
    images: Sequence[Image],
    output_path: str | Path,
    *,
    title: str | None = None,
    max_images: int = 9,
    cols: int | None = None,
    dpi: int = 150,
    thumbnail_size: int = 256,
) -> Path:
    """Save a grid preview of ``images`` to ``output_path``.

    Fixes the long-standing bug of saving inside the loop / leaking figures.
    Returns the written path.
    """
    import matplotlib.pyplot as plt

    if not images:
        raise ValueError("Cannot create a grid from an empty image list.")

    out = Path(output_path)
    selected = list(images[:max_images])
    n = len(selected)
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, img in zip(axes_flat, selected, strict=False):
        thumb = img.copy()
        thumb.thumbnail((thumbnail_size, thumbnail_size))
        ax.imshow(thumb)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _log.debug("Wrote grid: %s", out)
    return out
