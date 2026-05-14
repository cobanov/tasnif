from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tasnif.visualization import make_cluster_grid


def test_make_cluster_grid_writes_file(pil_images: list[Image.Image], tmp_path: Path) -> None:
    out = make_cluster_grid(pil_images, tmp_path / "grid.png", title="Test")
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_cluster_grid_truncates_to_max(pil_images: list[Image.Image], tmp_path: Path) -> None:
    out = make_cluster_grid(pil_images, tmp_path / "grid.png", max_images=4, cols=2)
    assert out.exists()


def test_make_cluster_grid_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_cluster_grid([], tmp_path / "grid.png")
