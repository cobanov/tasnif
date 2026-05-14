from __future__ import annotations

from pathlib import Path

import pytest

from tasnif.exceptions import NoImagesFoundError
from tasnif.io import discover_images, iter_images, load_image


def test_discover_images_finds_supported_formats(tmp_image_dir: Path) -> None:
    paths = discover_images(tmp_image_dir)
    assert len(paths) == 8
    assert all(p.suffix == ".png" for p in paths)


def test_discover_images_returns_sorted(tmp_image_dir: Path) -> None:
    paths = discover_images(tmp_image_dir)
    assert paths == sorted(paths)


def test_discover_images_recursive(tmp_image_dir: Path) -> None:
    nested = tmp_image_dir / "nested"
    nested.mkdir()
    (nested / "extra.png").write_bytes((tmp_image_dir / "img_00.png").read_bytes())
    paths = discover_images(tmp_image_dir, recursive=True)
    assert len(paths) == 9


def test_discover_images_non_recursive(tmp_image_dir: Path) -> None:
    nested = tmp_image_dir / "nested"
    nested.mkdir()
    (nested / "extra.png").write_bytes((tmp_image_dir / "img_00.png").read_bytes())
    paths = discover_images(tmp_image_dir, recursive=False)
    assert len(paths) == 8


def test_discover_images_empty(tmp_path: Path) -> None:
    with pytest.raises(NoImagesFoundError):
        discover_images(tmp_path)


def test_load_image_returns_rgb(tmp_image_dir: Path) -> None:
    img = load_image(next(tmp_image_dir.iterdir()))
    assert img.mode == "RGB"


def test_iter_images_lazy(tmp_image_dir: Path) -> None:
    paths = discover_images(tmp_image_dir)
    images = list(iter_images(paths))
    assert len(images) == len(paths)
