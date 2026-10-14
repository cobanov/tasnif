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


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: rotate 90° clockwise to display
    Image.new("RGB", (40, 20), "white").save(path, exif=exif)
    assert load_image(path).size == (20, 40)


def test_load_image_caps_the_shorter_side(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "big.jpg"
    Image.new("RGB", (1200, 900), "white").save(path)
    assert load_image(path, max_size=300).size == (400, 300)
    assert load_image(path, max_size=2000).size == (1200, 900)


def test_image_stream_skips_unreadable_files_in_order(tmp_image_dir: Path) -> None:
    from tasnif.io import ImageStream

    paths = discover_images(tmp_image_dir)
    broken = tmp_image_dir / "img_03.png"
    broken.write_bytes(b"not an image")
    stream = ImageStream(paths, workers=3)

    images = list(stream)

    assert len(stream) == 8
    assert len(images) == 7
    assert stream.skipped == [broken]
    assert stream.loaded == [p for p in paths if p != broken]
    # Iterating again starts over.
    assert len(list(stream)) == 7
