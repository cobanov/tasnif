from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a directory of 8 deterministic synthetic RGB images.

    Each image is filled with a distinct base color + noise, so PCA + KMeans
    can recover four clusters of two images each.
    """
    rng = np.random.default_rng(0)
    colors = [
        (220, 30, 30),  # red
        (30, 220, 30),  # green
        (30, 30, 220),  # blue
        (230, 230, 30),  # yellow
    ]
    out = tmp_path / "images"
    out.mkdir()
    for idx in range(8):
        base = colors[idx // 2]
        arr = np.tile(np.array(base, dtype=np.uint8), (64, 64, 1))
        noise = rng.integers(-20, 20, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out / f"img_{idx:02d}.png")
    return out


@pytest.fixture
def pil_images(tmp_image_dir: Path) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in sorted(tmp_image_dir.iterdir())]


@pytest.fixture
def fake_embeddings() -> np.ndarray:
    """Deterministic embeddings with 4 clusters of 5 samples each in 32-D."""
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(4, 32)) * 3
    embeddings = np.repeat(centers, 5, axis=0)
    embeddings = embeddings + rng.normal(scale=0.1, size=embeddings.shape)
    return embeddings.astype(np.float32)


class DummyEmbedder:
    """Embedder that returns a hash of image bytes - no model download needed."""

    name = "dummy"

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

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
        show_progress: bool = False,
    ) -> np.ndarray:
        imgs = list(images)
        out = np.empty((len(imgs), self._dim), dtype=np.float32)
        for i, img in enumerate(imgs):
            arr = np.asarray(img.resize((8, 8))).astype(np.float32).flatten()
            # Pad / truncate to dim
            if arr.size >= self._dim:
                out[i] = arr[: self._dim]
            else:
                buf = np.zeros(self._dim, dtype=np.float32)
                buf[: arr.size] = arr
                out[i] = buf
        # Normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def dummy_embedder() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture(autouse=True)
def _close_pil_images() -> Iterator[None]:
    yield
