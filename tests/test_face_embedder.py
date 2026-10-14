from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tasnif.embeddings import Embedder, create_embedder, list_backends
from tasnif.exceptions import BackendNotAvailableError


class _FakeDeepFace:
    """Stands in for ``deepface.DeepFace``: one face per image, embedding = mean BGR."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build_model(self, model_name: str) -> Any:
        return types.SimpleNamespace(output_shape=3, device="cpu")

    def represent(self, img_path: list[np.ndarray], **kwargs: Any) -> list[Any]:
        self.calls.append({"n": len(img_path), **kwargs})
        results = [
            [{"embedding": img.reshape(-1, 3).mean(axis=0).tolist(), "face_confidence": 0.9}]
            for img in img_path
        ]
        # DeepFace returns a single image's faces unwrapped.
        return results[0] if len(img_path) == 1 else results


@pytest.fixture
def fake_deepface(monkeypatch: pytest.MonkeyPatch) -> _FakeDeepFace:
    fake = _FakeDeepFace()
    module = types.ModuleType("deepface")
    module.DeepFace = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepface", module)
    return fake


def _solid(rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (16, 16), rgb)


def test_face_backend_is_listed() -> None:
    assert "face" in list_backends()


def test_face_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "deepface", None)
    with pytest.raises(BackendNotAvailableError, match=r"tasnif\[face\]"):
        create_embedder("face")


def test_embeds_one_row_per_image_across_batches(fake_deepface: _FakeDeepFace) -> None:
    backend = create_embedder("face", normalize=False)
    assert isinstance(backend, Embedder)
    assert backend.name == "face:Facenet512@retinaface"
    assert backend.dim == 3

    images = [_solid((10 * i, 0, 0)) for i in range(5)]
    out = backend.embed(images, batch_size=2, show_progress=False)

    assert out.shape == (5, 3)
    assert out.dtype == np.float32
    # Batches of 2, 2 and 1: the last one exercises the unwrapped single-image reply.
    assert [call["n"] for call in fake_deepface.calls] == [2, 2, 1]
    # DeepFace gets BGR, so red lands in the last channel.
    np.testing.assert_allclose(out[:, 2], [0, 10, 20, 30, 40])
    np.testing.assert_allclose(out[:, :2], 0)


def test_keeps_the_largest_face_and_falls_back_to_the_whole_image(
    fake_deepface: _FakeDeepFace,
) -> None:
    backend = create_embedder("face", model="ArcFace", detector="yunet")
    backend.embed([_solid((1, 2, 3))], show_progress=False)

    call = fake_deepface.calls[0]
    assert call["model_name"] == "ArcFace"
    assert call["detector_backend"] == "yunet"
    assert call["max_faces"] == 1
    assert call["enforce_detection"] is False


def test_normalizes_rows(fake_deepface: _FakeDeepFace) -> None:
    backend = create_embedder("face")
    out = backend.embed([_solid((200, 100, 50)), _solid((0, 0, 0))], show_progress=False)
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(out[1], 0)


def test_empty_input(fake_deepface: _FakeDeepFace) -> None:
    out = create_embedder("face").embed([], show_progress=False)
    assert out.shape == (0, 3)
    assert fake_deepface.calls == []


@pytest.mark.integration
def test_real_deepface_on_images_without_faces() -> None:
    pytest.importorskip("deepface")
    images = [
        Image.open(p).convert("RGB")
        for p in sorted((Path(__file__).parent / "test_images").iterdir())[:3]
    ]
    backend = create_embedder("face")
    out = backend.embed(images, show_progress=False)
    assert out.shape == (3, backend.dim)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
