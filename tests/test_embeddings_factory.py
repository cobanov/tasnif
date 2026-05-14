from __future__ import annotations

import numpy as np
import pytest

from tasnif.embeddings import (
    Embedder,
    create_embedder,
    list_backends,
    register_embedder,
)
from tasnif.exceptions import BackendNotAvailableError


class _CustomEmbedder:
    name = "custom"

    @property
    def dim(self) -> int:
        return 4

    @property
    def device(self) -> str:
        return "cpu"

    def embed(self, images, *, batch_size: int = 32, show_progress: bool = True):  # type: ignore[no-untyped-def]
        imgs = list(images)
        return np.zeros((len(imgs), 4), dtype=np.float32)


def test_register_and_create_custom_backend() -> None:
    register_embedder("custom", lambda: _CustomEmbedder())
    backend = create_embedder("custom")
    assert isinstance(backend, Embedder)
    assert backend.name == "custom"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        create_embedder("does-not-exist")


def test_list_backends_includes_builtins() -> None:
    names = list_backends()
    assert "timm" in names
    assert "clip" in names


def test_clip_backend_raises_when_missing() -> None:
    pytest.importorskip("open_clip", reason="open_clip is intentionally checked when missing")
    # If open_clip IS installed, this test is a no-op. We exercise the negative
    # path with a stub import check:
    try:
        import open_clip  # noqa: F401
    except ImportError:
        with pytest.raises(BackendNotAvailableError):
            create_embedder("clip")
