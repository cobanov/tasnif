from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..exceptions import BackendNotAvailableError
from .base import Embedder

_BackendFactory = Callable[..., Embedder]
_REGISTRY: dict[str, _BackendFactory] = {}


def register_embedder(name: str, factory: _BackendFactory) -> None:
    """Register a custom embedder factory under ``name``."""
    _REGISTRY[name] = factory


def list_backends() -> list[str]:
    return sorted({*_REGISTRY.keys(), "timm", "clip", "face"})


def create_embedder(name: str = "timm", /, **kwargs: Any) -> Embedder:
    """Construct an embedder by name.

    Built-in names:

    - ``"timm"``: any ``timm`` model (default ``resnet50``). Pass ``model="vit_base_patch16_224"`` etc.
    - ``"clip"``: open_clip / CLIP image encoder. Requires the ``[clip]`` extra.
    - ``"face"``: a face recognition model via DeepFace (default ``Facenet512``), embedding
      the largest face in each image. Requires the ``[face]`` extra.

    Names registered via :func:`register_embedder` take precedence.
    """
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)

    if name == "timm":
        from .timm_backend import TimmEmbedder

        return TimmEmbedder(**kwargs)

    if name == "clip":
        try:
            from .clip_backend import CLIPEmbedder
        except ImportError as exc:
            raise BackendNotAvailableError(
                "CLIP backend requires the 'clip' extra. Install with: pip install 'tasnif[clip]'"
            ) from exc
        return CLIPEmbedder(**kwargs)

    if name == "face":
        try:
            import deepface  # noqa: F401
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Face backend requires the 'face' extra. Install with: pip install 'tasnif[face]'"
            ) from exc
        from .face_backend import FaceEmbedder

        return FaceEmbedder(**kwargs)

    raise ValueError(f"Unknown embedder backend: {name!r}. Available: {list_backends()}")
