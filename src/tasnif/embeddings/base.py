from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns PIL images into a dense ``(N, dim)`` float32 matrix."""

    name: str

    @property
    def dim(self) -> int: ...

    @property
    def device(self) -> str: ...

    def embed(
        self,
        images: Sequence[Image] | Iterable[Image],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray: ...
