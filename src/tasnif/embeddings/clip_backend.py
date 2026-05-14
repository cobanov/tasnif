from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..logging import get_logger
from ._torch_utils import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("embeddings.clip")


class CLIPEmbedder:
    """Embed images with the CLIP image encoder via ``open_clip``.

    Sensible default: ``ViT-B-32`` / ``laion2b_s34b_b79k``. Override with the
    ``model`` and ``pretrained`` kwargs.
    """

    name: str

    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        *,
        device: str = "auto",
        normalize: bool = True,
        amp: bool = True,
    ) -> None:
        import open_clip
        import torch

        self.name = f"clip:{model}@{pretrained}"
        self._device = resolve_device(device)
        self._normalize = normalize
        self._amp = amp and self._device == "cuda"

        self._torch = torch
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model, pretrained=pretrained, device=self._device
        )
        clip_model.train(mode=False)
        self._model = clip_model
        self._preprocess = preprocess
        self._dim = int(clip_model.visual.output_dim)
        _log.info("Loaded %s (dim=%d, device=%s)", self.name, self._dim, self._device)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def embed(
        self,
        images: Sequence[Image] | Iterable[Image],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        torch = self._torch
        materialized: list[Image] = list(images)
        if not materialized:
            return np.empty((0, self._dim), dtype=np.float32)

        from tqdm.auto import tqdm

        iterator: Iterable[int] = range(0, len(materialized), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"embed[{self.name}]", unit="batch")

        out: list[np.ndarray] = []
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if self._amp else _NullContext()
        )

        with torch.inference_mode(), autocast:
            for start in iterator:
                batch = materialized[start : start + batch_size]
                tensors = torch.stack([self._preprocess(img) for img in batch]).to(self._device)
                feats = self._model.encode_image(tensors)
                if self._normalize:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.float().cpu().numpy())

        return np.concatenate(out, axis=0)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
