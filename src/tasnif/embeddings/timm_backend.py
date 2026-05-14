from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..logging import get_logger
from ._torch_utils import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("embeddings.timm")


class TimmEmbedder:
    """Embed images with any ``timm`` model.

    Uses the model's pooled feature output (``num_classes=0`` head removed),
    yielding an ``(N, dim)`` float32 numpy array.
    """

    name: str

    def __init__(
        self,
        model: str = "resnet50",
        *,
        device: str = "auto",
        pretrained: bool = True,
        normalize: bool = True,
        amp: bool = True,
    ) -> None:
        import timm
        import torch
        from timm.data import create_transform, resolve_data_config

        self.name = f"timm:{model}"
        self._device = resolve_device(device)
        self._normalize = normalize
        self._amp = amp and self._device == "cuda"

        self._torch = torch
        net = timm.create_model(model, pretrained=pretrained, num_classes=0)
        net.train(mode=False)
        net.to(self._device)
        self._model = net

        cfg = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**cfg)

        with torch.inference_mode():
            dummy = torch.zeros(1, *cfg["input_size"], device=self._device)
            feat = self._model(dummy)
        self._dim = int(feat.shape[-1])
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
                tensors = torch.stack([self._transform(img) for img in batch]).to(self._device)
                feats = self._model(tensors)
                if self._normalize:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.float().cpu().numpy())

        return np.concatenate(out, axis=0)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
