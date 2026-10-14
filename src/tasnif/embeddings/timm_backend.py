from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging import get_logger
from ._batching import iter_batches
from ._torch_utils import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("embeddings.timm")

DEFAULT_MODEL = "convnext_small.fb_in22k"


class TimmEmbedder:
    """Embed images with any ``timm`` model.

    Uses the model's pooled feature output (``num_classes=0`` head removed),
    yielding an ``(N, dim)`` float32 numpy array. ``model_kwargs`` go to
    ``timm.create_model``, e.g. ``{"img_size": 224}`` for a ViT.

    The default is ConvNeXt-S pretrained on ImageNet-22k. Its features separate
    categories outside ImageNet-1k far better than ``resnet50``'s, which was the
    default before 0.3.0 (see the benchmark in the README).
    """

    name: str

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        pretrained: bool = True,
        normalize: bool = True,
        amp: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import timm
        import torch
        from timm.data import create_transform, resolve_data_config

        self.name = f"timm:{model}"
        self._device = resolve_device(device)
        self._normalize = normalize
        self._amp = amp and self._device == "cuda"

        self._torch = torch
        net = timm.create_model(model, pretrained=pretrained, num_classes=0, **(model_kwargs or {}))
        net.train(mode=False)
        net.to(self._device)
        self._model = net

        cfg = resolve_data_config({}, model=self._model)
        img_size = (model_kwargs or {}).get("img_size")
        if img_size:
            # The data config follows the pretrained resolution, not an override.
            height, width = (img_size, img_size) if isinstance(img_size, int) else img_size
            cfg["input_size"] = (cfg["input_size"][0], height, width)
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
        out: list[np.ndarray] = []
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if self._amp else nullcontext()
        )

        with torch.inference_mode(), autocast:
            for batch in iter_batches(
                images, batch_size, desc=f"embed[{self.name}]", show_progress=show_progress
            ):
                tensors = torch.stack([self._transform(img) for img in batch]).to(self._device)
                feats = self._model(tensors)
                if self._normalize:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.float().cpu().numpy())

        if not out:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.concatenate(out, axis=0)
