from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging import get_logger
from ._batching import iter_batches

if TYPE_CHECKING:
    from PIL.Image import Image

_log = get_logger("embeddings.face")


class FaceEmbedder:
    """Embed images by the face in them, with a face recognition model via ``deepface``.

    Each image is represented by its largest face, so photos of the same person
    land near each other regardless of background, pose or framing. An image with
    no detectable face is embedded whole by the same model, which keeps one row
    per image and one embedding space.

    Sensible default: ``Facenet512`` with the ``retinaface`` detector. Override
    with the ``model`` and ``detector`` kwargs (any DeepFace model and detector
    backend). DeepFace runs on PyTorch when TensorFlow is not installed, and picks
    its own device.
    """

    name: str

    def __init__(
        self,
        model: str = "Facenet512",
        *,
        detector: str = "retinaface",
        align: bool = True,
        device: str = "auto",
        normalize: bool = True,
    ) -> None:
        from deepface import DeepFace

        if device != "auto":
            _log.warning("DeepFace picks its own device; ignoring device=%r", device)

        self.name = f"face:{model}@{detector}"
        self._deepface = DeepFace
        self._model = model
        self._detector = detector
        self._align = align
        self._normalize = normalize

        client = DeepFace.build_model(model_name=model)
        self._dim = int(client.output_shape)
        self._device = str(getattr(client, "device", "auto"))
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
        rows: list[np.ndarray] = []
        for batch in iter_batches(
            images, batch_size, desc=f"embed[{self.name}]", show_progress=show_progress
        ):
            results: list[Any] = self._deepface.represent(
                img_path=[_to_bgr(img) for img in batch],
                model_name=self._model,
                detector_backend=self._detector,
                enforce_detection=False,
                align=self._align,
                max_faces=1,
            )
            # A single image comes back as its list of faces, not wrapped in a list.
            if results and isinstance(results[0], dict):
                results = [results]
            rows.extend(
                np.asarray(faces[0]["embedding"], dtype=np.float32).reshape(-1) for faces in results
            )

        out = np.stack(rows) if rows else np.empty((0, self._dim), dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out /= norms
        return out


def _to_bgr(img: Image) -> np.ndarray:
    """DeepFace reads arrays in OpenCV's BGR order."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
