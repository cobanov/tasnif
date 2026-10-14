from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def iter_batches(
    images: Iterable[Image],
    batch_size: int,
    *,
    desc: str,
    show_progress: bool,
) -> Iterator[list[Image]]:
    """Group ``images`` into lists of ``batch_size`` without materializing the input.

    A streamed source (:class:`tasnif.io.ImageStream`) is consumed as it loads,
    so only one batch is in memory at a time. Progress is reported per image.
    """
    from tqdm.auto import tqdm

    total = len(images) if isinstance(images, Sized) else None
    with tqdm(total=total, desc=desc, unit="img", disable=not show_progress) as bar:
        batch: list[Image] = []
        for image in images:
            batch.append(image)
            if len(batch) == batch_size:
                yield batch
                bar.update(len(batch))
                batch = []
        if batch:
            yield batch
            bar.update(len(batch))
