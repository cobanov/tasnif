from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import NoImagesFoundError
from .logging import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
)

PathLike = str | Path

_log = get_logger("io")


def discover_images(
    source: PathLike | Iterable[PathLike],
    *,
    recursive: bool = True,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    sort: bool = True,
) -> list[Path]:
    """Discover image paths from a directory, a single file, or an iterable.

    Returns a deterministically ordered list (alphabetical by default).
    """
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    if isinstance(source, (str, Path)):
        sources: list[Path] = [Path(source)]
    else:
        sources = [Path(s) for s in source]

    found: list[Path] = []
    for src in sources:
        if src.is_dir():
            iterator = src.rglob("*") if recursive else src.iterdir()
            found.extend(p for p in iterator if p.is_file() and p.suffix.lower() in exts)
        elif src.is_file() and src.suffix.lower() in exts:
            found.append(src)
        else:
            _log.warning("Skipping non-existent or unsupported source: %s", src)

    if not found:
        raise NoImagesFoundError(
            f"No images with extensions {sorted(exts)} found in: "
            f"{', '.join(str(s) for s in sources)}"
        )

    if sort:
        found.sort()
    _log.info("Discovered %d image(s)", len(found))
    return found


def load_image(path: PathLike, *, max_size: int | None = None) -> Image:
    """Open ``path`` as an upright RGB ``PIL.Image``. Closes file handles eagerly.

    The EXIF orientation is applied, so phone photos are not sideways. With
    ``max_size``, an image whose shorter side is larger is scaled down to it; a
    JPEG is decoded at a reduced scale directly, which is faster and keeps a
    12 MP photo at about 1 MB in memory instead of 36 MB.
    """
    from PIL import Image as PILImage
    from PIL import ImageOps

    with PILImage.open(path) as img:
        if max_size:
            img.draft("RGB", (max_size, max_size))
        upright = ImageOps.exif_transpose(img)
        rgb = upright.convert("RGB")

    if max_size and min(rgb.size) > max_size:
        scale = max_size / min(rgb.size)
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        rgb = rgb.resize(size, PILImage.Resampling.BICUBIC, reducing_gap=3.0)
    return rgb


def iter_images(paths: Iterable[PathLike], *, max_size: int | None = None) -> Iterator[Image]:
    """Yield RGB ``PIL.Image`` objects lazily - does not retain the full list."""
    for p in paths:
        yield load_image(p, max_size=max_size)


class ImageStream:
    """Load images lazily and in order, decoding a few ahead on a thread pool.

    Only the images in flight are held in memory, so a folder of any size can be
    embedded. A file that cannot be read is skipped with a warning instead of
    stopping the run: after iterating, :attr:`loaded` lists the paths that were
    yielded, in order, and :attr:`skipped` the ones that were not.
    """

    def __init__(
        self,
        paths: Iterable[PathLike],
        *,
        max_size: int | None = None,
        workers: int | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.max_size = max_size
        self.workers = workers or min(8, os.cpu_count() or 1)
        self.loaded: list[Path] = []
        self.skipped: list[Path] = []

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Image]:
        self.loaded = []
        self.skipped = []
        remaining = iter(self.paths)
        with ThreadPoolExecutor(self.workers, thread_name_prefix="tasnif-load") as pool:
            pending: deque[tuple[Path, Future[Image]]] = deque()

            def submit_next() -> None:
                path = next(remaining, None)
                if path is not None:
                    pending.append((path, pool.submit(load_image, path, max_size=self.max_size)))

            for _ in range(self.workers * 2):
                submit_next()
            while pending:
                path, future = pending.popleft()
                submit_next()
                try:
                    image = future.result()
                except Exception as exc:
                    _log.warning("Skipping unreadable image %s: %s", path, exc)
                    self.skipped.append(path)
                    continue
                self.loaded.append(path)
                yield image
