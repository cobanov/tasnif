from __future__ import annotations

from collections.abc import Iterable, Iterator
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


def load_image(path: PathLike) -> Image:
    """Open ``path`` as an RGB ``PIL.Image``. Closes file handles eagerly."""
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        return img.convert("RGB")


def iter_images(paths: Iterable[PathLike]) -> Iterator[Image]:
    """Yield RGB ``PIL.Image`` objects lazily — does not retain the full list."""
    for p in paths:
        yield load_image(p)
