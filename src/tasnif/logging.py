from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOGGER_NAME = "tasnif"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    *,
    rich: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Attach a handler to the ``tasnif`` logger.

    Library code never touches ``logging.basicConfig`` — applications opt in by
    calling this function (or by configuring the ``tasnif`` logger themselves).
    """
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    if _CONFIGURED:
        return logger

    handler: logging.Handler
    if rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
                rich_tracebacks=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = _stderr_handler()
    else:
        handler = _stderr_handler()

    _clear_handlers(logger.handlers)
    logger.addHandler(handler)
    _CONFIGURED = True
    return logger


def _stderr_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    return handler


def _clear_handlers(handlers: Iterable[logging.Handler]) -> None:
    for handler in list(handlers):
        handler.close()
