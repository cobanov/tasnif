from __future__ import annotations


class TasnifError(Exception):
    """Base class for all tasnif errors."""


class NotFittedError(TasnifError):
    """Raised when an operation requires a fitted clusterer."""


class BackendNotAvailableError(TasnifError):
    """Raised when an embedding backend's optional dependency is missing."""


class NoImagesFoundError(TasnifError):
    """Raised when no images are discovered in the input source."""
