"""Tasnif — unsupervised image clustering with modern deep embeddings."""

from __future__ import annotations

from ._version import __version__
from .clustering import KMeansFit, fit_kmeans, predict_kmeans
from .core import TasnifClusterer, cluster_directory
from .embeddings import Embedder, create_embedder, list_backends, register_embedder
from .exceptions import (
    BackendNotAvailableError,
    NoImagesFoundError,
    NotFittedError,
    TasnifError,
)
from .export import export_clusters
from .io import discover_images, iter_images, load_image
from .logging import configure_logging, get_logger
from .reduction import reduce_pca
from .types import ClusterResult, ExportManifest
from .visualization import make_cluster_grid

__all__ = [
    "BackendNotAvailableError",
    "ClusterResult",
    "Embedder",
    "ExportManifest",
    "KMeansFit",
    "NoImagesFoundError",
    "NotFittedError",
    "TasnifClusterer",
    "TasnifError",
    "__version__",
    "cluster_directory",
    "configure_logging",
    "create_embedder",
    "discover_images",
    "export_clusters",
    "fit_kmeans",
    "get_logger",
    "iter_images",
    "list_backends",
    "load_image",
    "make_cluster_grid",
    "predict_kmeans",
    "reduce_pca",
    "register_embedder",
]
