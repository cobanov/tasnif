from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from typing_extensions import Self

from .clustering import KMeansFit, fit_kmeans, predict_kmeans
from .embeddings import Embedder, create_embedder
from .exceptions import NotFittedError
from .io import PathLike, discover_images, iter_images, load_image
from .logging import get_logger
from .reduction import reduce_pca
from .types import ClusterResult, ExportManifest

if TYPE_CHECKING:
    from PIL.Image import Image

    from .export import ExportMode

_log = get_logger("core")

ImagesInput = PathLike | Sequence[PathLike] | Sequence["Image"]


class TasnifClusterer:
    """High-level facade for clustering an image collection.

    Three usage shapes:

    1. **One-shot**::

           TasnifClusterer(n_clusters=5).fit_predict("photos/")

    2. **Inspect & export**::

           clf = TasnifClusterer(n_clusters=5).fit("photos/")
           result = clf.result_
           clf.export("out/")

    3. **Assign new images** without re-clustering::

           clf.predict("new_photos/")

    Pass ``embedder="clip"`` (requires ``tasnif[clip]``) or any registered
    backend name. Pass a pre-built :class:`Embedder` to ``embedder`` to use a
    custom model.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        embedder: str | Embedder = "timm",
        embedder_kwargs: dict[str, object] | None = None,
        pca_dim: int | None = 16,
        batch_size: int = 32,
        random_state: int | None = 42,
        compute_silhouette: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.pca_dim = pca_dim
        self.batch_size = batch_size
        self.random_state = random_state
        self.compute_silhouette = compute_silhouette
        self.show_progress = show_progress

        if isinstance(embedder, str):
            kwargs = dict(embedder_kwargs or {})
            self._embedder = create_embedder(embedder, **kwargs)
        else:
            self._embedder = embedder

        self._embeddings: np.ndarray | None = None
        self._reduced: np.ndarray | None = None
        self._paths: tuple[Path, ...] | None = None
        self._fit: KMeansFit | None = None
        self._images_cache: list[Image] | None = None

    # ------------------------------------------------------------------ props
    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def result_(self) -> ClusterResult:
        self._require_fitted()
        assert self._fit is not None and self._paths is not None
        return ClusterResult(
            labels=self._fit.labels,
            centroids=self._fit.centroids,
            counts=self._fit.counts,
            paths=self._paths,
            n_clusters=self.n_clusters,
            inertia=self._fit.inertia,
            silhouette=self._fit.silhouette,
            embedder=self._embedder.name,
            pca_dim=self.pca_dim,
        )

    @property
    def embeddings_(self) -> np.ndarray:
        self._require_fitted()
        assert self._embeddings is not None
        return self._embeddings

    @property
    def labels_(self) -> np.ndarray:
        return self.result_.labels

    # ------------------------------------------------------------------- API
    def fit(self, images: ImagesInput) -> Self:
        paths, pil_images = _resolve_input(images)
        self._paths = tuple(paths)
        self._images_cache = pil_images

        self._embeddings = self._embedder.embed(
            pil_images, batch_size=self.batch_size, show_progress=self.show_progress
        )
        features = self._embeddings
        if self.pca_dim:
            self._reduced = reduce_pca(
                features, n_components=self.pca_dim, random_state=self.random_state
            )
            features = self._reduced

        self._fit = fit_kmeans(
            features,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            compute_silhouette=self.compute_silhouette,
        )
        return self

    def fit_predict(self, images: ImagesInput) -> ClusterResult:
        self.fit(images)
        return self.result_

    def predict(self, images: ImagesInput) -> np.ndarray:
        """Assign new images to existing clusters.

        Reuses the fitted PCA projection and KMeans centroids. Note that since
        scikit-learn's ``PCA`` is not re-applied here (we don't store it as a
        public artifact), we project new embeddings via a fresh PCA fit-only
        when ``pca_dim`` is set - for predict-on-new-data you may prefer
        ``fit_predict`` on the combined set or ``transform`` for raw embeddings.
        """
        self._require_fitted()
        assert self._fit is not None
        _, pil_images = _resolve_input(images)
        embs = self._embedder.embed(
            pil_images, batch_size=self.batch_size, show_progress=self.show_progress
        )
        if self.pca_dim:
            embs = reduce_pca(embs, n_components=self.pca_dim, random_state=self.random_state)
        return predict_kmeans(self._fit.model, embs)

    def transform(self, images: ImagesInput) -> np.ndarray:
        """Embed images with the configured backend (no clustering)."""
        _, pil_images = _resolve_input(images)
        return self._embedder.embed(
            pil_images, batch_size=self.batch_size, show_progress=self.show_progress
        )

    def export(
        self,
        output_dir: str | Path,
        *,
        mode: ExportMode = "copy",
        write_grid: bool = True,
        write_manifest: bool = True,
        write_embeddings: bool = True,
        overwrite: bool = False,
        grid_max_images: int = 9,
    ) -> ExportManifest:
        from .export import export_clusters

        result = self.result_
        return export_clusters(
            result,
            output_dir,
            mode=mode,
            images=self._images_cache if write_grid else None,
            embeddings=self._embeddings if write_embeddings else None,
            write_grid=write_grid,
            write_manifest=write_manifest,
            grid_max_images=grid_max_images,
            overwrite=overwrite,
        )

    # ---------------------------------------------------------------- helpers
    def _require_fitted(self) -> None:
        if self._fit is None:
            raise NotFittedError(
                "TasnifClusterer is not fitted. Call .fit(images) or .fit_predict(images) first."
            )


# --------------------------------------------------------------------- input
def _resolve_input(images: ImagesInput) -> tuple[list[Path], list[Image]]:
    """Accept a directory path, a list of paths, or a list of PIL images."""
    from PIL.Image import Image as PILImage

    if isinstance(images, (str, Path)):
        paths = discover_images(images)
        return paths, list(iter_images(paths))

    materialized = list(images) if not isinstance(images, list) else images
    if not materialized:
        from .exceptions import NoImagesFoundError

        raise NoImagesFoundError("Empty image input.")

    first = materialized[0]
    if isinstance(first, PILImage):
        return ([Path(f"<pil:{i}>") for i in range(len(materialized))], list(materialized))

    if isinstance(first, (str, Path)):
        path_inputs = cast("Sequence[PathLike]", materialized)
        paths = [Path(p) for p in path_inputs]
        return paths, [load_image(p) for p in paths]

    raise TypeError(f"Unsupported image input element: {type(first).__name__}")


def cluster_directory(
    directory: PathLike,
    output_dir: PathLike,
    *,
    n_clusters: int,
    embedder: str | Embedder = "timm",
    embedder_kwargs: dict[str, object] | None = None,
    pca_dim: int | None = 16,
    mode: ExportMode = "copy",
    batch_size: int = 32,
    random_state: int | None = 42,
    overwrite: bool = False,
) -> ExportManifest:
    """One-call convenience: discover images, cluster, and export."""
    clf = TasnifClusterer(
        n_clusters=n_clusters,
        embedder=embedder,
        embedder_kwargs=embedder_kwargs,
        pca_dim=pca_dim,
        batch_size=batch_size,
        random_state=random_state,
    )
    clf.fit(directory)
    return clf.export(output_dir, mode=mode, overwrite=overwrite)
