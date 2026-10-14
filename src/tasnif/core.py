from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from typing_extensions import Self

from .clustering import KMeansFit, choose_n_clusters, fit_kmeans, predict_kmeans
from .embeddings import Embedder, create_embedder
from .exceptions import NoImagesFoundError, NotFittedError
from .io import ImageStream, PathLike, discover_images
from .logging import get_logger
from .reduction import fit_pca
from .types import ClusterResult, ExportManifest

if TYPE_CHECKING:
    from PIL.Image import Image
    from sklearn.decomposition import PCA

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

    ``n_clusters="auto"`` picks the number of clusters with the best silhouette
    score between ``k_min`` and ``k_max``; the chosen value is ``n_clusters_``.

    Images given as paths are streamed: they are decoded a few at a time on
    ``workers`` threads, scaled so their shorter side is at most
    ``max_image_size`` (``None`` keeps full resolution), and dropped once
    embedded. Unreadable files are skipped; ``skipped_`` lists the ones the
    last ``fit``, ``predict`` or ``transform`` call left out.
    """

    def __init__(
        self,
        n_clusters: int | Literal["auto"],
        *,
        embedder: str | Embedder = "timm",
        embedder_kwargs: dict[str, object] | None = None,
        pca_dim: int | None = 16,
        batch_size: int = 32,
        random_state: int | None = 42,
        compute_silhouette: bool = False,
        show_progress: bool = True,
        k_min: int = 2,
        k_max: int = 30,
        max_image_size: int | None = 512,
        workers: int | None = None,
    ) -> None:
        if n_clusters != "auto" and (not isinstance(n_clusters, int) or n_clusters < 1):
            raise ValueError(f"n_clusters must be a positive int or 'auto', got {n_clusters!r}")
        self.n_clusters = n_clusters
        self.k_min = k_min
        self.k_max = k_max
        self.max_image_size = max_image_size
        self.workers = workers
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
        self._pca: PCA | None = None
        self._paths: tuple[Path, ...] | None = None
        self._fit: KMeansFit | None = None
        self._images_cache: list[Image] | None = None
        self._k_scores: dict[int, float] | None = None
        self.skipped_: tuple[Path, ...] = ()

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
            n_clusters=len(self._fit.centroids),
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

    @property
    def n_clusters_(self) -> int:
        """The number of clusters used, which ``n_clusters="auto"`` chooses during ``fit``."""
        return self.result_.n_clusters

    @property
    def k_scores_(self) -> dict[int, float] | None:
        """Silhouette score per candidate ``k`` when ``n_clusters="auto"``, else ``None``."""
        self._require_fitted()
        return self._k_scores

    # ------------------------------------------------------------------- API
    def fit(self, images: ImagesInput) -> Self:
        paths, embeddings, pil_images = self._embed(images)
        self._paths = tuple(paths)
        self._embeddings = embeddings
        self._images_cache = pil_images

        features = embeddings
        self._pca = None
        if self.pca_dim:
            self._reduced, self._pca = fit_pca(
                features, n_components=self.pca_dim, random_state=self.random_state
            )
            features = self._reduced

        n_clusters = self.n_clusters
        self._k_scores = None
        if n_clusters == "auto":
            n_clusters, self._k_scores = choose_n_clusters(
                features, k_min=self.k_min, k_max=self.k_max, random_state=self.random_state
            )

        self._fit = fit_kmeans(
            features,
            n_clusters=n_clusters,
            random_state=self.random_state,
            compute_silhouette=self.compute_silhouette,
        )
        return self

    def fit_predict(self, images: ImagesInput) -> ClusterResult:
        self.fit(images)
        return self.result_

    def predict(self, images: ImagesInput) -> np.ndarray:
        """Assign new images to the fitted clusters.

        New embeddings go through the PCA projection fitted in ``fit`` before
        the nearest centroid is looked up. Unreadable files are skipped and
        listed in ``skipped_``; the result has one label per remaining image.
        """
        self._require_fitted()
        assert self._fit is not None
        _, embs, _ = self._embed(images)
        if self._pca is not None:
            embs = self._pca.transform(embs).astype(np.float32, copy=False)
        return predict_kmeans(self._fit.model, embs)

    def transform(self, images: ImagesInput) -> np.ndarray:
        """Embed images with the configured backend (no clustering)."""
        return self._embed(images)[1]

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
    def _embed(self, images: ImagesInput) -> tuple[list[Path], np.ndarray, list[Image] | None]:
        """Embed the input.

        Returns the paths that were embedded, their rows, and the images
        themselves when they were given as PIL images (``None`` for paths).
        """
        paths, source = _resolve_input(
            images, max_image_size=self.max_image_size, workers=self.workers
        )
        embeddings = self._embedder.embed(
            source, batch_size=self.batch_size, show_progress=self.show_progress
        )
        self.skipped_ = ()
        if isinstance(source, ImageStream):
            paths = source.loaded
            self.skipped_ = tuple(source.skipped)
            if source.skipped:
                _log.warning("Skipped %d unreadable image(s)", len(source.skipped))
            if not paths:
                raise NoImagesFoundError(f"None of the {len(source)} image(s) could be read.")
        if len(embeddings) != len(paths):
            raise RuntimeError(
                f"Embedder {self._embedder.name!r} returned {len(embeddings)} rows "
                f"for {len(paths)} images"
            )
        return paths, embeddings, None if isinstance(source, ImageStream) else source

    def _require_fitted(self) -> None:
        if self._fit is None:
            raise NotFittedError(
                "TasnifClusterer is not fitted. Call .fit(images) or .fit_predict(images) first."
            )


# --------------------------------------------------------------------- input
def _resolve_input(
    images: ImagesInput, *, max_image_size: int | None = None, workers: int | None = None
) -> tuple[list[Path], ImageStream | list[Image]]:
    """Accept a directory path, a list of paths, or a list of PIL images.

    Paths come back as a lazy :class:`ImageStream`; PIL images as they are.
    """
    from PIL.Image import Image as PILImage

    if isinstance(images, (str, Path)):
        paths = discover_images(images)
        return paths, ImageStream(paths, max_size=max_image_size, workers=workers)

    materialized = list(images) if not isinstance(images, list) else images
    if not materialized:
        raise NoImagesFoundError("Empty image input.")

    first = materialized[0]
    if isinstance(first, PILImage):
        pil_images = cast("list[Image]", materialized)
        return [Path(f"<pil:{i}>") for i in range(len(pil_images))], pil_images

    if isinstance(first, (str, Path)):
        path_inputs = cast("Sequence[PathLike]", materialized)
        paths = [Path(p) for p in path_inputs]
        return paths, ImageStream(paths, max_size=max_image_size, workers=workers)

    raise TypeError(f"Unsupported image input element: {type(first).__name__}")


def cluster_directory(
    directory: PathLike,
    output_dir: PathLike,
    *,
    n_clusters: int | Literal["auto"],
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
