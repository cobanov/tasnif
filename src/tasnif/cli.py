from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

try:
    import typer
except ImportError as exc:
    sys.stderr.write(
        "The 'tasnif' CLI requires the [cli] extra. Install with: pip install 'tasnif[cli]'\n"
    )
    raise SystemExit(1) from exc

from ._version import __version__
from .core import TasnifClusterer
from .embeddings import list_backends
from .logging import configure_logging

app = typer.Typer(
    name="tasnif",
    help="Cluster images into folders using deep embeddings, PCA and K-Means.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tasnif {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Errors only.")] = False,
) -> None:
    if quiet:
        configure_logging(level="ERROR")
    elif verbose:
        configure_logging(level="DEBUG")
    else:
        configure_logging(level="INFO")


@app.command()
def cluster(
    source: Annotated[Path, typer.Argument(exists=True, help="Directory of images to cluster.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")] = Path(
        "./tasnif_out"
    ),
    n_clusters: Annotated[
        int, typer.Option("--clusters", "-k", min=2, help="Number of clusters.")
    ] = 5,
    embedder: Annotated[
        str, typer.Option("--embedder", "-e", help="Backend: timm or clip.")
    ] = "timm",
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Model name for the chosen backend (e.g. resnet50, ViT-B-32)."
        ),
    ] = None,
    pca_dim: Annotated[int, typer.Option("--pca", help="PCA dim (0 disables PCA).")] = 16,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 32,
    mode: Annotated[str, typer.Option("--mode", help="copy | symlink | move | none")] = "copy",
    device: Annotated[str, typer.Option("--device", help="auto | cpu | cuda | mps")] = "auto",
    random_state: Annotated[int, typer.Option("--seed")] = 42,
    silhouette: Annotated[
        bool, typer.Option("--silhouette", help="Compute silhouette score.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Allow non-empty output dir.")
    ] = False,
    no_grid: Annotated[bool, typer.Option("--no-grid", help="Skip cluster preview grids.")] = False,
    no_embeddings: Annotated[
        bool, typer.Option("--no-embeddings", help="Don't save embeddings.npy")
    ] = False,
) -> None:
    """Cluster images in SOURCE into OUTPUT/cluster_X folders."""
    embedder_kwargs: dict[str, object] = {"device": device}
    if model:
        embedder_kwargs["model"] = model

    clf = TasnifClusterer(
        n_clusters=n_clusters,
        embedder=embedder,
        embedder_kwargs=embedder_kwargs,
        pca_dim=pca_dim if pca_dim > 0 else None,
        batch_size=batch_size,
        random_state=random_state,
        compute_silhouette=silhouette,
    )
    clf.fit(source)
    manifest = clf.export(
        output,
        mode=mode,  # type: ignore[arg-type]
        write_grid=not no_grid,
        write_embeddings=not no_embeddings,
        overwrite=overwrite,
    )
    typer.echo(f"\n✓ Clustered {manifest.n_files} images into {manifest.n_clusters} clusters")
    typer.echo(f"  Output: {manifest.output_dir}")
    if manifest.manifest_csv:
        typer.echo(f"  Manifest: {manifest.manifest_csv}")


@app.command()
def embed(
    source: Annotated[Path, typer.Argument(exists=True, help="Directory of images.")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("./embeddings.npy"),
    embedder: Annotated[str, typer.Option("--embedder", "-e")] = "timm",
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Compute embeddings for a directory and save them as .npy + .json metadata."""
    import numpy as np

    from .embeddings import create_embedder
    from .io import discover_images, iter_images

    kwargs: dict[str, object] = {"device": device}
    if model:
        kwargs["model"] = model
    backend = create_embedder(embedder, **kwargs)

    paths = discover_images(source)
    embs = backend.embed(list(iter_images(paths)), batch_size=batch_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embs)
    meta = output.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "embedder": backend.name,
                "dim": backend.dim,
                "n_samples": int(embs.shape[0]),
                "paths": [str(p) for p in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"✓ Saved {embs.shape} -> {output}")
    typer.echo(f"  Metadata: {meta}")


@app.command()
def backends() -> None:
    """List available embedding backends."""
    for name in list_backends():
        typer.echo(name)


if __name__ == "__main__":
    app()
