# Contributing to Tasnif

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/cobanov/tasnif
cd tasnif
uv sync --extra dev
uv run pre-commit install
```

## Checks

The CI runs all of these — please run them locally first:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Markers:

- `slow` — long-running tests
- `gpu` — requires a CUDA / MPS device
- `integration` — downloads pretrained model weights

CI runs only the default set (`-m "not integration and not slow and not gpu"`).
Run the full suite locally when changing embedder code:

```bash
uv run pytest -m ""
```

## Adding a new embedder backend

1. Implement the `Embedder` protocol in `src/tasnif/embeddings/<backend>.py`.
2. Wire it into `src/tasnif/embeddings/factory.py::create_embedder`.
3. Add tests under `tests/test_embeddings_factory.py`.
4. Add a section to `README.md`.

## Commit / PR style

- One topic per PR.
- Update `CHANGELOG.md` under "Unreleased".
- Keep public-API changes documented in the README.

## Release

Tags matching `v*` trigger `release.yml`, which publishes to PyPI via
trusted publishing. Bump `src/tasnif/_version.py` + `pyproject.toml`, tag, push.
