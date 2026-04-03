# Contributing to rouge-rust

Thanks for contributing.

## Development setup

```bash
uv venv
uv pip install -e .[dev]
uv run maturin develop --release
```

## Before opening a PR

Run the full validation suite:

```bash
cargo test
uv run pytest -q
PAIR_COUNT=100000 REPEATS=3 uv run python benchmark.py
```

## Contribution guidelines

- Keep Python parity intact for `score()`, `score_batch()`, and `score_batch_flat()`.
- Prefer measured optimizations over speculative refactors.
- Add or update tests for behavior changes.
- Do not commit local build artifacts such as `dist/`, `.venv/`, or `target/`.
- Do not commit secrets, tokens, or machine-specific configuration.

## Release checklist

```bash
uv run maturin build --release --sdist -o dist
uv run twine check dist/*
```

## Automated releases

The repository is configured to publish from GitHub Actions on tags matching `v*`.

Typical flow:

```bash
git checkout main
git pull --ff-only
git tag v0.1.0
git push origin main --tags
```

PyPI publishing uses GitHub OIDC trusted publishing. No PyPI API token should be stored in the repository.
