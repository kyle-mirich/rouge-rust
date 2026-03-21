# Contributing to fast_rouge

Thanks for contributing.

## Development setup

```bash
source "$HOME/.cargo/env"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip maturin pytest rouge-score twine
.venv/bin/maturin develop --release
```

## Before opening a PR

Run the full validation suite:

```bash
source "$HOME/.cargo/env"
cargo test
.venv/bin/pytest -q
PAIR_COUNT=100000 REPEATS=3 .venv/bin/python benchmark.py
```

## Contribution guidelines

- Keep Python parity intact for `score()`, `score_batch()`, and `score_batch_flat()`.
- Prefer measured optimizations over speculative refactors.
- Add or update tests for behavior changes.
- Do not commit local build artifacts such as `dist/`, `.venv/`, or `target/`.
- Do not commit secrets, tokens, or machine-specific configuration.

## Release checklist

```bash
source "$HOME/.cargo/env"
.venv/bin/maturin build --release --sdist -o dist
.venv/bin/python -m twine check dist/*
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
