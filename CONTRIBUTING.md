# Contributing to rouge-rust

Thank you for improving `rouge-rust`. Focused correctness, compatibility, portability, and measured performance changes are welcome.

## Development setup

Install Python 3.8 or newer, Rust stable, and `uv`, then run:

```bash
git clone https://github.com/kyle-mirich/rouge-rust.git
cd rouge-rust
uv venv
uv pip install -e ".[dev]"
```

No environment variables or external services are required.

## Verification

Run the full local check before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
.venv/bin/python -m pytest -q
```

For scoring changes, also run a local reference comparison:

```bash
PAIR_COUNT=10000 REPEATS=3 uv run --extra dev python benchmark.py
```

Build release artifacts when changing packaging or release configuration:

```bash
.venv/bin/maturin build --release --sdist -i .venv/bin/python -o dist
.venv/bin/twine check dist/*
```

## Contribution guidelines

- Preserve parity with `rouge-score` for ROUGE-1, ROUGE-2, and ROUGE-L with stemming disabled.
- Add a reference-parity test for scoring behavior changes.
- Prefer measured optimizations over speculative refactors.
- Update the README and changelog for user-visible changes.
- Do not commit wheels, virtual environments, generated benchmark data, or credentials.

## Releases

Tags matching `v*` trigger wheel and source-distribution builds. PyPI publishing uses GitHub OIDC trusted publishing; no PyPI API token belongs in the repository.
