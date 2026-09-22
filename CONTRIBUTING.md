# Contributing to rouge-rust

Focused correctness, compatibility, portability, and measured performance changes
are welcome. Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## Development setup

Install Rust 1.88+, a C linker, Python, and [uv](https://docs.astral.sh/uv/).
Python 3.13 is a convenient development default; CI tests CPython 3.8–3.14.

```bash
git clone https://github.com/kyle-mirich/rouge-rust.git
cd rouge-rust
uv venv --python 3.13
uv pip install -e '.[dev]'
```

Re-run `uv pip install -e '.[dev]'` after modifying Rust code. An already-imported
extension stays loaded, so restart Python or your notebook kernel after rebuilding.
No service credentials or datasets are needed for development.

## Verification

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest -q
```

Format changes with `cargo fmt` and `uv run --extra dev ruff format .`. For scoring
changes, run the [benchmark](docs/benchmarking.md). For typing changes, validate
the installed stubs with `uv run --with mypy python -m mypy.stubtest fast_rouge --allowlist scripts/stubtest-allowlist.txt`.

Rust unit tests deliberately omit the Python binding module to avoid requiring
libpython at link time. Passing `cargo test` alone is insufficient: the Python
suite tests the installed extension and compares it with Google's implementation.

## Contribution guidelines

- Preserve exact score parity with the default `rouge-score` tokenizer and
  stemming disabled for all supported metrics.
- Add a failing regression or reference-parity test before fixing scoring behavior.
- Keep changes focused; explain measured benefits of performance changes.
- Check both batch APIs, empty inputs, repeated tokens, and Unicode boundaries.
- Keep Python runtime and stub syntax compatible with the supported versions.
- Update API documentation, type stubs, and the changelog for public changes.
- Do not commit generated wheels, environments, benchmarks, or credentials.

Use the bug-report template for incorrect scores or installation failures. Include
minimal input strings, package/Python versions, operating system, and architecture.
For substantial features, explain the use case in an issue before implementing it.

## Packaging and releases

[The release guide](docs/releasing.md) covers build commands, validation gates,
versioning, and publication recovery. Releases use GitHub OIDC trusted publishing;
PyPI tokens do not belong in this repository.
