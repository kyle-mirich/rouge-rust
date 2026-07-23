# rouge-rust

[![CI](https://github.com/kyle-mirich/rouge-rust/actions/workflows/release.yml/badge.svg)](https://github.com/kyle-mirich/rouge-rust/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/rouge-rust)](https://pypi.org/project/rouge-rust/)
[![Python](https://img.shields.io/pypi/pyversions/rouge-rust)](https://pypi.org/project/rouge-rust/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`rouge-rust` provides fast ROUGE-1, ROUGE-2, and ROUGE-L scoring to Python through a Rust extension. Its scores match `rouge-score` with stemming disabled for the supported metrics, and its batch APIs avoid a Python loop for evaluation workloads.

## Features

- ROUGE-1 and ROUGE-2 n-gram overlap scores
- ROUGE-L longest-common-subsequence scores
- Single-pair and parallel batch APIs
- A column-oriented batch result for analysis pipelines
- Exact-reference tests against `rouge-score`
- Prebuilt wheels for common CPython platforms, with source builds as a fallback

## Installation

Install the published package from PyPI:

```bash
python -m pip install rouge-rust
```

The distribution is named `rouge-rust`; the Python module is named `fast_rouge`.

## Quick start

```python
import fast_rouge

scores = fast_rouge.score("the cat sat", "the cat sat")
print(scores["rouge1"].fmeasure)  # 1.0
print(scores["rouge2"].precision)  # 1.0
print(scores["rougeL"].recall)  # 1.0
```

## Batch APIs

`score_batch()` returns the same nested score objects as `score()`:

```python
results = fast_rouge.score_batch(
    ["the cat sat", "hello world"],
    ["the dog sat", "hello there"],
)
print(results[0]["rougeL"].fmeasure)
```

`score_batch_flat()` returns one numeric list per metric field:

```python
result = fast_rouge.score_batch_flat(
    ["the cat sat", "hello world"],
    ["the dog sat", "hello there"],
)
print(result.rouge1_precision)
print(result.rougeL_fmeasure)
```

Both batch functions require equal-length reference and prediction lists and raise `ValueError` otherwise.

## Supported behavior and limitations

This project implements a focused subset of `rouge-score`: `rouge1`, `rouge2`, and `rougeL` with stemming disabled. It does not implement stemming, `rougeLsum`, bootstrap aggregation, or the full `rouge-score` class API. Tokenization follows the reference package's lowercase ASCII-alphanumeric behavior; non-ASCII characters act as token boundaries.

ROUGE-L uses dynamic programming and memory proportional to the shorter token sequence. Runtime remains proportional to the product of the two sequence lengths, so very long inputs can be expensive.

The package is currently an alpha release. See [CHANGELOG.md](CHANGELOG.md) for release history.

## Benchmarking

The repository includes a reproducible comparison against `rouge-score`:

```bash
PAIR_COUNT=10000 REPEATS=3 uv run --extra dev python benchmark.py
```

The benchmark validates sampled outputs before printing timings. Results depend on the machine, Python version, input shape, and pair count; run it locally instead of treating any single result as a general performance guarantee.

## Development

Prerequisites: Python 3.8 or newer, Rust stable, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kyle-mirich/rouge-rust.git
cd rouge-rust
uv venv
uv pip install -e ".[dev]"
```

Run the same quality checks used by CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
.venv/bin/python -m pytest -q
```

Build and validate release artifacts:

```bash
.venv/bin/maturin build --release --sdist -i .venv/bin/python -o dist
.venv/bin/twine check dist/*
```

## Project structure

- `src/scorer.rs`: tokenization and ROUGE algorithms
- `src/lib.rs`: PyO3 classes and Python-facing functions
- `tests/`: Python API and reference-parity tests
- `benchmark.py`: reproducible local benchmark harness
- `.github/workflows/release.yml`: CI, wheel builds, and trusted publishing
- [`docs/flows.md`](docs/flows.md): scoring and release flows

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and focused pull requests are welcome.

Report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
