# rouge-rust

`rouge-rust` is a Rust-powered replacement for Google's `rouge-score` Python package.
It provides matching ROUGE-1, ROUGE-2, and ROUGE-L metrics through a PyO3 extension module,
with fast batch APIs for large-scale evaluation workloads.

## Features

- ROUGE-1, ROUGE-2, and ROUGE-L scoring
- `score()` for drop-in per-pair scoring
- `score_batch()` for list-of-dicts batch scoring
- `score_batch_flat()` for high-throughput struct-of-arrays batch scoring
- Rust core optimized for large datasets

## Installation

```bash
uv add rouge-rust
```

## Usage

```python
import fast_rouge

single = fast_rouge.score("the cat sat", "the cat sat")
print(single["rouge1"].fmeasure)

batch = fast_rouge.score_batch(
    ["the cat sat", "hello world"],
    ["the dog sat", "hello there"],
)
print(batch[0]["rougeL"].precision)

flat = fast_rouge.score_batch_flat(
    ["the cat sat", "hello world"],
    ["the dog sat", "hello there"],
)
print(flat.rouge1_fmeasure[0])
```

## Local setup

Create a local environment and install the package in editable mode:

```bash
uv venv
uv pip install -e .[dev]
uv run maturin develop --release
```

Run the Rust tests:

```bash
cargo test
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Release

Build a source distribution and wheel:

```bash
uvx maturin build --release --sdist -o dist
```

Validate the distributions:

```bash
uvx twine check dist/*
```

Upload to PyPI:

```bash
uvx maturin upload dist/*
```

## GitHub Actions release flow

This repo includes a GitHub Actions pipeline in `.github/workflows/release.yml`.

- pushes and pull requests to `main` and `develop` build and test wheels
- tags matching `v*` build release artifacts
- tag builds publish to PyPI and create a GitHub Release

### One-time PyPI setup

Configure PyPI trusted publishing for:

- owner: `kyle-mirich`
- repository: `rouge-rust`
- workflow: `release.yml`
- environment: `pypi`

After that, creating and pushing a tag such as `v0.1.0` will trigger publishing.
