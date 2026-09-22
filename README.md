# rouge-rust

[![CI](https://github.com/kyle-mirich/rouge-rust/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/kyle-mirich/rouge-rust/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/rouge-rust)](https://pypi.org/project/rouge-rust/)
[![Python](https://img.shields.io/pypi/pyversions/rouge-rust)](https://pypi.org/project/rouge-rust/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kyle-mirich/rouge-rust/blob/main/LICENSE)

Fast **ROUGE-1, ROUGE-2, and ROUGE-L** scoring for Python, implemented in Rust.
Evaluate one pair or a batch with the same numerical results as Google's
[`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge)
for these metrics with stemming disabled and its default tokenizer.

- Parallel batch scoring with results in input order.
- A column-oriented API for analysis pipelines.
- Rust computation releases the Python GIL in all three APIs.
- Type hints, no runtime Python dependencies, and MIT licensing.
- Tested wheels for CPython 3.8–3.14 on Linux (x86-64 and ARM64), macOS
  (Intel and Apple Silicon), and Windows x64.

[API reference](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/api.md) · [Benchmarks](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/benchmarking.md) ·
[Architecture](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/flows.md) · [Contributing](https://github.com/kyle-mirich/rouge-rust/blob/main/CONTRIBUTING.md) · [Changelog](https://github.com/kyle-mirich/rouge-rust/blob/main/CHANGELOG.md)

## Install

```bash
python -m pip install rouge-rust
```

The distribution is named **`rouge-rust`**; import **`fast_rouge`**.
Prebuilt wheels require no Rust installation. Source builds require Rust 1.88+
and a C linker. Linux wheels target glibc 2.17+; musl, PyPy, and free-threaded
Python are not part of the tested wheel matrix.

## Quick start

```python
import fast_rouge

scores = fast_rouge.score("the cat sat", "the cat")
print(scores["rouge1"].precision)  # 1.0
print(scores["rouge1"].recall)  # 0.6666666666666666
print(scores["rouge1"].fmeasure)  # 0.8
```

The **reference comes first**, followed by the prediction. Each metric exposes
read-only `precision`, `recall`, and `fmeasure` floats between 0 and 1.

## Score batches

Choose the result shape that suits your pipeline:

| Function | Result | Use when |
| --- | --- | --- |
| `score(reference, prediction)` | Dict of three `Score` objects | Evaluating one pair |
| `score_batch(references, predictions)` | List of score dicts | Iterating over examples |
| `score_batch_flat(references, predictions)` | Nine metric columns | Computing statistics or creating a table |

```python
references = ["the cat sat", "hello world"]
predictions = ["the dog sat", "hello there"]

results = fast_rouge.score_batch(references, predictions)
print(results[0]["rougeL"].fmeasure)  # 0.6666666666666666

flat = fast_rouge.score_batch_flat(references, predictions)
f1 = flat.rouge1_fmeasure
print(f1)  # [0.6666666666666666, 0.5]
print(sum(f1) / len(f1))
```

Batch inputs must be equal-length sequences of strings; unequal lengths raise
`ValueError`. Empty batches are valid. Each flat property access creates a
**new Python list**, so save a column once before looping over its values.
The batch APIs eagerly consume their inputs; use chunks to bound memory.
See [API examples](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/api.md) for chunking and DataFrame conversion.

## Compatibility and limits

This is a focused scoring library, not the complete `rouge-score` class API.
It does not implement stemming, custom tokenizers, `rougeLsum`, multi-reference
selection, bootstrap confidence intervals, or aggregation.

Tokenization first lowercases Unicode text, then splits on characters outside
ASCII `a-z0-9`, matching the reference algorithm. For example, `Kelvin` becomes
`kelvin`, `İSTANBUL` becomes `i stanbul`, and `café` becomes `caf`. Text with no
remaining tokens receives zero for every score, including two empty inputs.
This tokenizer is not suitable for general multilingual evaluation.

ROUGE-L takes O(m × n) time and O(min(m, n)) working memory for token sequence
lengths m and n. Long documents can be expensive. Batch workers use Rayon's
shared thread pool; set `RAYON_NUM_THREADS` **before starting Python** to bound
CPU concurrency. The package is still in the 0.1 alpha series.

## Benchmark

From a development checkout:

```bash
uv run --extra dev python benchmark.py --pairs 10000 --repeats 3
```

The benchmark warms up all paths, compares every score with `rouge-score`, and
reports median timings. It measures flat scoring both before and after copying
all nine columns to Python lists. Hardware, input length, and thread count
matter; [the methodology](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/benchmarking.md) explains how to reproduce and
interpret results.

## Develop

Install Python, Rust 1.88+, and [`uv`](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/kyle-mirich/rouge-rust.git
cd rouge-rust
uv venv --python 3.13
uv pip install -e '.[dev]'
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest -q
```

Rebuild after Rust edits with `uv pip install -e '.[dev]'`. See
[CONTRIBUTING.md](https://github.com/kyle-mirich/rouge-rust/blob/main/CONTRIBUTING.md) for testing guidance and
[the release guide](https://github.com/kyle-mirich/rouge-rust/blob/main/docs/releasing.md) for packaging and trusted publishing.

## Community and license

Bug reports and focused pull requests are welcome. Follow the
[contribution guide](https://github.com/kyle-mirich/rouge-rust/blob/main/CONTRIBUTING.md) and [code of conduct](https://github.com/kyle-mirich/rouge-rust/blob/main/CODE_OF_CONDUCT.md).
Report vulnerabilities privately through [the security policy](https://github.com/kyle-mirich/rouge-rust/blob/main/SECURITY.md).

Released under the [MIT License](https://github.com/kyle-mirich/rouge-rust/blob/main/LICENSE).
