# Project brief

**rouge-rust** is Kyle Mirich's independent, MIT-licensed evaluation library. It
brings Rust/PyO3 computation to Python pipelines for ROUGE-1, ROUGE-2, and ROUGE-L,
with parallel ordered batches and a column-oriented result API. The project shows
native/Python integration, evaluation correctness, memory-conscious algorithms,
and reproducible engineering evidence.

- [Public source and quick start](https://github.com/kyle-mirich/rouge-rust)
- [Published package, version 0.1.12](https://pypi.org/project/rouge-rust/0.1.12/)
- [Runnable local demo](demo.md) and [example source](../examples/score_jsonl.py)
- [Architecture](flows.md), [scoring contract](correctness.md), and [benchmark verification](benchmark-results.md)

## Résumé wording supported by this repository

- Built a Rust/PyO3 evaluation library for Python with ROUGE-1, ROUGE-2, and ROUGE-L scoring, GIL release, and ordered parallel batch APIs.
- Verified all three Python APIs against Google's unstemmed ROUGE implementation, including exhaustive comparisons on 961 token-sequence pairs and independent metric regression tests.
- Built a reproducible benchmark harness with exact-reference validation, fixed thread settings, seeded input fingerprints, and raw timing reports.

These are independent-project accomplishments, not employer deliverables. Timing
experiments in this pass ran on a shared workstation under significant load, so
no performance number or speedup claim is recommended for a résumé.

## Evidence and practical limits

| Claim | Inspectable evidence |
| --- | --- |
| Native computation and GIL release | [`src/python.rs`](../src/python.rs) and [architecture](flows.md) |
| Clipped n-gram counts and rolling-row LCS | [`src/scorer.rs`](../src/scorer.rs) |
| Exact supported-reference comparisons | [`tests/test_parity.py`](../tests/test_parity.py), [`tests/test_semantics.py`](../tests/test_semantics.py) |
| Local demo and failure handling | [`examples/score_jsonl.py`](../examples/score_jsonl.py), [`tests/test_tools.py`](../tests/test_tools.py) |
| Benchmark validation, input fingerprints, measurement limits | [Benchmark verification](benchmark-results.md), [`benchmark.py`](../benchmark.py) |
| Distribution and typing validation | [`scripts/check_release.py`](../scripts/check_release.py), [release guide](releasing.md) |

On 2026-10-01, local verification passed 69 Python tests, 13 Rust tests on stable
Rust and Rust 1.88, Clippy, formatting/lint, Rust docs, installed stub validation,
and fresh wheel/source installs on CPython 3.13/macOS ARM64. The full platform
wheel matrix was not rerun for this tooling/documentation pass. The existing
[0.1.12 release](https://github.com/kyle-mirich/rouge-rust/releases/tag/v0.1.12)
contains the published wheels; no new package release accompanies this pass.

The library remains alpha. It targets Google's default ASCII-oriented tokenizer
with stemming disabled and does not implement ROUGE-Lsum, multi-reference
selection, corpus aggregation, or confidence intervals. Lexical overlap does not
establish factual correctness or semantic similarity. No adoption, customer,
download, or production-scale claim is supported here. The demo runs locally;
there is no hosted interactive deployment.
