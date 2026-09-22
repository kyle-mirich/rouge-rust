# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.12] - 2026-09-21

### Fixed

- Match reference tokenization for Unicode characters that lowercase to ASCII,
  including Kelvin sign (`K`) and dotted capital I (`İ`).
- Validate all benchmark outputs and include the cost of materializing flat
  columns, with warmup and median timings.

### Added

- Python type stubs, API reference, chunking examples, architecture notes,
  benchmark methodology, release guide, and code of conduct.
- Regression tests for Unicode, empty/repeated inputs, invalid types, result
  ownership, concurrent callers, and 400 seeded random input pairs.
- An independent brute-force oracle for LCS and higher-order n-gram tests.
- Tested CPython 3.8 and 3.14 wheels across all five platforms, plus native
  Linux ARM64 wheel testing and source-distribution installation tests.
- Release gates for tag/version consistency, type information, complete artifact
  coverage, metadata validity, and minimum Rust 1.88 compatibility.

### Changed

- Release the GIL during Rust computation in all three Python APIs.
- Preserve specialized unigram/bigram hash keys; simplify flat result allocation.
- Separate Python bindings from the Rust core, remove unused thread-local
  caches, and forbid unsafe code in this crate.
- Update PyO3 to 0.29.2 and pin GitHub Actions to reviewed commit SHAs.
- Publish GitHub Releases after PyPI succeeds, with curated notes and checksums.

## [0.1.11] - 2026-07-22

### Added

- Public package version metadata and empty-batch regression tests.
- Explicit documentation of the supported `rouge-score` subset and algorithmic limitations.
- Contributor-facing issue and pull request templates.

### Changed

- Reworked installation, API, benchmarking, and release documentation around verified commands.
- Removed unsupported drop-in-compatibility and fixed-performance implications from project copy.
- Updated PyO3 and the locked dependency graph to versions without known RustSec advisories.
- Updated official GitHub Actions to their current Node 24-based generations.

## [0.1.10] - 2026-04-03

### Changed

- Simplified the Python bridge and refreshed the public project presentation.

[Unreleased]: https://github.com/kyle-mirich/rouge-rust/compare/v0.1.12...HEAD
[0.1.12]: https://github.com/kyle-mirich/rouge-rust/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/kyle-mirich/rouge-rust/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/kyle-mirich/rouge-rust/releases/tag/v0.1.10
