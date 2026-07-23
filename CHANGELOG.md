# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Updated the GitHub Release action to its Node 24-based generation.

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

[Unreleased]: https://github.com/kyle-mirich/rouge-rust/compare/v0.1.11...HEAD
[0.1.11]: https://github.com/kyle-mirich/rouge-rust/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/kyle-mirich/rouge-rust/releases/tag/v0.1.10
