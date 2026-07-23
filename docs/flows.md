# Project flows

## Scoring flow

1. Python calls `score`, `score_batch`, or `score_batch_flat` in the PyO3 extension.
2. Rust normalizes each input to lowercase ASCII-alphanumeric tokens.
3. The scorer counts unigram and bigram overlap and computes a longest common subsequence for ROUGE-L.
4. Precision, recall, and F-measure values are returned as Python score objects.

Batch functions validate equal list lengths before using Rayon to compute independent pairs in parallel. The flat API stores each metric field in a separate list; it does not change the scoring algorithm.

## Validation flow

Rust unit tests cover tokenization, n-gram counting, caching, and LCS behavior. Python tests cover the extension boundary and compare all supported metrics with `rouge-score` using stemming disabled.

## Release flow

Pushes and pull requests run formatting, Clippy, Rust tests, and Python reference tests. A `v*` tag builds platform wheels and a source distribution. GitHub trusted publishing uploads those artifacts to PyPI and the workflow creates a GitHub Release.
