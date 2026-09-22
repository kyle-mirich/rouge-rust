# Architecture and validation

## Code map

| File | Responsibility |
| --- | --- |
| `src/lib.rs` | Module wiring; forbids unsafe Rust in this crate |
| `src/scorer.rs` | Pure Rust tokenization, n-gram overlap, and LCS |
| `src/python.rs` | PyO3 conversion, result classes, GIL boundaries, Rayon batches |
| `fast_rouge.pyi` | Public Python type information |
| `tests/` | Installed-extension behavior and differential tests |
| `scripts/check_release.py` | Version, artifact, and wheel-matrix validation |

The Cargo package is the build unit for a Python extension (`cdylib`); this
repository does not publish a separate Rust crate API to crates.io.

## Scoring flow

1. PyO3 extracts strings; batch functions validate matching input lengths.
2. The binding detaches from Python while computing scores.
3. Rust lowercases Unicode text and splits on non-ASCII-alphanumeric characters.
   The ASCII path normalizes and records token spans in one pass. Borrowed spans
   avoid allocating a separate string for every token.
4. Unigram and bigram hash maps count reference occurrences. Prediction matches
   consume those counts so overlap cannot exceed either input's multiplicity.
5. ROUGE-L computes the longest common subsequence with two rolling rows, sized
   to the shorter token sequence.
6. The binding reattaches to Python to allocate result objects.

Batch pairs run independently on Rayon's shared pool. Indexed parallel iterators
preserve input order. The flat API writes directly into nine preallocated vectors
through disjoint mutable references; property getters later copy them to Python
lists. The scoring core retains no text or score cache across calls.

N-gram scoring is expected linear time with hash maps; ROUGE-L remains quadratic
in the two token lengths. `FxHashMap` prioritizes throughput and is not a hardened
hash table for adversarial input. Services should enforce input-size and request
limits appropriate to their workloads.

## Validation

- Rust tests cover tokenization, repeated n-grams, arbitrary `n`, empty inputs,
  and text/token API agreement. An independent brute-force subsequence oracle
  checks all pairs of binary token sequences up to length five.
- Python tests compare all three entrypoints exactly with `rouge-score==0.1.2`
  on curated edge cases and seeded randomized pairs. API tests cover ordering,
  input validation, read-only properties, list copies, version metadata, and
  concurrent Python callers with a deadlock timeout.
- CI checks formatting, Clippy, Python lint, Rust documentation, and installed
  type stubs. It tests both stable Rust and the minimum supported Rust version.
- Each released CPython/platform wheel is installed and tested on its native
  platform. The source distribution is also built, installed, and tested.

See [releasing.md](releasing.md) for the publication gates and recovery process.
