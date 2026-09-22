# Benchmarking

Use a release build and record the environment along with any timings:

```bash
uv venv --python 3.13
uv pip install -e '.[dev]'
uv run --extra dev python benchmark.py --pairs 10000 --repeats 3
```

`uv pip install` builds an optimized wheel. If using maturin directly, pass
`--release`; debug-build comparisons are not meaningful.

## Method

The harness uses a seeded random generator and a fixed 24-word ASCII vocabulary.
By default, each reference contains 6–18 tokens. Roughly a quarter of positions
are selected for replacement to create its prediction; replacement can choose
the same token or revisit a position. This is a synthetic short-text workload,
not a representative natural-language corpus.

Every path is warmed up before timing. Three repeats are the default. Reported
timings are medians and exclude input generation, result validation, and result
destruction. The measurements include:

1. A Python loop over `rouge-score` with stemming disabled.
2. `score_batch`, including conversion to Python dictionaries and Score objects.
3. `score_batch_flat`, returning vectors stored in the extension object.
4. The same flat call plus materialization of all nine columns as Python lists.

Every field of every output is compared exactly against the reference on every
repeat. A mismatch aborts the benchmark. Validation occurs outside timed regions.
Thread count, CPU count, platform, Python version, and package versions are printed.

## Change the workload

```bash
# The original environment-variable interface remains supported.
PAIR_COUNT=100000 REPEATS=5 uv run --extra dev python benchmark.py

# Longer sequences increase the cost of quadratic ROUGE-L.
uv run --extra dev python benchmark.py --pairs 1000 --repeats 3 --min-tokens 100 --max-tokens 200

# Fix parallelism before the first batch call initializes the pool.
RAYON_NUM_THREADS=1 uv run --extra dev python benchmark.py --pairs 10000
```

Counts, repeats, and token limits must be positive; minimum length cannot exceed
maximum length. `--help` lists all options.

## Interpret results

This compares a sequential Python implementation with parallel native code; it
measures both implementation efficiency and parallelism. Tiny batches may not
amortize thread scheduling and allocation. Long sequences are dominated by LCS.
Flat results defer Python allocation, so use the materialized timing when your
application will consume all columns. If it needs one column, measure that path.

Do not generalize a speedup from this workload to all datasets. When proposing
an optimization, report before/after timings with identical hardware, inputs,
Python version, build mode, thread count, and repeat count. Also run the parity
suite; speed is useful only if scores stay correct.
