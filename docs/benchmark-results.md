# Benchmark verification and measurement limits

On 2026-10-01, the benchmark harness was exercised against `rouge-score==0.1.2`
with its default tokenizer, stemming disabled, and exactly ROUGE-1, ROUGE-2,
ROUGE-L. All nine fields matched exactly for every output on every repeat in all
five paths: the Google loop, native single-pair loop, dict batch, deferred flat
batch, and flat batch with all Python lists materialized.

The experiments used an optimized local wheel on Apple M4/macOS ARM64, CPython
3.13.9, Rust 1.94.0, and maturin 1.15.0. The native core is unchanged from 0.1.12.
The tested harness source was clean commit
[`942d39e80c254a34ee3f7c3f4ab5cd6de5486553`](https://github.com/kyle-mirich/rouge-rust/commit/942d39e80c254a34ee3f7c3f4ab5cd6de5486553).

## What was validated

| Synthetic input | Token lengths per pair | Seed | Repeats per setting |
| --- | ---: | ---: | ---: |
| 10,000 short pairs | 6–18 | 0 | 5 |
| 1,000 longer pairs | 100–200 | 0 | 5 |
| One tiny pair | 6–18 | 0 | 25 |

Each experiment ran in a fresh process with a fixed Rayon setting. One- and
four-worker experiments completed before subsequent resource coordination capped
this task's checks at one worker. The paired datasets had identical fingerprints
across worker settings:

- Short: `be20a7e7e56de47b20052cd7369fb7b9b0ca0f2cab81fd74a0b42ae6eae85f2e`
- Long: `da2502e5801ffde53683123534f8a90e0525a5851fb7b21500bc427bbb2c60b9`
- Tiny: `a8f53b17453289668301890747067684b8a6059ed9c06bdbee0ed9061573cd48`

These hashes cover compact JSON of `[references, predictions]`, encoded as UTF-8.
The JSON reports retained the reference settings, every timing sample, timing
order, input digest, warmup count, environment, and source revision/dirty state.
They were kept as local verification artifacts, outside source control.

## Why no performance result is claimed

Other project tasks were active on this shared workstation, and resource
coordination reported substantial machine pressure during the work session.
CPU scheduling, power mode, frequency, thermals, and background load were not
controlled. These runs validate the harness and reference agreement; they do not
support a published speedup, idle-machine latency, or deployment-capacity claim.
No performance number from this pass belongs in a résumé.

Synthetic inputs also cannot establish performance on an application's real
corpus. There was no before/after optimization, memory/RSS measurement, energy
measurement, statistical significance test, or confidence interval. Input/output
conversion costs and parallelism must be part of a fair comparison.

## Reproduce under controlled conditions

Use the [development setup](../CONTRIBUTING.md) and [methodology](benchmarking.md).
Choose a quiet environment, record its load, CPU model, compiler, Python version,
build mode, package revisions and thread setting. Begin with one worker to
separate native implementation costs from parallelism. Run each configuration in
its own process and retain raw JSON reports outside the repository:

```bash
RAYON_NUM_THREADS=1 .venv/bin/python benchmark.py --pairs 10000 --repeats 5 --seed 0 --json-output /tmp/short-1.json
RAYON_NUM_THREADS=1 .venv/bin/python benchmark.py --pairs 1000 --min-tokens 100 --max-tokens 200 --repeats 5 --seed 0 --json-output /tmp/long-1.json
RAYON_NUM_THREADS=1 .venv/bin/python benchmark.py --pairs 1 --repeats 25 --seed 0 --json-output /tmp/tiny-1.json
```

Report both medians and variability. Include Python-list materialization when
that is what the application consumes. Tiny batches can expose scheduling and
allocation overhead; long sequences stress quadratic LCS; Unicode and larger
vocabularies may change preprocessing/hash costs. ROUGE-L still takes O(m × n)
time with O(min(m, n)) DP memory, and both batch inputs and outputs are eagerly
materialized. [Scoring/variant limits](correctness.md) apply to every comparison.
