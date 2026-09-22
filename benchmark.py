"""Reproducible, fully validated throughput comparison; see docs/benchmarking.md."""

from __future__ import annotations

import argparse
import os
import platform
from importlib.metadata import version
from random import Random
from statistics import median
from time import perf_counter

from rouge_score import rouge_scorer

import fast_rouge

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
FIELDS = ("precision", "recall", "fmeasure")
VOCABULARY = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_pairs(count: int, min_tokens: int = 6, max_tokens: int = 18):
    rng = Random(0)
    references, predictions = [], []
    for _ in range(count):
        tokens = rng.choices(VOCABULARY, k=rng.randint(min_tokens, max_tokens))
        references.append(" ".join(tokens))
        for _ in range(max(1, len(tokens) // 4)):
            tokens[rng.randrange(len(tokens))] = rng.choice(VOCABULARY)
        predictions.append(" ".join(tokens))
    return references, predictions


def materialize_columns(result):
    # Each access copies a Vec to Python; do it once per column, never per row.
    return {
        f"{metric}_{field}": getattr(result, f"{metric}_{field}")
        for metric in ROUGE_TYPES
        for field in FIELDS
    }


def validate_results(baseline, batch, columns):
    if len(batch) != len(baseline) or any(len(c) != len(baseline) for c in columns.values()):
        raise RuntimeError("benchmark validation failed: result lengths differ")
    for index, expected in enumerate(baseline):
        for metric in ROUGE_TYPES:
            for field in FIELDS:
                value = getattr(expected[metric], field)
                if getattr(batch[index][metric], field) != value:
                    raise RuntimeError(f"batch mismatch: {metric}.{field} at index {index}")
                if columns[f"{metric}_{field}"][index] != value:
                    raise RuntimeError(f"flat mismatch: {metric}.{field} at index {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=positive_int, default=os.environ.get("PAIR_COUNT", "10000"))
    parser.add_argument("--repeats", type=positive_int, default=os.environ.get("REPEATS", "3"))
    parser.add_argument("--min-tokens", type=positive_int, default=6)
    parser.add_argument("--max-tokens", type=positive_int, default=18)
    args = parser.parse_args()
    if args.min_tokens > args.max_tokens:
        parser.error("--min-tokens cannot exceed --max-tokens")

    references, predictions = make_pairs(args.pairs, args.min_tokens, args.max_tokens)
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False)
    warmup = min(100, args.pairs)
    for reference, prediction in zip(references[:warmup], predictions[:warmup]):
        scorer.score(reference, prediction)
    fast_rouge.score_batch(references[:warmup], predictions[:warmup])
    materialize_columns(fast_rouge.score_batch_flat(references[:warmup], predictions[:warmup]))

    timings = {
        "rouge-score loop": [],
        "score_batch": [],
        "score_batch_flat": [],
        "score_batch_flat + Python lists": [],
    }
    for _ in range(args.repeats):
        start = perf_counter()
        baseline = [scorer.score(r, p) for r, p in zip(references, predictions)]
        timings["rouge-score loop"].append(perf_counter() - start)
        start = perf_counter()
        batch = fast_rouge.score_batch(references, predictions)
        timings["score_batch"].append(perf_counter() - start)
        start = perf_counter()
        flat = fast_rouge.score_batch_flat(references, predictions)
        timings["score_batch_flat"].append(perf_counter() - start)
        columns = materialize_columns(flat)
        timings["score_batch_flat + Python lists"].append(perf_counter() - start)
        validate_results(baseline, batch, columns)
        # Keep destruction of previous results out of the next iteration's timings.
        del baseline, batch, flat, columns

    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"platform: {platform.platform()}; CPUs: {os.cpu_count()}")
    print(f"rouge-rust: {fast_rouge.__version__}; rouge-score: {version('rouge-score')}")
    print(f"RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'automatic')}")
    print(
        f"pairs: {args.pairs}; repeats: {args.repeats}; tokens: {args.min_tokens}-{args.max_tokens}"
    )
    baseline_seconds = median(timings["rouge-score loop"])
    for label, runs in timings.items():
        seconds = median(runs)
        speedup = baseline_seconds / seconds if seconds else float("inf")
        print(f"{label}: {seconds:.6f}s median ({speedup:.2f}x)")
    print("validation: every metric in every output matches rouge-score on every repeat")


if __name__ == "__main__":
    main()
