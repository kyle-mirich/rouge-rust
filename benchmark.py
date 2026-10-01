"""Reproducible, fully validated throughput comparison; see docs/benchmarking.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
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


def make_pairs(count: int, min_tokens: int = 6, max_tokens: int = 18, seed: int = 0):
    rng = Random(seed)
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


def source_revision():
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
        return {"commit": revision, "working_tree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=positive_int, default=os.environ.get("PAIR_COUNT", "10000"))
    parser.add_argument("--repeats", type=positive_int, default=os.environ.get("REPEATS", "3"))
    parser.add_argument("--min-tokens", type=positive_int, default=6)
    parser.add_argument("--max-tokens", type=positive_int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, help="save metadata and every timing sample")
    args = parser.parse_args()
    if args.min_tokens > args.max_tokens:
        parser.error("--min-tokens cannot exceed --max-tokens")

    references, predictions = make_pairs(args.pairs, args.min_tokens, args.max_tokens, args.seed)
    input_digest = hashlib.sha256(
        json.dumps([references, predictions], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False)
    warmup = min(100, args.pairs)
    for reference, prediction in zip(references[:warmup], predictions[:warmup]):
        scorer.score(reference, prediction)
        fast_rouge.score(reference, prediction)
    fast_rouge.score_batch(references[:warmup], predictions[:warmup])
    materialize_columns(fast_rouge.score_batch_flat(references[:warmup], predictions[:warmup]))

    paths = {
        "rouge-score loop": lambda: [scorer.score(r, p) for r, p in zip(references, predictions)],
        "score loop": lambda: [fast_rouge.score(r, p) for r, p in zip(references, predictions)],
        "score_batch": lambda: fast_rouge.score_batch(references, predictions),
        "score_batch_flat": lambda: fast_rouge.score_batch_flat(references, predictions),
        "score_batch_flat + Python lists": lambda: materialize_columns(
            fast_rouge.score_batch_flat(references, predictions)
        ),
    }
    timings = {label: [] for label in paths}
    labels = list(paths)
    orders = []
    for repeat in range(args.repeats):
        offset = repeat % len(labels)
        order = labels[offset:] + labels[:offset]
        orders.append(order)
        results = {}
        for label in order:
            start = perf_counter()
            result = paths[label]()
            elapsed = perf_counter() - start
            timings[label].append(elapsed)
            results[label] = result
        baseline = results["rouge-score loop"]
        validate_results(
            baseline, results["score loop"], results["score_batch_flat + Python lists"]
        )
        columns = materialize_columns(results["score_batch_flat"])
        validate_results(baseline, results["score_batch"], columns)
        # Keep result destruction outside subsequent timing intervals.
        del result, results, baseline, columns

    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"platform: {platform.platform()}; CPUs: {os.cpu_count()}")
    print(f"rouge-rust: {fast_rouge.__version__}; rouge-score: {version('rouge-score')}")
    print(f"RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'automatic')}")
    print(
        f"pairs: {args.pairs}; repeats: {args.repeats}; "
        f"tokens: {args.min_tokens}-{args.max_tokens}; "
        f"seed: {args.seed}; input SHA-256: {input_digest}"
    )
    baseline_seconds = median(timings["rouge-score loop"])
    for label, runs in timings.items():
        seconds = median(runs)
        speedup = baseline_seconds / seconds if seconds else float("inf")
        print(f"{label}: {seconds:.6f}s median ({speedup:.2f}x)")
    print("validation: every metric in every output matches rouge-score on every repeat")
    if args.json_output:
        report = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_revision(),
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpus": os.cpu_count(),
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "automatic"),
                "rouge_rust": fast_rouge.__version__,
                "rouge_score": version("rouge-score"),
            },
            "workload": {
                "pairs": args.pairs,
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "vocabulary": VOCABULARY,
                "input_sha256": input_digest,
                "warmup_pairs": warmup,
                "repeats": args.repeats,
            },
            "reference_settings": {
                "metrics": ROUGE_TYPES,
                "use_stemmer": False,
                "tokenizer": "default",
            },
            "timing_order": orders,
            "timings": {
                label: {
                    "seconds": runs,
                    "median_seconds": median(runs),
                    "pairs_per_second": args.pairs / median(runs),
                    "speedup_vs_reference": baseline_seconds / median(runs),
                }
                for label, runs in timings.items()
            },
            "validation": {"exact_match": True, "fields_per_pair_per_path": 9},
        }
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.json_output}")


if __name__ == "__main__":
    main()
