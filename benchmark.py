import os
from random import Random
from time import perf_counter

import fast_rouge
from rouge_score import rouge_scorer


PAIR_COUNT = int(os.environ.get("PAIR_COUNT", "10000"))
REPEATS = int(os.environ.get("REPEATS", "1"))
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
VOCABULARY = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]


def make_sentence(rng: Random) -> str:
    length = rng.randint(6, 18)
    return " ".join(rng.choices(VOCABULARY, k=length))


def make_pairs(count: int) -> tuple[list[str], list[str]]:
    rng = Random(0)
    references: list[str] = []
    predictions: list[str] = []

    for _ in range(count):
        reference = make_sentence(rng)
        prediction_tokens = reference.split()

        for _ in range(max(1, len(prediction_tokens) // 4)):
            index = rng.randrange(len(prediction_tokens))
            prediction_tokens[index] = rng.choice(VOCABULARY)

        references.append(reference)
        predictions.append(" ".join(prediction_tokens))

    return references, predictions


def score_tuple(score: object) -> tuple[float, float, float]:
    return (score.precision, score.recall, score.fmeasure)


def flat_score_tuple(batch_result: object, index: int, prefix: str) -> tuple[float, float, float]:
    return (
        getattr(batch_result, f"{prefix}_precision")[index],
        getattr(batch_result, f"{prefix}_recall")[index],
        getattr(batch_result, f"{prefix}_fmeasure")[index],
    )


def validate_results(
    baseline_scores: list[object],
    fast_scores: list[object],
    flat_scores: object,
) -> None:
    if len(baseline_scores) != len(fast_scores):
        raise RuntimeError("benchmark validation failed: result lengths differ")

    if len(flat_scores.rouge1_precision) != len(baseline_scores):
        raise RuntimeError("flat benchmark validation failed: result lengths differ")

    sample_indexes = sorted({0, len(baseline_scores) // 2, len(baseline_scores) - 1})

    for index in sample_indexes:
        baseline_score = baseline_scores[index]
        fast_score = fast_scores[index]

        for rouge_type in ROUGE_TYPES:
            baseline_tuple = score_tuple(baseline_score[rouge_type])
            fast_tuple = score_tuple(fast_score[rouge_type])
            flat_tuple = flat_score_tuple(flat_scores, index, rouge_type)

            if baseline_tuple != fast_tuple:
                raise RuntimeError(
                    f"benchmark validation failed for {rouge_type} at index {index}"
                )

            if baseline_tuple != flat_tuple:
                raise RuntimeError(
                    f"flat benchmark validation failed for {rouge_type} at index {index}"
                )


def main() -> None:
    references, predictions = make_pairs(PAIR_COUNT)
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False)

    def run_baseline() -> tuple[float, list[object]]:
        start = perf_counter()
        scores = [
            scorer.score(reference, prediction)
            for reference, prediction in zip(references, predictions)
        ]
        return perf_counter() - start, scores

    def run_dict_batch() -> tuple[float, list[object]]:
        start = perf_counter()
        scores = fast_rouge.score_batch(references, predictions)
        return perf_counter() - start, scores

    def run_flat_batch() -> tuple[float, object]:
        start = perf_counter()
        scores = fast_rouge.score_batch_flat(references, predictions)
        return perf_counter() - start, scores

    baseline_runs: list[float] = []
    dict_runs: list[float] = []
    flat_runs: list[float] = []
    baseline_scores = None
    fast_scores = None
    flat_scores = None

    for _ in range(REPEATS):
        baseline_seconds, baseline_scores = run_baseline()
        fast_seconds, fast_scores = run_dict_batch()
        flat_seconds, flat_scores = run_flat_batch()
        baseline_runs.append(baseline_seconds)
        dict_runs.append(fast_seconds)
        flat_runs.append(flat_seconds)

    validate_results(baseline_scores, fast_scores, flat_scores)

    baseline_seconds = min(baseline_runs)
    fast_seconds = min(dict_runs)
    flat_seconds = min(flat_runs)

    dict_speedup = baseline_seconds / fast_seconds if fast_seconds else float("inf")
    flat_speedup = baseline_seconds / flat_seconds if flat_seconds else float("inf")

    print(f"pair_count: {PAIR_COUNT}")
    print(f"repeats: {REPEATS}")
    print(f"rouge-score loop: {baseline_seconds:.4f}s")
    print(f"fast_rouge.score_batch: {fast_seconds:.4f}s")
    print(f"score_batch speedup: {dict_speedup:.2f}x")
    print(f"fast_rouge.score_batch_flat: {flat_seconds:.4f}s")
    print(f"score_batch_flat speedup: {flat_speedup:.2f}x")
    print("validation: sampled outputs match rouge-score")


if __name__ == "__main__":
    main()
