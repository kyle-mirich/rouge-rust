from random import Random
from time import perf_counter

import fast_rouge
from rouge_score import rouge_scorer


PAIR_COUNT = 10_000
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


def main() -> None:
    references, predictions = make_pairs(PAIR_COUNT)
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False)

    baseline_start = perf_counter()
    baseline_scores = [
        scorer.score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]
    baseline_seconds = perf_counter() - baseline_start

    fast_start = perf_counter()
    fast_scores = fast_rouge.score_batch(references, predictions)
    fast_seconds = perf_counter() - fast_start

    flat_start = perf_counter()
    flat_scores = fast_rouge.score_batch_flat(references, predictions)
    flat_seconds = perf_counter() - flat_start

    if len(baseline_scores) != len(fast_scores):
        raise RuntimeError("benchmark validation failed: result lengths differ")

    if len(flat_scores.rouge1_precision) != len(baseline_scores):
        raise RuntimeError("flat benchmark validation failed: result lengths differ")

    dict_speedup = baseline_seconds / fast_seconds if fast_seconds else float("inf")
    flat_speedup = baseline_seconds / flat_seconds if flat_seconds else float("inf")

    print(f"rouge-score loop: {baseline_seconds:.4f}s")
    print(f"fast_rouge.score_batch: {fast_seconds:.4f}s")
    print(f"score_batch speedup: {dict_speedup:.2f}x")
    print(f"fast_rouge.score_batch_flat: {flat_seconds:.4f}s")
    print(f"score_batch_flat speedup: {flat_speedup:.2f}x")


if __name__ == "__main__":
    main()
