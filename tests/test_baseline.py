from rouge_score import rouge_scorer

import fast_rouge


CASES = [
    ("", ""),
    ("", "non empty prediction"),
    ("the cat sat", "the cat sat"),
    ("the cat sat", "the dog sat"),
    ("The CAT sat", "the cat sat"),
    ("The QUICK, brown-fox! 123", "the quick brown fox 123"),
]

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def _score_tuple(score):
    return (score.precision, score.recall, score.fmeasure)


def _flat_score_tuple(batch_result, index, prefix):
    return (
        getattr(batch_result, f"{prefix}_precision")[index],
        getattr(batch_result, f"{prefix}_recall")[index],
        getattr(batch_result, f"{prefix}_fmeasure")[index],
    )


def test_score_matches_rouge_score_reference() -> None:
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=False)

    for reference, prediction in CASES:
        expected = scorer.score(reference, prediction)
        actual = fast_rouge.score(reference, prediction)

        assert set(actual) == set(ROUGE_TYPES)

        for rouge_type in ROUGE_TYPES:
            assert _score_tuple(actual[rouge_type]) == _score_tuple(expected[rouge_type])


def test_score_batch_matches_iterative_score() -> None:
    references = [reference for reference, _ in CASES]
    predictions = [prediction for _, prediction in CASES]

    batch_scores = fast_rouge.score_batch(references, predictions)
    iterative_scores = [
        fast_rouge.score(reference, prediction)
        for reference, prediction in CASES
    ]

    assert len(batch_scores) == len(iterative_scores)

    for batch_score, iterative_score in zip(batch_scores, iterative_scores):
        assert set(batch_score) == set(ROUGE_TYPES)

        for rouge_type in ROUGE_TYPES:
            assert _score_tuple(batch_score[rouge_type]) == _score_tuple(
                iterative_score[rouge_type]
            )


def test_score_batch_flat_matches_iterative_score() -> None:
    references = [reference for reference, _ in CASES]
    predictions = [prediction for _, prediction in CASES]

    batch_scores = fast_rouge.score_batch_flat(references, predictions)
    iterative_scores = [
        fast_rouge.score(reference, prediction)
        for reference, prediction in CASES
    ]

    assert len(batch_scores.rouge1_precision) == len(iterative_scores)
    assert len(batch_scores.rouge2_precision) == len(iterative_scores)
    assert len(batch_scores.rougeL_precision) == len(iterative_scores)

    for index, iterative_score in enumerate(iterative_scores):
        assert _flat_score_tuple(batch_scores, index, "rouge1") == _score_tuple(
            iterative_score["rouge1"]
        )
        assert _flat_score_tuple(batch_scores, index, "rouge2") == _score_tuple(
            iterative_score["rouge2"]
        )
        assert _flat_score_tuple(batch_scores, index, "rougeL") == _score_tuple(
            iterative_score["rougeL"]
        )
