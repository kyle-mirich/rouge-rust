"""Independent count examples and exhaustive installed-API reference checks."""

from itertools import product

import pytest
from rouge_score import rouge_scorer

import fast_rouge

METRICS = ("rouge1", "rouge2", "rougeL")
FIELDS = ("precision", "recall", "fmeasure")


@pytest.mark.parametrize(
    "reference,prediction,counts",
    [
        ("a a a", "a a", [(2, 3, 2), (1, 2, 1), (2, 3, 2)]),
        ("a b a b", "a b", [(2, 4, 2), (1, 3, 1), (2, 4, 2)]),
        ("a b c d", "d c b a", [(4, 4, 4), (0, 3, 3), (1, 4, 4)]),
        ("a b c d e", "a c e", [(3, 5, 3), (0, 4, 2), (3, 5, 3)]),
        ("a\nb", "b\na", [(2, 2, 2), (0, 1, 1), (1, 2, 2)]),
        ("café", "cafe\u0301", [(0, 1, 1), (0, 0, 0), (0, 1, 1)]),
        ("running cats", "run cat", [(0, 2, 2), (0, 1, 1), (0, 2, 2)]),
        ("a", "a", [(1, 1, 1), (0, 0, 0), (1, 1, 1)]),
        ("你好 !!!", "你好 !!!", [(0, 0, 0)] * 3),
    ],
)
def test_scores_follow_hand_counted_units(reference, prediction, counts):
    single = fast_rouge.score(reference, prediction)
    batch = fast_rouge.score_batch([reference], [prediction])[0]
    flat = fast_rouge.score_batch_flat([reference], [prediction])
    for metric, (overlap, reference_total, prediction_total) in zip(METRICS, counts):
        # Compute F1 from the unit counts, independently of the scorer's formula.
        expected = (
            overlap / prediction_total if prediction_total else 0.0,
            overlap / reference_total if reference_total else 0.0,
            2 * overlap / (reference_total + prediction_total)
            if reference_total + prediction_total
            else 0.0,
        )
        for field, value in zip(FIELDS, expected):
            assert getattr(single[metric], field) == pytest.approx(value)
            assert getattr(batch[metric], field) == pytest.approx(value)
            assert getattr(flat, f"{metric}_{field}")[0] == pytest.approx(value)


def test_all_short_binary_text_pairs_match_reference_in_all_apis():
    # 31 sequences, 961 ordered pairs: covers asymmetry, ties and multiplicity.
    texts = [" ".join(tokens) for length in range(5) for tokens in product("ab", repeat=length)]
    pairs = list(product(texts, repeat=2))
    references, predictions = map(list, zip(*pairs))
    reference_scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=False)
    batch = fast_rouge.score_batch(references, predictions)
    flat = fast_rouge.score_batch_flat(references, predictions)
    columns = {
        (metric, field): getattr(flat, f"{metric}_{field}")
        for metric in METRICS
        for field in FIELDS
    }
    assert len(batch) == len(pairs) == 961
    assert all(len(column) == len(pairs) for column in columns.values())
    for index, (reference, prediction) in enumerate(pairs):
        expected = reference_scorer.score(reference, prediction)
        single = fast_rouge.score(reference, prediction)
        reversed_score = fast_rouge.score(prediction, reference)
        for metric in METRICS:
            for field in FIELDS:
                value = getattr(expected[metric], field)
                assert getattr(single[metric], field) == value
                assert getattr(batch[index][metric], field) == value
                assert columns[metric, field][index] == value
            assert single[metric].precision == reversed_score[metric].recall
            assert single[metric].fmeasure == reversed_score[metric].fmeasure


def test_rouge_l_is_distinct_from_summary_level_union_lcs():
    reference, prediction = "a\nb", "b\na"
    google = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=False)
    expected = google.score(reference, prediction)
    assert expected["rougeL"].fmeasure == 0.5
    assert expected["rougeLsum"].fmeasure == 1.0
    assert fast_rouge.score(reference, prediction)["rougeL"].fmeasure == 0.5
