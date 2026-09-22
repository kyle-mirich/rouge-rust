"""Differential tests against Google's unstemmed, default tokenizer."""

from random import Random

import pytest
from rouge_score import rouge_scorer

import fast_rouge

METRICS = ("rouge1", "rouge2", "rougeL")
FIELDS = ("precision", "recall", "fmeasure")
REFERENCE = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=False)
CASES = [
    ("", ""),
    ("", "non empty prediction"),
    ("non empty reference", ""),
    ("!!!", "???"),
    ("a", "a"),
    ("a a a b", "a b b b b"),
    ("a b c d", "d c b a"),
    ("a b c d e f", "b d"),
    ("The QUICK, brown-fox! 123", "the quick brown fox 123"),
    ("a\tb\nc\r\nd\u00a0e\x00f", "a b c d e f"),
    ("naïve façade CAFÉ", "na ve fa ade caf"),
    ("Kelvin", "kelvin"),
    ("İSTANBUL", "i stanbul"),
    ("aİb K x", "ai b k x"),
    ("你好世界 🌎", "你好世界 🌎"),
    ("ＡＢＣ １２３", "abc 123"),
    ("e\u0301 café", "e caf"),
    ("running cats", "run cat"),
    ("a " * 130 + "b", "a " * 99 + "c"),
]


def assert_scores(actual, expected):
    assert set(actual) == set(METRICS)
    for metric in METRICS:
        for field in FIELDS:
            assert getattr(actual[metric], field) == getattr(expected[metric], field)


@pytest.mark.parametrize("reference,prediction", CASES)
def test_edge_cases_match_reference(reference, prediction):
    assert_scores(fast_rouge.score(reference, prediction), REFERENCE.score(reference, prediction))


def test_all_apis_match_reference_on_seeded_random_inputs():
    rng = Random(42)
    vocabulary = ["a", "a", "b", "c", "123", "HELLO", "İ", "K", "naïve", "你好", "🌎"]
    separators = [" ", "\n", "-", "_", "\x00", "\t"]

    def sentence():
        return rng.choice(separators).join(rng.choices(vocabulary, k=rng.randrange(36)))

    cases = CASES + [(sentence(), sentence()) for _ in range(400)]
    references, predictions = map(list, zip(*cases))
    batch = fast_rouge.score_batch(references, predictions)
    flat = fast_rouge.score_batch_flat(references, predictions)
    # Get each column once: PyO3's Vec getters return copies, not live views.
    columns = {
        f"{metric}_{field}": getattr(flat, f"{metric}_{field}")
        for metric in METRICS
        for field in FIELDS
    }
    assert len(batch) == len(cases)
    assert all(len(column) == len(cases) for column in columns.values())
    for index, (reference, prediction) in enumerate(cases):
        expected = REFERENCE.score(reference, prediction)
        assert_scores(fast_rouge.score(reference, prediction), expected)
        assert_scores(batch[index], expected)
        for metric in METRICS:
            for field in FIELDS:
                assert columns[f"{metric}_{field}"][index] == getattr(expected[metric], field)
