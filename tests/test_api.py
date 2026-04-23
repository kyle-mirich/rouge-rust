import pytest

import fast_rouge


def test_score_returns_expected_metrics():
    scores = fast_rouge.score("the cat sat", "the cat sat")

    assert scores["rouge1"].precision == pytest.approx(1.0)
    assert scores["rouge2"].recall == pytest.approx(1.0)
    assert scores["rougeL"].fmeasure == pytest.approx(1.0)


def test_public_api_exposes_intended_entrypoints():
    assert not hasattr(fast_rouge, "dummy_score")
    assert hasattr(fast_rouge, "score")
    assert hasattr(fast_rouge, "score_batch")
    assert hasattr(fast_rouge, "score_batch_flat")


def test_batch_apis_match_single_score():
    references = ["the cat sat", "hello world"]
    predictions = ["the cat sat", "hello there"]

    batch_scores = fast_rouge.score_batch(references, predictions)
    flat_scores = fast_rouge.score_batch_flat(references, predictions)

    assert len(batch_scores) == 2
    assert batch_scores[0]["rouge1"].fmeasure == pytest.approx(1.0)
    assert batch_scores[1]["rouge2"].precision == pytest.approx(0.0)
    assert flat_scores.rouge1_fmeasure == pytest.approx([1.0, 0.5])
    assert flat_scores.rougeL_precision == pytest.approx([1.0, 0.5])


def test_batch_apis_validate_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        fast_rouge.score_batch(["a"], [])

    with pytest.raises(ValueError, match="same length"):
        fast_rouge.score_batch_flat(["a"], [])
