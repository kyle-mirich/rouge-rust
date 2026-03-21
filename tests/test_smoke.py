import fast_rouge


def test_dummy_score_smoke() -> None:
    assert fast_rouge.dummy_score() == "fast_rouge-ready"
