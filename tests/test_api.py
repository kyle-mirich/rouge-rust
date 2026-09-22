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


def test_empty_batches_return_empty_results():
    assert fast_rouge.score_batch([], []) == []

    result = fast_rouge.score_batch_flat([], [])
    assert result.rouge1_precision == []
    assert result.rouge2_recall == []
    assert result.rougeL_fmeasure == []


@pytest.mark.parametrize("function", [fast_rouge.score_batch, fast_rouge.score_batch_flat])
@pytest.mark.parametrize("references,predictions", [([], ["a"]), (["a", "b"], ["a"])])
def test_batch_length_mismatch_in_both_directions(function, references, predictions):
    with pytest.raises(ValueError, match="same length"):
        function(references, predictions)


@pytest.mark.parametrize("invalid", [None, 42, b"text", ["text"]])
def test_single_rejects_non_strings(invalid):
    with pytest.raises(TypeError):
        fast_rouge.score(invalid, "text")
    with pytest.raises(TypeError):
        fast_rouge.score("text", invalid)


@pytest.mark.parametrize("function", [fast_rouge.score_batch, fast_rouge.score_batch_flat])
@pytest.mark.parametrize("invalid", [None, 42, b"text"])
def test_batches_reject_non_string_elements(function, invalid):
    with pytest.raises(TypeError):
        function([invalid], ["text"])
    with pytest.raises(TypeError):
        function(["text"], [invalid])


@pytest.mark.parametrize("function", [fast_rouge.score_batch, fast_rouge.score_batch_flat])
def test_batches_reject_bare_strings(function):
    with pytest.raises(TypeError):
        function("abc", "abc")


def test_keyword_arguments_and_tuple_inputs():
    result = fast_rouge.score(reference="one two", prediction="one")
    assert result["rouge1"].precision == 1.0
    assert result["rouge1"].recall == 0.5
    batch = fast_rouge.score_batch(references=("one",), predictions=("one",))
    flat = fast_rouge.score_batch_flat(references=("one",), predictions=("one",))
    assert batch[0]["rouge1"].fmeasure == flat.rouge1_fmeasure[0] == 1.0


def test_result_properties_are_read_only_and_columns_are_copies():
    score = fast_rouge.score("a", "a")["rouge1"]
    with pytest.raises(AttributeError):
        score.precision = 0.0
    flat = fast_rouge.score_batch_flat(["a"], ["a"])
    with pytest.raises(AttributeError):
        flat.rouge1_precision = []
    column = flat.rouge1_precision
    column[0] = 0.0
    assert flat.rouge1_precision == [1.0]


def test_version_matches_distribution_metadata():
    from importlib.metadata import version

    assert fast_rouge.__version__ == version("rouge-rust")


def test_concurrent_calls_finish_and_preserve_results():
    # Run in a subprocess so a GIL/Rayon deadlock fails with a bounded timeout.
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        from concurrent.futures import ThreadPoolExecutor
        import fast_rouge

        def work(index):
            reference = "a b c " * 30
            prediction = "a x c " * 30
            expected = fast_rouge.score(reference, prediction)["rougeL"].fmeasure
            batch = fast_rouge.score_batch([reference] * 12, [prediction] * 12)
            flat = fast_rouge.score_batch_flat([reference] * 12, [prediction] * 12)
            assert all(s["rougeL"].fmeasure == expected for s in batch)
            assert flat.rougeL_fmeasure == [expected] * 12

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work, range(16)))
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
