"""Exercise the runnable demo and benchmark through their actual CLI boundaries."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "score_jsonl.py"


def run_demo(text, *args):
    return subprocess.run(
        [sys.executable, str(DEMO), *args],
        input=text,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_demo_chunks_keep_order_ids_and_empty_input():
    records = [
        {"id": "first", "reference": "a b c", "prediction": "a"},
        {"id": 42, "reference": "", "prediction": ""},
        {"reference": "a b", "prediction": "a b"},
    ]
    text = "\n" + "\n".join(json.dumps(record) for record in records)
    result = run_demo(text, "--batch-size", "2")
    assert result.returncode == 0, result.stderr
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["line"] for row in output] == [2, 3, 4]
    assert [row.get("id") for row in output] == ["first", 42, None]
    assert output[0]["scores"]["rouge1"] == {
        "precision": 1.0,
        "recall": 1 / 3,
        "fmeasure": 0.5,
    }
    assert output[1]["scores"]["rougeL"]["fmeasure"] == 0.0
    assert output[2]["scores"]["rouge2"]["fmeasure"] == 1.0
    assert run_demo("").stdout == ""


def test_demo_file_matches_stdin():
    example = ROOT / "examples" / "pairs.jsonl"
    from_file = run_demo("", str(example))
    from_stdin = run_demo(example.read_text(encoding="utf-8"))
    assert from_file.returncode == from_stdin.returncode == 0
    assert from_file.stdout == from_stdin.stdout


@pytest.mark.parametrize("text", ["{bad}", "[]", "{}", '{"reference":1,"prediction":"a"}'])
def test_demo_errors_identify_line(text):
    result = run_demo("\n" + text)
    assert result.returncode == 2
    assert "line 2" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_demo_rejects_zero_batch_size():
    result = run_demo("", "--batch-size", "0")
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_benchmark_report_contains_reproducible_inputs_and_raw_samples(tmp_path):
    report = tmp_path / "benchmark.json"
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmark.py"),
            "--pairs",
            "7",
            "--repeats",
            "2",
            "--seed",
            "123",
            "--json-output",
            str(report),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["validation"]["exact_match"] is True
    assert data["reference_settings"] == {
        "metrics": ["rouge1", "rouge2", "rougeL"],
        "use_stemmer": False,
        "tokenizer": "default",
    }
    assert data["environment"]["rayon_num_threads"] == "1"
    assert data["workload"]["pairs"] == 7
    assert data["workload"]["seed"] == 123
    assert len(data["workload"]["input_sha256"]) == hashlib.sha256().digest_size * 2
    assert len(data["timings"]) == 5
    assert data["timing_order"][0] != data["timing_order"][1]
    for path in data["timings"].values():
        assert len(path["seconds"]) == 2
        assert all(value > 0 for value in path["seconds"])
        assert path["pairs_per_second"] == 7 / path["median_seconds"]


@pytest.mark.parametrize("args", [["--pairs", "0"], ["--min-tokens", "8", "--max-tokens", "2"]])
def test_benchmark_rejects_invalid_workloads(args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), *args],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
