"""Score JSONL reference/prediction pairs in bounded batches; see docs/demo.md."""

import argparse
import json
import sys
from contextlib import nullcontext
from itertools import islice

import fast_rouge

METRICS = ("rouge1", "rouge2", "rougeL")
FIELDS = ("precision", "recall", "fmeasure")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def read_records(handle):
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) for field in ("reference", "prediction")
        ):
            raise ValueError(f"line {line_number}: reference and prediction must be strings")
        yield line_number, record


def score_records(handle, output, batch_size):
    records = read_records(handle)
    while True:
        chunk = list(islice(records, batch_size))
        if not chunk:
            return
        flat = fast_rouge.score_batch_flat(
            [record["reference"] for _, record in chunk],
            [record["prediction"] for _, record in chunk],
        )
        columns = {
            (metric, field): getattr(flat, f"{metric}_{field}")
            for metric in METRICS
            for field in FIELDS
        }
        for index, (line_number, record) in enumerate(chunk):
            result = {
                "line": line_number,
                "scores": {
                    metric: {field: columns[metric, field][index] for field in FIELDS}
                    for metric in METRICS
                },
            }
            if "id" in record:
                result["id"] = record["id"]
            print(json.dumps(result, allow_nan=False), file=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="UTF-8 JSONL file, or - for stdin")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    args = parser.parse_args()
    try:
        source = nullcontext(sys.stdin) if args.input == "-" else open(args.input, encoding="utf-8")
        with source as handle:
            score_records(handle, sys.stdout, args.batch_size)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
