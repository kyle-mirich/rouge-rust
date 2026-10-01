# Runnable evaluation demo

This repository example scores a UTF-8 JSONL file with the installed `fast_rouge`
extension. It needs no service, credentials, model calls, or dataset download.
The example script is included in the source distribution; the wheel exposes the
Python API and does not install a CLI command.

From a checkout after `python -m pip install rouge-rust`:

```bash
python examples/score_jsonl.py examples/pairs.jsonl --batch-size 2
# Or supply your own records on stdin:
printf '%s\n' '{"reference":"a b c","prediction":"a c"}' | python examples/score_jsonl.py
```

Each nonblank input line must be a JSON object with string `reference` and
`prediction` fields. Optional `id` values pass through to the output. Output is
one JSON object per pair, in input order, with its original one-based line number
and all nine precision/recall/F1 values. Blank lines are skipped.

The bundled four-pair example produces these F1 values:

| ID | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| `complete` | 1.0 | 1.0 | 1.0 |
| `omission` | 0.666667 | 0.571429 | 0.666667 |
| `reordered` | 1.0 | 0.0 | 0.25 |
| `empty` | 0.0 | 0.0 | 0.0 |

Values in this table are rounded; the JSON output retains full float precision.
The reordered example illustrates why overlap, adjacency, and sequence order are
separate signals. These metrics measure lexical overlap, not factual accuracy or
semantic equivalence.

## Stream larger files

```bash
RAYON_NUM_THREADS=4 python examples/score_jsonl.py pairs.jsonl --batch-size 256 > scores.jsonl
```

The script retains at most one batch of records and materializes each metric
column once per batch. Batch size bounds the number of pairs, not text length;
ROUGE-L still costs O(m × n) per pair. Input must be valid UTF-8 and accepted by
the extension's [normalization rules](correctness.md).

Malformed JSON, missing/non-string fields, invalid batch sizes, and unreadable
files exit with status 2 and a message on stderr. Parsing errors include the line
number. Streaming can emit earlier batches before discovering a later bad record,
so a nonzero exit means the output may be partial. A batch with an invalid record
is not emitted. Use the exit status when consuming output in a pipeline.

The example is covered by subprocess tests for stdin/files, chunk boundaries,
ordering, IDs, empty data, and invalid input. It returns per-pair scores; it does
not compute corpus statistics or confidence intervals.
