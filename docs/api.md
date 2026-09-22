# Python API

Install `rouge-rust`, then `import fast_rouge`. `fast_rouge.__version__` reports
the installed package version. The wheel includes PEP 561 typing information.

## Functions

```python
score(reference: str, prediction: str) -> dict[str, Score]
score_batch(references: Sequence[str], predictions: Sequence[str]) -> list[dict[str, Score]]
score_batch_flat(references: Sequence[str], predictions: Sequence[str]) -> BatchScoreResult
```

These signatures use modern Python notation; the installed stubs also support
Python 3.8. Arguments may be passed by keyword. Lists and tuples work for batch
inputs. Use sequences, not generators or bare strings.

- `reference` is the ground-truth text; `prediction` is the generated text.
- All functions compute `rouge1`, `rouge2`, and `rougeL` without stemming.
- Batch results preserve input order, including duplicate pairs.
- Batch inputs of unequal length raise `ValueError`.
- Non-string inputs or batch elements raise `TypeError`. Strings containing
  lone surrogate code points cannot be encoded as UTF-8 and raise `UnicodeEncodeError`.
- Empty batches return an empty list or a result with nine empty columns.
- Rust scoring releases the GIL; Python argument/result conversion still occurs
  while attached to the interpreter.

## Score

`score()` and each item in `score_batch()` return a dictionary with exactly
three keys: `rouge1`, `rouge2`, and `rougeL`. Each value is a `fast_rouge.Score`
with three read-only float properties:

| Field | Meaning |
| --- | --- |
| `precision` | Overlap divided by prediction n-grams or tokens |
| `recall` | Overlap divided by reference n-grams or tokens |
| `fmeasure` | F1: `2 * precision * recall / (precision + recall)` |

N-gram overlap uses counts, so repeating one token cannot match more occurrences
than the reference contains. ROUGE-L overlap is the longest common **subsequence**
of tokens; matching tokens need not be adjacent. Newlines act as token separators,
not sentence boundaries for a summary-level metric.

A zero denominator or zero overlap produces all-zero scores. Identical single-token
inputs therefore score 1 for ROUGE-1 and ROUGE-L, but 0 for ROUGE-2.

These extension objects are returned by scoring functions and cannot be directly
constructed. They expose attributes, not namedtuple indexing or `_asdict()`.
For JSON serialization, convert the properties explicitly:

```python
import json
import fast_rouge

scores = fast_rouge.score("one two", "one")
plain = {
    metric: {field: getattr(value, field) for field in ("precision", "recall", "fmeasure")}
    for metric, value in scores.items()
}
print(json.dumps(plain))
```

## BatchScoreResult

`score_batch_flat()` returns an object with the following read-only properties:

| Metric | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ROUGE-1 | `rouge1_precision` | `rouge1_recall` | `rouge1_fmeasure` |
| ROUGE-2 | `rouge2_precision` | `rouge2_recall` | `rouge2_fmeasure` |
| ROUGE-L | `rougeL_precision` | `rougeL_recall` | `rougeL_fmeasure` |

Each property returns `list[float]` with one value per input pair. These are copies
of Rust vectors, not NumPy arrays or zero-copy views. Mutating a returned list
never changes the stored result. Read a column once to avoid repeatedly copying it.

Optional pandas integration (install pandas separately):

```python
import pandas as pd
import fast_rouge

flat = fast_rouge.score_batch_flat(["a b", "c d"], ["a", "c"])
frame = pd.DataFrame(
    {
        f"{metric}_{field}": getattr(flat, f"{metric}_{field}")
        for metric in ("rouge1", "rouge2", "rougeL")
        for field in ("precision", "recall", "fmeasure")
    }
)
```

## Bound memory with chunks

The batch functions copy inputs into Rust and retain all outputs until the call
returns. For large datasets, process chunks and retain only the metrics you need:

```python
import fast_rouge

references = ["a b", "c d", "e f"]
predictions = ["a", "c", "e"]
if len(references) != len(predictions):
    raise ValueError("references and predictions must have the same length")

chunk_size = 1024
total_f1 = 0.0
count = 0
for start in range(0, len(references), chunk_size):
    result = fast_rouge.score_batch_flat(
        references[start : start + chunk_size],
        predictions[start : start + chunk_size],
    )
    column = result.rouge1_fmeasure
    total_f1 += sum(column)
    count += len(column)
mean_f1 = total_f1 / count if count else 0.0
```

This computes an arithmetic mean of per-pair F1 values, not a corpus-level ROUGE
statistic or confidence interval. Rayon uses a process-wide shared pool; multiple
Python caller threads share it. Set `RAYON_NUM_THREADS=4` before launching Python
when you need to limit scoring workers.
