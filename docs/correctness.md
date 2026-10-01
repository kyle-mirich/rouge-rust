# Scoring contract and reference evidence

ROUGE measures overlap with a reference text. The original
[ROUGE paper (Lin, 2004)](https://aclanthology.org/W04-1013/) describes a family of
summary-evaluation metrics. This implementation targets a specific subset of
[Google's Python implementation](https://github.com/google-research/google-research/tree/master/rouge):
`rouge-score==0.1.2`, metrics `rouge1`, `rouge2`, `rougeL`, default tokenizer,
`use_stemmer=False`, and one reference per prediction. The argument order is
`score(reference, prediction)`.

| Metric | Matching unit | Behavior |
| --- | --- | --- |
| ROUGE-1 | Individual normalized tokens | Count clipped overlap, including repetitions |
| ROUGE-2 | Adjacent overlapping token pairs | Count clipped overlap, including repetitions |
| ROUGE-L | Longest common token subsequence | Preserve order; gaps are allowed |

For overlap O, reference unit count R, and prediction unit count P, precision is
O/P, recall is O/R, and F1 is their harmonic mean. Zero overlap or either zero
unit count produces three zeros. Two identical single-token inputs have ROUGE-2
zero because neither has a bigram. Repeating prediction tokens cannot match more
occurrences than exist in the reference.

## Normalization matters

Google's [tokenizer source](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py)
first lowercases, replaces everything outside ASCII `a-z0-9` with separators,
and drops empty tokens. The Rust implementation follows that order.

| Input | Tokens |
| --- | --- |
| `The QUICK, brown-fox! 123` | `the`, `quick`, `brown`, `fox`, `123` |
| `Kelvin İSTANBUL` | `kelvin`, `i`, `stanbul` |
| `café` (composed accent) | `caf` |
| `café` (combining accent) | `cafe` |
| `你好 !!!` | none |

There is no Unicode normalization, accent folding, case folding, or stemming.
`running cats` and `run cat` therefore have zero overlap. Punctuation, embedded
NULs, tabs, and newlines act as token separators. The tokenizer is unsuitable for
general multilingual evaluation. Rust and Python provide their own Unicode
lowercasing tables; parity checks describe tested runtimes and inputs, not an
assurance about every future Unicode/runtime revision.

## ROUGE-L and ROUGE-Lsum differ

The supported ROUGE-L computes one LCS across all tokens. It does not implement
Google's sentence-level union-LCS `rougeLsum`. For reference `"a\nb"` and
prediction `"b\na"`, unstemmed Google ROUGE-L F1 is 0.5 while ROUGE-Lsum F1 is
1.0 with newline sentence splitting. A regression test records this distinction.

Stemming, alternate tokenizers, weighted/skip-bigram variants, multi-reference
selection, bootstrap intervals, and corpus aggregation are outside this API.
Matching three scores does not establish equivalence with the original Perl
package or other implementations' preprocessing and aggregation choices.

## Evidence

- **Independent known answers:** nine hand-counted examples check all three APIs
  for clipped multiplicity, asymmetric denominators, reordered tokens, a gapped
  subsequence, absent bigrams, accent composition, no stemming, and empty tokens.
- **Exhaustive small inputs:** all 961 ordered pairs of binary token sequences
  of length zero through four match Google's nine fields exactly in `score`,
  `score_batch`, and `score_batch_flat`. Swapping inputs also checks precision/
  recall reversal and F1 symmetry.
- **Broader differential coverage:** 19 curated edge cases and 400 seeded random
  pairs cover Unicode boundaries, repeated tokens, different lengths, separators,
  and empty inputs across all entrypoints.
- **Independent Rust oracle:** brute-force subsequence enumeration checks all
  3,969 pairs of binary sequences up to length five, independently of the LCS
  dynamic program. Other unit tests cover arbitrary n-grams and text/token APIs.
- **Boundary checks:** installed-extension tests cover argument types, batch
  ordering, read-only results, copied columns, and concurrent Python callers.

Run both `cargo test --locked` and `python -m pytest -q` after installing the
checkout. The Python suite must use the rebuilt extension; Rust tests alone do
not exercise PyO3 or the production tokenized path. See [CONTRIBUTING](../CONTRIBUTING.md)
for the full verification commands and [benchmarking](benchmarking.md) for
validated timing comparisons.
