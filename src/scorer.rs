use rustc_hash::FxHashMap as HashMap;

/// Precision, recall, and F1 for a single metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Fraction of prediction units that match the reference.
    pub precision: f64,
    /// Fraction of reference units matched by the prediction.
    pub recall: f64,
    /// Harmonic mean of precision and recall (F1).
    pub fmeasure: f64,
}

impl Score {
    fn zero() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            fmeasure: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct ScoreSet {
    rouge1: Score,
    rouge2: Score,
    rouge_l: Score,
}

struct TokenizedText {
    normalized: String,
    spans: Vec<(usize, usize)>,
}

impl TokenizedText {
    fn len(&self) -> usize {
        self.spans.len()
    }

    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    fn token(&self, index: usize) -> &str {
        let (start, end) = self.spans[index];
        &self.normalized[start..end]
    }

    fn to_owned_tokens(&self) -> Vec<String> {
        self.spans
            .iter()
            .map(|&(start, end)| self.normalized[start..end].to_owned())
            .collect()
    }
}

/// Lowercase Unicode, then split on everything outside ASCII `a-z0-9`.
pub fn tokenize(text: &str) -> Vec<String> {
    tokenize_tokenized(text).to_owned_tokens()
}

/// Count overlapping n-grams without computing unrelated metrics.
/// Returns zero when `n == 0` or either input has fewer than `n` tokens.
pub fn rouge_n(reference: &str, prediction: &str, n: usize) -> Score {
    if n == 0 {
        return Score::zero();
    }
    let reference_tokens = tokenize_tokenized(reference);
    let prediction_tokens = tokenize_tokenized(prediction);
    rouge_n_tokenized(&reference_tokens, &prediction_tokens, n)
}

/// Score the longest common subsequence of two normalized inputs.
pub fn rouge_l(reference: &str, prediction: &str) -> Score {
    let reference_tokens = tokenize_tokenized(reference);
    let prediction_tokens = tokenize_tokenized(prediction);
    rouge_l_tokenized(&reference_tokens, &prediction_tokens)
}

/// Compute ROUGE-1, ROUGE-2, and ROUGE-L while tokenizing each input once.
pub fn score_all(reference: &str, prediction: &str) -> (Score, Score, Score) {
    let reference_tokens = tokenize_tokenized(reference);
    let prediction_tokens = tokenize_tokenized(prediction);
    let scores = score_set(&reference_tokens, &prediction_tokens);

    (scores.rouge1, scores.rouge2, scores.rouge_l)
}

/// Score pre-tokenized n-grams without further normalization.
/// A zero `n` or a sequence shorter than `n` yields zero scores.
pub fn rouge_n_tokens<T: AsRef<str>>(
    reference_tokens: &[T],
    prediction_tokens: &[T],
    n: usize,
) -> Score {
    if n == 0 {
        return Score::zero();
    }

    if n == 1 {
        return rouge1_tokens(reference_tokens, prediction_tokens);
    }

    if n == 2 {
        return rouge2_tokens(reference_tokens, prediction_tokens);
    }

    let reference_total = total_ngrams(reference_tokens.len(), n);
    let prediction_total = total_ngrams(prediction_tokens.len(), n);

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let reference_counts = ngram_counts_from_slice(reference_tokens, n);
    let prediction_counts = ngram_counts_from_slice(prediction_tokens, n);
    let overlap = overlap_count(&reference_counts, &prediction_counts);

    score_from_counts(overlap, reference_total, prediction_total)
}

/// Score a longest common subsequence without normalizing the supplied tokens.
pub fn rouge_l_tokens<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T]) -> Score {
    if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        return Score::zero();
    }

    let lcs = lcs_len(reference_tokens, prediction_tokens);
    score_from_counts(lcs, reference_tokens.len(), prediction_tokens.len())
}

/// Longest common subsequence length, with O(min(m, n)) memory and O(m * n) time.
pub fn lcs_len<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T]) -> usize {
    if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        return 0;
    }

    let (row_tokens, column_tokens) = if reference_tokens.len() >= prediction_tokens.len() {
        (reference_tokens, prediction_tokens)
    } else {
        (prediction_tokens, reference_tokens)
    };

    let width = column_tokens.len() + 1;
    let mut previous = vec![0; width];
    let mut current = vec![0; width];

    for reference in row_tokens {
        for (index, prediction) in column_tokens.iter().enumerate() {
            current[index + 1] = if reference.as_ref() == prediction.as_ref() {
                previous[index] + 1
            } else {
                previous[index + 1].max(current[index])
            };
        }

        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }

    previous[column_tokens.len()]
}

fn tokenize_tokenized(text: &str) -> TokenizedText {
    // Lowercase before filtering, as rouge-score does. Unicode characters such
    // as Kelvin sign (K) and dotted capital I (İ) can lowercase to ASCII.
    // ASCII text takes the single-pass path below.
    let lowered;
    let text = if text.is_ascii() {
        text
    } else {
        lowered = text.to_lowercase();
        &lowered
    };
    let mut normalized = String::with_capacity(text.len());
    let mut spans = Vec::with_capacity(estimated_token_capacity(text));
    let mut token_start = None;

    for &byte in text.as_bytes() {
        let output = if byte.is_ascii_alphanumeric() {
            byte.to_ascii_lowercase()
        } else {
            b' '
        };

        let index = normalized.len();
        normalized.push(char::from(output));

        if output == b' ' {
            if let Some(start) = token_start.take() {
                spans.push((start, index));
            }
        } else if token_start.is_none() {
            token_start = Some(index);
        }
    }

    if let Some(start) = token_start {
        spans.push((start, normalized.len()));
    }

    TokenizedText { normalized, spans }
}

fn estimated_token_capacity(text: &str) -> usize {
    (text.len() / 4).saturating_add(1)
}

fn score_set(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText) -> ScoreSet {
    ScoreSet {
        rouge1: rouge1_tokenized(reference_tokens, prediction_tokens),
        rouge2: rouge2_tokenized(reference_tokens, prediction_tokens),
        rouge_l: rouge_l_tokenized(reference_tokens, prediction_tokens),
    }
}

fn rouge_n_tokenized(
    reference_tokens: &TokenizedText,
    prediction_tokens: &TokenizedText,
    n: usize,
) -> Score {
    if n == 1 {
        return rouge1_tokenized(reference_tokens, prediction_tokens);
    }

    if n == 2 {
        return rouge2_tokenized(reference_tokens, prediction_tokens);
    }

    let reference_total = total_ngrams(reference_tokens.len(), n);
    let prediction_total = total_ngrams(prediction_tokens.len(), n);

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let reference_counts = ngram_counts_tokenized(reference_tokens, n);
    let prediction_counts = ngram_counts_tokenized(prediction_tokens, n);
    let overlap = overlap_count(&reference_counts, &prediction_counts);

    score_from_counts(overlap, reference_total, prediction_total)
}

fn rouge1_tokenized(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText) -> Score {
    let reference_total = reference_tokens.len();
    let prediction_total = prediction_tokens.len();

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let mut counts: HashMap<&str, usize> =
        HashMap::with_capacity_and_hasher(reference_total, Default::default());

    for index in 0..reference_tokens.len() {
        *counts.entry(reference_tokens.token(index)).or_insert(0) += 1;
    }

    let mut overlap = 0;

    for index in 0..prediction_tokens.len() {
        if let Some(count) = counts.get_mut(prediction_tokens.token(index))
            && *count > 0
        {
            *count -= 1;
            overlap += 1;
        }
    }

    score_from_counts(overlap, reference_total, prediction_total)
}

fn rouge2_tokenized(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText) -> Score {
    let reference_total = total_ngrams(reference_tokens.len(), 2);
    let prediction_total = total_ngrams(prediction_tokens.len(), 2);

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let mut counts: HashMap<(&str, &str), usize> =
        HashMap::with_capacity_and_hasher(reference_total, Default::default());

    for index in 0..(reference_tokens.len() - 1) {
        *counts
            .entry((
                reference_tokens.token(index),
                reference_tokens.token(index + 1),
            ))
            .or_insert(0) += 1;
    }

    let mut overlap = 0;

    for index in 0..(prediction_tokens.len() - 1) {
        let key = (
            prediction_tokens.token(index),
            prediction_tokens.token(index + 1),
        );

        if let Some(count) = counts.get_mut(&key)
            && *count > 0
        {
            *count -= 1;
            overlap += 1;
        }
    }

    score_from_counts(overlap, reference_total, prediction_total)
}

fn rouge1_tokens<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T]) -> Score {
    let reference_total = reference_tokens.len();
    let prediction_total = prediction_tokens.len();

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let mut counts: HashMap<&str, usize> =
        HashMap::with_capacity_and_hasher(reference_total, Default::default());

    for token in reference_tokens {
        *counts.entry(token.as_ref()).or_insert(0) += 1;
    }

    let mut overlap = 0;

    for token in prediction_tokens {
        if let Some(count) = counts.get_mut(token.as_ref())
            && *count > 0
        {
            *count -= 1;
            overlap += 1;
        }
    }

    score_from_counts(overlap, reference_total, prediction_total)
}

fn rouge2_tokens<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T]) -> Score {
    let reference_total = total_ngrams(reference_tokens.len(), 2);
    let prediction_total = total_ngrams(prediction_tokens.len(), 2);

    if reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let mut counts: HashMap<(&str, &str), usize> =
        HashMap::with_capacity_and_hasher(reference_total, Default::default());

    for window in reference_tokens.windows(2) {
        *counts
            .entry((window[0].as_ref(), window[1].as_ref()))
            .or_insert(0) += 1;
    }

    let mut overlap = 0;

    for window in prediction_tokens.windows(2) {
        let key = (window[0].as_ref(), window[1].as_ref());

        if let Some(count) = counts.get_mut(&key)
            && *count > 0
        {
            *count -= 1;
            overlap += 1;
        }
    }

    score_from_counts(overlap, reference_total, prediction_total)
}

fn rouge_l_tokenized(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText) -> Score {
    if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        return Score::zero();
    }

    let lcs = lcs_len_tokenized(reference_tokens, prediction_tokens);
    score_from_counts(lcs, reference_tokens.len(), prediction_tokens.len())
}

fn lcs_len_tokenized(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText) -> usize {
    if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        return 0;
    }

    let (rows, cols) = if reference_tokens.len() >= prediction_tokens.len() {
        (reference_tokens, prediction_tokens)
    } else {
        (prediction_tokens, reference_tokens)
    };

    let width = cols.len() + 1;
    let mut previous = vec![0; width];
    let mut current = vec![0; width];

    for row_index in 0..rows.len() {
        let row_token = rows.token(row_index);

        for col_index in 0..cols.len() {
            current[col_index + 1] = if row_token == cols.token(col_index) {
                previous[col_index] + 1
            } else {
                previous[col_index + 1].max(current[col_index])
            };
        }

        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }

    previous[cols.len()]
}

fn total_ngrams(token_count: usize, n: usize) -> usize {
    token_count
        .checked_sub(n)
        .map_or(0, |remaining| remaining + 1)
}

fn ngram_counts_tokenized(tokens: &TokenizedText, n: usize) -> HashMap<Vec<&str>, usize> {
    let total = total_ngrams(tokens.len(), n);
    let mut counts = HashMap::with_capacity_and_hasher(total, Default::default());
    for start in 0..total {
        let key = (start..start + n)
            .map(|index| tokens.token(index))
            .collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn ngram_counts_from_slice<T: AsRef<str>>(tokens: &[T], n: usize) -> HashMap<Vec<&str>, usize> {
    let mut counts =
        HashMap::with_capacity_and_hasher(total_ngrams(tokens.len(), n), Default::default());
    for window in tokens.windows(n) {
        let key = window.iter().map(AsRef::as_ref).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn overlap_count(
    reference_counts: &HashMap<Vec<&str>, usize>,
    prediction_counts: &HashMap<Vec<&str>, usize>,
) -> usize {
    reference_counts
        .iter()
        .map(|(ngram, reference_count)| {
            prediction_counts.get(ngram).map_or(0, |prediction_count| {
                (*reference_count).min(*prediction_count)
            })
        })
        .sum()
}

fn score_from_counts(overlap: usize, reference_total: usize, prediction_total: usize) -> Score {
    if overlap == 0 || reference_total == 0 || prediction_total == 0 {
        return Score::zero();
    }

    let precision = overlap as f64 / prediction_total as f64;
    let recall = overlap as f64 / reference_total as f64;
    let fmeasure = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Score {
        precision,
        recall,
        fmeasure,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Score, lcs_len, rouge_l, rouge_l_tokens, rouge_n, rouge_n_tokens, score_all, tokenize,
    };

    fn assert_score_close(actual: Score, expected: Score) {
        let epsilon = 1e-12;

        assert!((actual.precision - expected.precision).abs() < epsilon);
        assert!((actual.recall - expected.recall).abs() < epsilon);
        assert!((actual.fmeasure - expected.fmeasure).abs() < epsilon);
    }

    #[test]
    fn tokenize_lowercases_and_splits_non_alphanumeric() {
        assert_eq!(
            tokenize("The QUICK, brown-fox! 123"),
            vec!["the", "quick", "brown", "fox", "123"]
        );
    }

    #[test]
    fn tokenize_matches_ascii_only_reference_behavior() {
        assert_eq!(tokenize("naïve façade"), vec!["na", "ve", "fa", "ade"]);
    }

    #[test]
    fn rouge1_counts_repeated_unigrams() {
        let score = rouge_n("a a b", "a b b", 1);

        assert_score_close(
            score,
            Score {
                precision: 2.0 / 3.0,
                recall: 2.0 / 3.0,
                fmeasure: 2.0 / 3.0,
            },
        );
    }

    #[test]
    fn rouge2_counts_bigram_overlap() {
        let score = rouge_n("the cat sat on the mat", "the cat slept on the mat", 2);

        assert_score_close(
            score,
            Score {
                precision: 0.6,
                recall: 0.6,
                fmeasure: 0.6,
            },
        );
    }

    #[test]
    fn rouge_n_returns_zero_for_empty_ngrams() {
        let score = rouge_n("", "anything", 1);

        assert_eq!(score, Score::zero());
    }

    #[test]
    fn rouge_n_returns_zero_for_zero_n() {
        let score = rouge_n("the cat sat", "the cat sat", 0);

        assert_eq!(score, Score::zero());
    }

    #[test]
    fn lcs_len_finds_longest_common_subsequence() {
        let reference = tokenize("the cat was found under the bed");
        let prediction = tokenize("the cat was under the bed");

        assert_eq!(lcs_len(&reference, &prediction), 6);
    }

    #[test]
    fn rouge_l_scores_from_lcs_length() {
        let score = rouge_l("A B C D", "A X C");

        assert_score_close(
            score,
            Score {
                precision: 2.0 / 3.0,
                recall: 0.5,
                fmeasure: 4.0 / 7.0,
            },
        );
    }

    #[test]
    fn score_all_keeps_rouge_l_stable_across_calls() {
        let warmup_pairs = [
            ("eta beta alpha", "eta alpha"),
            ("gamma delta epsilon zeta", "delta epsilon"),
            ("theta eta zeta", "theta zeta"),
        ];

        for (reference, prediction) in warmup_pairs {
            let _ = score_all(reference, prediction);
        }

        let reference = "zeta alpha delta eta beta gamma eta beta epsilon beta theta eta";
        let prediction = "alpha theta theta theta delta gamma gamma";

        let first = score_all(reference, prediction).2;
        let second = score_all(reference, prediction).2;

        let expected = Score {
            precision: 3.0 / 7.0,
            recall: 3.0 / 12.0,
            fmeasure: 0.3157894736842105,
        };

        assert_score_close(first, expected);
        assert_score_close(second, expected);
    }

    #[test]
    fn unicode_lowercasing_precedes_ascii_filtering() {
        assert_eq!(
            tokenize("Kelvin İSTANBUL aİb"),
            vec!["kelvin", "i", "stanbul", "ai", "b"]
        );
        assert!(tokenize("你好 🌎 ＡＢＣ").is_empty());
        assert_eq!(tokenize("a\0b\tC"), vec!["a", "b", "c"]);
    }

    #[test]
    fn higher_order_ngrams_count_multiplicity_and_handle_extreme_n() {
        let reference = ["a", "a", "a", "a"];
        let prediction = ["a", "a", "a", "b", "b"];
        let expected = Score {
            precision: 1.0 / 3.0,
            recall: 0.5,
            fmeasure: 0.4,
        };
        assert_score_close(rouge_n_tokens(&reference, &prediction, 3), expected);
        assert_score_close(rouge_n("a a a a", "a a a b b", 3), expected);
        for n in [0, 6, usize::MAX] {
            assert_eq!(rouge_n_tokens(&reference, &prediction, n), Score::zero());
            assert_eq!(rouge_n("a a a a", "a a a b b", n), Score::zero());
        }
    }

    #[test]
    fn text_and_token_apis_agree() {
        for (reference, prediction) in [("", "a"), ("a b a", "b a"), ("İ K café", "i k caf")] {
            let a = tokenize(reference);
            let b = tokenize(prediction);
            for n in 0..=5 {
                assert_eq!(rouge_n(reference, prediction, n), rouge_n_tokens(&a, &b, n));
            }
            assert_eq!(rouge_l(reference, prediction), rouge_l_tokens(&a, &b));
            let scores = score_all(reference, prediction);
            assert_eq!(scores.0, rouge_n_tokens(&a, &b, 1));
            assert_eq!(scores.1, rouge_n_tokens(&a, &b, 2));
            assert_eq!(scores.2, rouge_l_tokens(&a, &b));
        }
    }

    #[test]
    fn lcs_matches_brute_force_for_all_short_binary_sequences() {
        // This oracle enumerates subsequences instead of repeating the DP algorithm.
        let mut sequences = Vec::new();
        for len in 0..=5 {
            for bits in 0..1usize << len {
                sequences.push(
                    (0..len)
                        .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" })
                        .collect::<Vec<_>>(),
                );
            }
        }
        for a in &sequences {
            for b in &sequences {
                let mut expected = 0;
                for mask in 0..1usize << a.len() {
                    let subsequence: Vec<_> = a
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, token)| token)
                        .collect();
                    let mut rest = b.iter();
                    if subsequence
                        .iter()
                        .all(|token| rest.any(|other| other == *token))
                    {
                        expected = expected.max(subsequence.len());
                    }
                }
                assert_eq!(lcs_len(a, b), expected, "{a:?} vs {b:?}");
                let score = score_all(&a.join(" "), &b.join(" ")).2;
                let expected_precision = if b.is_empty() {
                    0.0
                } else {
                    expected as f64 / b.len() as f64
                };
                assert_eq!(score.precision, expected_precision);
            }
        }
    }
}
