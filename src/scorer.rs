use std::cell::RefCell;
use std::rc::Rc;

use rustc_hash::FxHashMap as HashMap;

thread_local! {
    static TOKEN_CACHE: RefCell<Vec<CacheEntry>> = const { RefCell::new(Vec::new()) };
    static SCORE_CACHE: RefCell<Option<PairScoreCache>> = const { RefCell::new(None) };
    static LCS_ROWS: RefCell<(Vec<usize>, Vec<usize>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

const TOKEN_CACHE_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub precision: f64,
    pub recall: f64,
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

#[derive(Debug, Hash, Eq, PartialEq)]
enum NgramKey<'a> {
    Unigram(&'a str),
    Bigram(&'a str, &'a str),
    Higher(Vec<&'a str>),
}

#[derive(Clone)]
struct CacheEntry {
    text: String,
    tokenized: Rc<TokenizedText>,
}

#[derive(Clone, Copy)]
struct ScoreSet {
    rouge1: Score,
    rouge2: Score,
    rouge_l: Score,
}

struct PairScoreCache {
    reference: String,
    prediction: String,
    scores: ScoreSet,
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

pub fn tokenize(text: &str) -> Vec<String> {
    tokenize_tokenized(text).to_owned_tokens()
}

pub fn rouge_n(reference: &str, prediction: &str, n: usize) -> Score {
    if let Some(scores) = cached_scores(reference, prediction) {
        return match n {
            1 => scores.rouge1,
            2 => scores.rouge2,
            _ => {
                let reference_tokens = tokenize_cached(reference);
                let prediction_tokens = tokenize_cached(prediction);
                rouge_n_tokenized(&reference_tokens, &prediction_tokens, n)
            }
        };
    }

    let reference_tokens = tokenize_cached(reference);
    let prediction_tokens = tokenize_cached(prediction);

    if n == 1 || n == 2 {
        let scores = score_set(&reference_tokens, &prediction_tokens);
        store_scores(reference, prediction, scores);

        return if n == 1 { scores.rouge1 } else { scores.rouge2 };
    }

    rouge_n_tokenized(&reference_tokens, &prediction_tokens, n)
}

pub fn rouge_l(reference: &str, prediction: &str) -> Score {
    if let Some(scores) = cached_scores(reference, prediction) {
        return scores.rouge_l;
    }

    let reference_tokens = tokenize_cached(reference);
    let prediction_tokens = tokenize_cached(prediction);
    let scores = score_set(&reference_tokens, &prediction_tokens);
    store_scores(reference, prediction, scores);

    scores.rouge_l
}

pub fn rouge_n_tokens<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T], n: usize) -> Score {
    if n == 0 {
        return Score::zero();
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

pub fn rouge_l_tokens<T: AsRef<str>>(reference_tokens: &[T], prediction_tokens: &[T]) -> Score {
    if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        return Score::zero();
    }

    let lcs = lcs_len(reference_tokens, prediction_tokens);
    score_from_counts(lcs, reference_tokens.len(), prediction_tokens.len())
}

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

    LCS_ROWS.with(|rows| {
        let mut rows = rows.borrow_mut();
        ensure_zeroed_len(&mut rows.0, width);
        ensure_zeroed_len(&mut rows.1, width);

        let (previous, current) = &mut *rows;

        for reference in row_tokens {
            current[0] = 0;

            for (index, prediction) in column_tokens.iter().enumerate() {
                current[index + 1] = if reference.as_ref() == prediction.as_ref() {
                    previous[index] + 1
                } else {
                    previous[index + 1].max(current[index])
                };
            }

            std::mem::swap(previous, current);
        }

        previous[column_tokens.len()]
    })
}

fn tokenize_cached(text: &str) -> Rc<TokenizedText> {
    TOKEN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if let Some(entry) = cache.iter().find(|entry| entry.text == text) {
            return Rc::clone(&entry.tokenized);
        }

        let tokenized = Rc::new(tokenize_tokenized(text));

        if cache.len() == TOKEN_CACHE_SIZE {
            cache.remove(0);
        }

        cache.push(CacheEntry {
            text: text.to_owned(),
            tokenized: Rc::clone(&tokenized),
        });

        tokenized
    })
}

fn tokenize_tokenized(text: &str) -> TokenizedText {
    let mut normalized = String::with_capacity(text.len());
    let mut spans = Vec::with_capacity(estimated_token_capacity(text));
    let mut token_start = None;

    for character in text.chars() {
        let output = if character.is_ascii_alphanumeric() {
            character.to_ascii_lowercase()
        } else {
            ' '
        };

        let index = normalized.len();
        normalized.push(output);

        if output == ' ' {
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
    let reference_unigrams = ngram_counts_tokenized(reference_tokens, 1);
    let prediction_unigrams = ngram_counts_tokenized(prediction_tokens, 1);
    let reference_bigrams = ngram_counts_tokenized(reference_tokens, 2);
    let prediction_bigrams = ngram_counts_tokenized(prediction_tokens, 2);

    ScoreSet {
        rouge1: score_from_counts(
            overlap_count(&reference_unigrams, &prediction_unigrams),
            reference_tokens.len(),
            prediction_tokens.len(),
        ),
        rouge2: score_from_counts(
            overlap_count(&reference_bigrams, &prediction_bigrams),
            total_ngrams(reference_tokens.len(), 2),
            total_ngrams(prediction_tokens.len(), 2),
        ),
        rouge_l: rouge_l_tokenized(reference_tokens, prediction_tokens),
    }
}

fn rouge_n_tokenized(reference_tokens: &TokenizedText, prediction_tokens: &TokenizedText, n: usize) -> Score {
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

    LCS_ROWS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        ensure_zeroed_len(&mut buffers.0, width);
        ensure_zeroed_len(&mut buffers.1, width);

        let (previous, current) = &mut *buffers;

        for row_index in 0..rows.len() {
            current[0] = 0;
            let row_token = rows.token(row_index);

            for col_index in 0..cols.len() {
                current[col_index + 1] = if row_token == cols.token(col_index) {
                    previous[col_index] + 1
                } else {
                    previous[col_index + 1].max(current[col_index])
                };
            }

            std::mem::swap(previous, current);
        }

        previous[cols.len()]
    })
}

fn cached_scores(reference: &str, prediction: &str) -> Option<ScoreSet> {
    SCORE_CACHE.with(|cache| {
        cache
            .borrow()
            .as_ref()
            .filter(|entry| entry.reference == reference && entry.prediction == prediction)
            .map(|entry| entry.scores)
    })
}

fn store_scores(reference: &str, prediction: &str, scores: ScoreSet) {
    SCORE_CACHE.with(|cache| {
        *cache.borrow_mut() = Some(PairScoreCache {
            reference: reference.to_owned(),
            prediction: prediction.to_owned(),
            scores,
        });
    });
}

fn total_ngrams(token_count: usize, n: usize) -> usize {
    token_count.checked_sub(n).map_or(0, |remaining| remaining + 1)
}

fn ensure_zeroed_len(buffer: &mut Vec<usize>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0);
    } else {
        buffer[..len].fill(0);
    }
}

fn ngram_counts_tokenized<'a>(tokens: &'a TokenizedText, n: usize) -> HashMap<NgramKey<'a>, usize> {
    let mut counts = HashMap::with_capacity_and_hasher(total_ngrams(tokens.len(), n), Default::default());

    if tokens.len() < n {
        return counts;
    }

    match n {
        1 => {
            for index in 0..tokens.len() {
                *counts.entry(NgramKey::Unigram(tokens.token(index))).or_insert(0) += 1;
            }
        }
        2 => {
            for index in 0..(tokens.len() - 1) {
                *counts
                    .entry(NgramKey::Bigram(tokens.token(index), tokens.token(index + 1)))
                    .or_insert(0) += 1;
            }
        }
        _ => {
            for start in 0..=tokens.len() - n {
                let mut key = Vec::with_capacity(n);

                for offset in 0..n {
                    key.push(tokens.token(start + offset));
                }

                *counts.entry(NgramKey::Higher(key)).or_insert(0) += 1;
            }
        }
    }

    counts
}

fn ngram_counts_from_slice<'a, T: AsRef<str>>(tokens: &'a [T], n: usize) -> HashMap<NgramKey<'a>, usize> {
    let mut counts = HashMap::with_capacity_and_hasher(total_ngrams(tokens.len(), n), Default::default());

    if tokens.len() < n {
        return counts;
    }

    match n {
        1 => {
            for token in tokens {
                *counts.entry(NgramKey::Unigram(token.as_ref())).or_insert(0) += 1;
            }
        }
        2 => {
            for window in tokens.windows(2) {
                *counts
                    .entry(NgramKey::Bigram(window[0].as_ref(), window[1].as_ref()))
                    .or_insert(0) += 1;
            }
        }
        _ => {
            for window in tokens.windows(n) {
                let mut key = Vec::with_capacity(n);

                for token in window {
                    key.push(token.as_ref());
                }

                *counts.entry(NgramKey::Higher(key)).or_insert(0) += 1;
            }
        }
    }

    counts
}

fn overlap_count(
    reference_counts: &HashMap<NgramKey<'_>, usize>,
    prediction_counts: &HashMap<NgramKey<'_>, usize>,
) -> usize {
    reference_counts
        .iter()
        .map(|(ngram, reference_count)| {
            prediction_counts
                .get(ngram)
                .map_or(0, |prediction_count| (*reference_count).min(*prediction_count))
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
    use super::{Score, lcs_len, rouge_l, rouge_n, tokenize};

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
}
