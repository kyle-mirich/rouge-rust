use crate::scorer;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass(name = "Score", module = "fast_rouge", skip_from_py_object)]
#[derive(Clone)]
/// Precision, recall, and F-measure values for one ROUGE metric.
struct PyScore {
    #[pyo3(get)]
    precision: f64,
    #[pyo3(get)]
    recall: f64,
    #[pyo3(get)]
    fmeasure: f64,
}

impl From<scorer::Score> for PyScore {
    fn from(score: scorer::Score) -> Self {
        Self {
            precision: score.precision,
            recall: score.recall,
            fmeasure: score.fmeasure,
        }
    }
}

#[pyclass(name = "BatchScoreResult", module = "fast_rouge")]
#[allow(non_snake_case)]
/// Column-oriented ROUGE scores returned by `score_batch_flat`.
struct BatchScoreResult {
    #[pyo3(get)]
    rouge1_precision: Vec<f64>,
    #[pyo3(get)]
    rouge1_recall: Vec<f64>,
    #[pyo3(get)]
    rouge1_fmeasure: Vec<f64>,
    #[pyo3(get)]
    rouge2_precision: Vec<f64>,
    #[pyo3(get)]
    rouge2_recall: Vec<f64>,
    #[pyo3(get)]
    rouge2_fmeasure: Vec<f64>,
    #[pyo3(get)]
    rougeL_precision: Vec<f64>,
    #[pyo3(get)]
    rougeL_recall: Vec<f64>,
    #[pyo3(get)]
    rougeL_fmeasure: Vec<f64>,
}

struct ScoreBundle {
    rouge1: scorer::Score,
    rouge2: scorer::Score,
    rouge_l: scorer::Score,
}

fn compute_scores(reference: &str, prediction: &str) -> ScoreBundle {
    let (rouge1, rouge2, rouge_l) = scorer::score_all(reference, prediction);

    ScoreBundle {
        rouge1,
        rouge2,
        rouge_l,
    }
}

fn to_python_dict(py: Python<'_>, scores: ScoreBundle) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("rouge1", Py::new(py, PyScore::from(scores.rouge1))?)?;
    result.set_item("rouge2", Py::new(py, PyScore::from(scores.rouge2))?)?;
    result.set_item("rougeL", Py::new(py, PyScore::from(scores.rouge_l))?)?;
    Ok(result.unbind())
}

/// Score one reference/prediction pair with ROUGE-1, ROUGE-2, and ROUGE-L.
///
/// Returns a dict mapping metric names to read-only Score objects. Empty
/// token sequences produce zero scores. Rust computation releases the GIL.
#[pyfunction]
fn score(py: Python<'_>, reference: &str, prediction: &str) -> PyResult<Py<PyDict>> {
    let scores = py.detach(|| compute_scores(reference, prediction));
    to_python_dict(py, scores)
}

/// Score equally sized reference and prediction sequences in parallel.
///
/// Returns a list of score dicts in input order. Raises ValueError when lengths
/// differ and TypeError for non-string elements. Rust computation releases the GIL.
#[pyfunction]
fn score_batch(
    py: Python<'_>,
    references: Vec<String>,
    predictions: Vec<String>,
) -> PyResult<Vec<Py<PyDict>>> {
    if references.len() != predictions.len() {
        return Err(PyValueError::new_err(
            "references and predictions must have the same length",
        ));
    }

    let results: Vec<ScoreBundle> = py.detach(|| {
        references
            .into_par_iter()
            .zip(predictions.into_par_iter())
            .map(|(reference, prediction)| compute_scores(&reference, &prediction))
            .collect()
    });

    results
        .into_iter()
        .map(|scores| to_python_dict(py, scores))
        .collect()
}

/// Score equally sized sequences and return nine metric columns in input order.
///
/// Each attribute access copies a column to a Python list; retain the list for
/// repeated access. Raises ValueError for unequal lengths. Releases the GIL.
#[pyfunction]
fn score_batch_flat(
    py: Python<'_>,
    references: Vec<String>,
    predictions: Vec<String>,
) -> PyResult<Py<BatchScoreResult>> {
    if references.len() != predictions.len() {
        return Err(PyValueError::new_err(
            "references and predictions must have the same length",
        ));
    }

    let result = py.detach(|| {
        let capacity = references.len();
        let mut result = BatchScoreResult {
            rouge1_precision: vec![0.0; capacity],
            rouge1_recall: vec![0.0; capacity],
            rouge1_fmeasure: vec![0.0; capacity],
            rouge2_precision: vec![0.0; capacity],
            rouge2_recall: vec![0.0; capacity],
            rouge2_fmeasure: vec![0.0; capacity],
            rougeL_precision: vec![0.0; capacity],
            rougeL_recall: vec![0.0; capacity],
            rougeL_fmeasure: vec![0.0; capacity],
        };

        (
            references.into_par_iter(),
            predictions.into_par_iter(),
            result.rouge1_precision.par_iter_mut(),
            result.rouge1_recall.par_iter_mut(),
            result.rouge1_fmeasure.par_iter_mut(),
            result.rouge2_precision.par_iter_mut(),
            result.rouge2_recall.par_iter_mut(),
            result.rouge2_fmeasure.par_iter_mut(),
            result.rougeL_precision.par_iter_mut(),
            result.rougeL_recall.par_iter_mut(),
            result.rougeL_fmeasure.par_iter_mut(),
        )
            .into_par_iter()
            .for_each(
                |(
                    reference,
                    prediction,
                    rouge1_precision,
                    rouge1_recall,
                    rouge1_fmeasure,
                    rouge2_precision,
                    rouge2_recall,
                    rouge2_fmeasure,
                    rouge_l_precision,
                    rouge_l_recall,
                    rouge_l_fmeasure,
                )| {
                    let scores = compute_scores(&reference, &prediction);
                    *rouge1_precision = scores.rouge1.precision;
                    *rouge1_recall = scores.rouge1.recall;
                    *rouge1_fmeasure = scores.rouge1.fmeasure;
                    *rouge2_precision = scores.rouge2.precision;
                    *rouge2_recall = scores.rouge2.recall;
                    *rouge2_fmeasure = scores.rouge2.fmeasure;
                    *rouge_l_precision = scores.rouge_l.precision;
                    *rouge_l_recall = scores.rouge_l.recall;
                    *rouge_l_fmeasure = scores.rouge_l.fmeasure;
                },
            );

        result
    });

    Py::new(py, result)
}

/// Fast ROUGE-1, ROUGE-2, and ROUGE-L scoring with stemming disabled.
#[pymodule]
fn fast_rouge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyScore>()?;
    m.add_class::<BatchScoreResult>()?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(score_batch_flat, m)?)?;
    Ok(())
}
