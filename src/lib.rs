pub mod scorer;

#[cfg(not(test))]
use pyo3::exceptions::PyValueError;
#[cfg(not(test))]
use pyo3::prelude::*;
#[cfg(not(test))]
use pyo3::types::PyDict;
#[cfg(not(test))]
use rayon::prelude::*;
#[cfg(not(test))]
use std::mem::{ManuallyDrop, MaybeUninit};

/// Returns a constant string so the Python bridge can be smoke-tested.
#[cfg(not(test))]
#[pyfunction]
fn dummy_score() -> &'static str {
    "fast_rouge-ready"
}

#[cfg(not(test))]
#[pyclass(name = "Score", module = "fast_rouge")]
#[derive(Clone)]
struct PyScore {
    #[pyo3(get)]
    precision: f64,
    #[pyo3(get)]
    recall: f64,
    #[pyo3(get)]
    fmeasure: f64,
}

#[cfg(not(test))]
impl From<scorer::Score> for PyScore {
    fn from(score: scorer::Score) -> Self {
        Self {
            precision: score.precision,
            recall: score.recall,
            fmeasure: score.fmeasure,
        }
    }
}

#[cfg(not(test))]
#[pyclass(name = "BatchScoreResult", module = "fast_rouge")]
#[allow(non_snake_case)]
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

#[cfg(not(test))]
struct ScoreBundle {
    rouge1: scorer::Score,
    rouge2: scorer::Score,
    rouge_l: scorer::Score,
}

#[cfg(not(test))]
fn compute_scores(reference: &str, prediction: &str) -> ScoreBundle {
    let (rouge1, rouge2, rouge_l) = scorer::score_all(reference, prediction);

    ScoreBundle {
        rouge1,
        rouge2,
        rouge_l,
    }
}

#[cfg(not(test))]
fn to_python_dict(py: Python<'_>, scores: ScoreBundle) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("rouge1", Py::new(py, PyScore::from(scores.rouge1))?)?;
    result.set_item("rouge2", Py::new(py, PyScore::from(scores.rouge2))?)?;
    result.set_item("rougeL", Py::new(py, PyScore::from(scores.rouge_l))?)?;
    Ok(result.unbind())
}

#[cfg(not(test))]
fn uninit_f64_vec(len: usize) -> Vec<MaybeUninit<f64>> {
    let mut values = Vec::with_capacity(len);
    unsafe {
        values.set_len(len);
    }
    values
}

#[cfg(not(test))]
fn assume_init_f64_vec(values: Vec<MaybeUninit<f64>>) -> Vec<f64> {
    let mut values = ManuallyDrop::new(values);
    unsafe {
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut f64,
            values.len(),
            values.capacity(),
        )
    }
}

#[cfg(not(test))]
#[pyfunction]
fn score(py: Python<'_>, reference: &str, prediction: &str) -> PyResult<Py<PyDict>> {
    to_python_dict(py, compute_scores(reference, prediction))
}

#[cfg(not(test))]
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

    let results: Vec<ScoreBundle> = references
        .into_par_iter()
        .zip(predictions.into_par_iter())
        .map(|(reference, prediction)| compute_scores(&reference, &prediction))
        .collect();

    results
        .into_iter()
        .map(|scores| to_python_dict(py, scores))
        .collect()
}

#[cfg(not(test))]
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

    let capacity = references.len();
    let mut rouge1_precision = uninit_f64_vec(capacity);
    let mut rouge1_recall = uninit_f64_vec(capacity);
    let mut rouge1_fmeasure = uninit_f64_vec(capacity);
    let mut rouge2_precision = uninit_f64_vec(capacity);
    let mut rouge2_recall = uninit_f64_vec(capacity);
    let mut rouge2_fmeasure = uninit_f64_vec(capacity);
    let mut rouge_l_precision = uninit_f64_vec(capacity);
    let mut rouge_l_recall = uninit_f64_vec(capacity);
    let mut rouge_l_fmeasure = uninit_f64_vec(capacity);

    (
        references.into_par_iter(),
        predictions.into_par_iter(),
        rouge1_precision.par_iter_mut(),
        rouge1_recall.par_iter_mut(),
        rouge1_fmeasure.par_iter_mut(),
        rouge2_precision.par_iter_mut(),
        rouge2_recall.par_iter_mut(),
        rouge2_fmeasure.par_iter_mut(),
        rouge_l_precision.par_iter_mut(),
        rouge_l_recall.par_iter_mut(),
        rouge_l_fmeasure.par_iter_mut(),
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
                rouge1_precision.write(scores.rouge1.precision);
                rouge1_recall.write(scores.rouge1.recall);
                rouge1_fmeasure.write(scores.rouge1.fmeasure);
                rouge2_precision.write(scores.rouge2.precision);
                rouge2_recall.write(scores.rouge2.recall);
                rouge2_fmeasure.write(scores.rouge2.fmeasure);
                rouge_l_precision.write(scores.rouge_l.precision);
                rouge_l_recall.write(scores.rouge_l.recall);
                rouge_l_fmeasure.write(scores.rouge_l.fmeasure);
            },
        );

    Py::new(
        py,
        BatchScoreResult {
            rouge1_precision: assume_init_f64_vec(rouge1_precision),
            rouge1_recall: assume_init_f64_vec(rouge1_recall),
            rouge1_fmeasure: assume_init_f64_vec(rouge1_fmeasure),
            rouge2_precision: assume_init_f64_vec(rouge2_precision),
            rouge2_recall: assume_init_f64_vec(rouge2_recall),
            rouge2_fmeasure: assume_init_f64_vec(rouge2_fmeasure),
            rougeL_precision: assume_init_f64_vec(rouge_l_precision),
            rougeL_recall: assume_init_f64_vec(rouge_l_recall),
            rougeL_fmeasure: assume_init_f64_vec(rouge_l_fmeasure),
        },
    )
}

/// A Python module implemented in Rust.
#[cfg(not(test))]
#[pymodule]
fn fast_rouge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScore>()?;
    m.add_class::<BatchScoreResult>()?;
    m.add_function(wrap_pyfunction!(dummy_score, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(score_batch_flat, m)?)?;
    Ok(())
}
