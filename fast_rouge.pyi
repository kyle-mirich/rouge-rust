"""Unstemmed ROUGE-1, ROUGE-2, and ROUGE-L with reference-compatible tokenization."""

from typing import Dict, List, Sequence, final

__all__ = ["__version__", "Score", "BatchScoreResult", "score", "score_batch", "score_batch_flat"]

__version__: str

@final
class Score:
    """Read-only precision, recall, and F1 values for one metric."""
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def fmeasure(self) -> float: ...

@final
class BatchScoreResult:
    """Nine score columns. Each property access returns a fresh Python list."""
    @property
    def rouge1_precision(self) -> List[float]: ...
    @property
    def rouge1_recall(self) -> List[float]: ...
    @property
    def rouge1_fmeasure(self) -> List[float]: ...
    @property
    def rouge2_precision(self) -> List[float]: ...
    @property
    def rouge2_recall(self) -> List[float]: ...
    @property
    def rouge2_fmeasure(self) -> List[float]: ...
    @property
    def rougeL_precision(self) -> List[float]: ...
    @property
    def rougeL_recall(self) -> List[float]: ...
    @property
    def rougeL_fmeasure(self) -> List[float]: ...

def score(reference: str, prediction: str) -> Dict[str, Score]:
    """Score one pair. Empty token sequences produce zero-valued scores."""
    ...

def score_batch(references: Sequence[str], predictions: Sequence[str]) -> List[Dict[str, Score]]:
    """Score equal-length sequences in parallel, preserving input order."""
    ...

def score_batch_flat(references: Sequence[str], predictions: Sequence[str]) -> BatchScoreResult:
    """Score equal-length sequences in parallel and return metric columns."""
    ...
