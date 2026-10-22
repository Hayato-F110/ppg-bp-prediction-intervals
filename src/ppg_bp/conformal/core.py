"""Conformal prediction: split (absolute residual), CQR and Mondrian (group-wise) calibration.

Calibration gives every subject the same total weight (analysis plan, Section 4.1). Each segment is weighted by
1 / (number of segments of its subject), and the correction is the weighted quantile with one new subject's weight
added. Subjects with many segments therefore do not dominate the correction.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def subject_weights(subjects: ArrayLike) -> np.ndarray:
    """Weight of each segment; the weights of each subject sum to 1."""
    subjects = pd.Series(np.asarray(subjects))
    return (1.0 / subjects.map(subjects.value_counts())).to_numpy()


def conformal_quantile(scores: ArrayLike, weights: ArrayLike, alpha: float, test_weight: float = 1.0) -> float:
    """Weighted conformal quantile.

    The (1 − alpha) quantile of the distribution with weight w_i on each calibration score and test_weight on the
    new point. The new point's score is treated as +inf, so +inf is returned when there are too few calibration data.
    The usual unweighted method corresponds to all weights equal to 1 and test_weight = 1.
    """
    scores, weights = np.asarray(scores, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order]) / (weights.sum() + test_weight)
    position = np.searchsorted(cumulative, 1.0 - alpha, side="left")
    return float(scores[order][position]) if position < len(scores) else float("inf")


def split_scores(y: ArrayLike, pred: ArrayLike) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))


def cqr_scores(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """CQR score: how far y lies outside the quantile interval (negative inside)."""
    y = np.asarray(y, dtype=np.float64)
    return np.maximum(np.asarray(lower, dtype=np.float64) - y, y - np.asarray(upper, dtype=np.float64))


def calibrate(
    scores: ArrayLike,
    subjects: ArrayLike,
    alpha: float,
    groups: ArrayLike | None = None,
    min_subjects: int = 30,
    equal_subjects: bool = True,
) -> dict:
    """Compute the corrections. Returns {group name: correction}; the key None holds the overall correction.

    If groups is given, each group is calibrated separately (Mondrian). Groups with fewer than min_subjects subjects
    use the overall correction, as do segments without a group (None, e.g. a missing estimate in band()).
    equal_subjects=False weights all segments equally (sensitivity analysis).
    """
    scores, subjects = np.asarray(scores, dtype=np.float64), np.asarray(subjects)

    def one(mask: np.ndarray) -> float:
        weights = subject_weights(subjects[mask]) if equal_subjects else np.ones(int(mask.sum()))
        return conformal_quantile(scores[mask], weights, alpha)

    everything = np.ones(len(scores), dtype=bool)
    out: dict = {None: one(everything)}
    if groups is not None:
        groups = np.asarray(groups)
        for group in pd.unique(groups):
            if group is None:  # None is the key of the overall correction
                continue
            mask = groups == group
            out[group] = one(mask) if len(pd.unique(subjects[mask])) >= min_subjects else out[None]
    return out


def lookup(corrections: dict, groups: ArrayLike | None, n: int) -> np.ndarray:
    """Correction for each segment; groups not seen in calibration use the overall correction."""
    if groups is None:
        return np.full(n, corrections[None])
    return np.array([corrections.get(group, corrections[None]) for group in np.asarray(groups)], dtype=np.float64)


def split_interval(
    pred: ArrayLike, corrections: dict, groups: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    q = lookup(corrections, groups, len(pred))
    return pred - q, pred + q


def cqr_interval(
    lower: ArrayLike, upper: ArrayLike, corrections: dict, groups: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    q = lookup(corrections, groups, len(lower))
    return lower - q, upper + q
