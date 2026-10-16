"""Accuracy measures for point estimates. Error = estimate − reference (mmHg)."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# BHS grades: minimum percentages of absolute errors within 5, 10 and 15 mmHg
BHS_GRADES = (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85)))
AAMI_MAX_ABS_ME, AAMI_MAX_SD = 5.0, 8.0


def bhs_grade(within5: float, within10: float, within15: float) -> str:
    for grade, (p5, p10, p15) in BHS_GRADES:
        if within5 >= p5 and within10 >= p10 and within15 >= p15:
            return grade
    return "D"


def point_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """MAE, mean error, SD of the error, BHS percentages and grade, and whether the AAMI error criterion is met."""
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    absolute = np.abs(error)
    within = [100.0 * np.mean(absolute <= limit) for limit in (5, 10, 15)]
    me, sd = float(error.mean()), float(error.std(ddof=1))
    return {
        "n": len(error),
        "mae": float(absolute.mean()),
        "me": me,
        "sd": sd,
        "within5": within[0],
        "within10": within[1],
        "within15": within[2],
        "bhs": bhs_grade(*within),
        "aami_error_ok": bool(abs(me) <= AAMI_MAX_ABS_ME and sd <= AAMI_MAX_SD),
    }


def subject_level_mae(y_true: ArrayLike, y_pred: ArrayLike, subjects: ArrayLike) -> dict:
    """Mean and quartiles of the per-subject MAE (not dominated by subjects with many segments)."""
    table = pd.DataFrame({"subject": np.asarray(subjects), "abs": np.abs(np.asarray(y_pred) - np.asarray(y_true))})
    per_subject = table.groupby("subject")["abs"].mean()
    q25, q50, q75 = per_subject.quantile([0.25, 0.5, 0.75])
    return {
        "subjects": len(per_subject),
        "subject_mae_mean": float(per_subject.mean()),
        "subject_mae_q25": float(q25),
        "subject_mae_median": float(q50),
        "subject_mae_q75": float(q75),
    }
