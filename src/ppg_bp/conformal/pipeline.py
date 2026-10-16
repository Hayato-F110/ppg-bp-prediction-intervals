"""Build conformal prediction intervals from tables of estimates (shared by the within- and cross-dataset analyses).

The corrections are computed on the calibration table and applied to the evaluation table. Columns are named
<target>_<model>_mean, <target>_<model>_q050 and so on. Mondrian groups use only quantities known from the input at
test time (band of the estimate, signal quality tertile sqi_tertile).
"""

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ppg_bp.conformal.core import (
    calibrate,
    cqr_interval,
    cqr_scores,
    split_interval,
    split_scores,
)

METHODS = (
    "split",
    "cqr",
    "mondrian_pred",
    "mondrian_sqi",
    "mondrian_pred_sqi",
    "cqr_mondrian_pred_sqi",
    "split_segment",
)


def sqi_tertile(sqi: ArrayLike, cuts: dict) -> np.ndarray:
    """Signal quality tertile. The cut points were set on the training subjects (configs/sqi_tertiles.yaml)."""
    sqi = np.asarray(sqi, dtype=np.float64)
    return np.select([sqi < cuts["lower"], sqi < cuts["upper"]], ["low", "middle"], "high")


def band(values: ArrayLike, edges: Sequence[float]) -> np.ndarray:
    """Split values into bands and return labels such as "<90", "90–<100" and "≥160". Missing values give None."""
    values = np.asarray(values, dtype=np.float64)
    labels = [f"<{edges[0]:g}"] + [f"{a:g}–<{b:g}" for a, b in itertools.pairwise(edges)] + [f"≥{edges[-1]:g}"]
    out = np.array(labels, dtype=object)[np.digitize(values, edges)]
    out[np.isnan(values)] = None
    return out


def quantile_column(target: str, model: str, q: float) -> str:
    return f"{target}_{model}_q{round(q * 1000):03d}"


def mondrian_groups(
    table: pd.DataFrame, target: str, model: str, method: str, edges: Sequence[float]
) -> np.ndarray | None:
    """Groups for Mondrian calibration, using only quantities known from the input at test time."""
    pred_band = band(table[f"{target}_{model}_mean"], edges)
    if method == "mondrian_pred":
        return pred_band
    if method == "mondrian_sqi":
        return table["sqi_tertile"].to_numpy()
    if method in ("mondrian_pred_sqi", "cqr_mondrian_pred_sqi"):
        return np.array([f"{a}|{b}" for a, b in zip(pred_band, table["sqi_tertile"], strict=True)], dtype=object)
    return None


def intervals(
    cal: pd.DataFrame, test: pd.DataFrame, target: str, model: str, method: str, level: float, settings: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate on cal and return the intervals for test as (lower, upper, dict of corrections)."""
    alpha = 1.0 - level
    edges = settings["strata"].get(target, settings["strata"]["sbp"] if target == "mbp" else None)
    # MBP (secondary) uses quintiles of the estimate rather than cut points between those of SBP and DBP
    if target == "mbp":
        edges = [float(x) for x in np.quantile(cal[f"{target}_{model}_mean"], [0.2, 0.4, 0.6, 0.8])]
    groups_cal, groups_test = (mondrian_groups(t, target, model, method, edges) for t in (cal, test))
    min_subjects = settings["min_subjects_per_stratum"]
    if method.startswith("cqr"):
        q_lo, q_hi = settings["levels"][level]
        lo_cal, hi_cal = cal[quantile_column(target, model, q_lo)], cal[quantile_column(target, model, q_hi)]
        lo_test, hi_test = test[quantile_column(target, model, q_lo)], test[quantile_column(target, model, q_hi)]
        # swap crossed quantiles
        lo_cal, hi_cal = np.minimum(lo_cal, hi_cal), np.maximum(lo_cal, hi_cal)
        lo_test, hi_test = np.minimum(lo_test, hi_test), np.maximum(lo_test, hi_test)
        corrections = calibrate(
            cqr_scores(cal[target], lo_cal, hi_cal), cal["subjectid"], alpha, groups_cal, min_subjects
        )
        return (*cqr_interval(lo_test, hi_test, corrections, groups_test), corrections)
    scores = split_scores(cal[target], cal[f"{target}_{model}_mean"])
    corrections = calibrate(
        scores, cal["subjectid"], alpha, groups_cal, min_subjects, equal_subjects=method != "split_segment"
    )
    return (*split_interval(test[f"{target}_{model}_mean"], corrections, groups_test), corrections)


def read_predictions(results: Path, name: str, cuts: dict) -> pd.DataFrame:
    """Table of estimates with the signal-quality tertile added.

    name is "calibration" or "test" (VitalDB) or the name of an external dataset such as "mimic3wdb".
    """
    if name in ("calibration", "test"):
        path = results / "conformal" / f"predictions_{name}.parquet"
    else:
        path = results / "external" / name / "predictions.parquet"
    table = pd.read_parquet(path)
    table["sqi_tertile"] = sqi_tertile(table["sqi_template"], cuts)
    return table
