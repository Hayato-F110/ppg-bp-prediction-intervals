"""Build conformal prediction intervals from tables of estimates (shared by the within- and cross-dataset analyses).

The corrections are computed on the calibration table and applied to the evaluation table. Columns are named
<target>_<model>_mean, <target>_<model>_q050 and so on. Mondrian groups use only quantities known from the input at
test time (band of the estimate, signal quality tertile sqi_tertile).
"""

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

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
from ppg_bp.constants import MAIN_LEVEL
from ppg_bp.evaluation.coverage import evaluate

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


def band_names(edges: Sequence[float]) -> list[str]:
    """Names of the bands that band() gives, in order: "<90", "90–<100", ..., "≥160"."""
    return [f"<{edges[0]:g}"] + [f"{a:g}–<{b:g}" for a, b in itertools.pairwise(edges)] + [f"≥{edges[-1]:g}"]


def band_labels(edges: Sequence[float]) -> list[str]:
    """Labels of the bands for tables and figures (whole mmHg): "<90", "90–99", ..., "≥160"."""
    return [f"<{edges[0]:g}"] + [f"{a:g}–{b - 1:g}" for a, b in itertools.pairwise(edges)] + [f"≥{edges[-1]:g}"]


def band(values: ArrayLike, edges: Sequence[float]) -> np.ndarray:
    """Split values into bands and return labels such as "<90", "90–<100" and "≥160". Missing values give None."""
    values = np.asarray(values, dtype=np.float64)
    out = np.array(band_names(edges), dtype=object)[np.digitize(values, edges)]
    out[np.isnan(values)] = None
    return out


def quantile_column(target: str, model: str, q: float) -> str:
    return f"{target}_{model}_q{round(q * 1000):03d}"


def mondrian_groups(
    table: pd.DataFrame, target: str, model: str, method: str, edges: Sequence[float]
) -> np.ndarray | None:
    """Groups for Mondrian calibration, using only quantities known from the input at test time."""
    if method == "mondrian_sqi":
        return table["sqi_tertile"].to_numpy()
    if method not in ("mondrian_pred", "mondrian_pred_sqi", "cqr_mondrian_pred_sqi"):
        return None  # split, cqr and split_segment use no groups
    pred_band = band(table[f"{target}_{model}_mean"], edges)
    if method == "mondrian_pred":
        return pred_band
    return np.array([f"{a}|{b}" for a, b in zip(pred_band, table["sqi_tertile"], strict=True)], dtype=object)


def intervals(
    cal: pd.DataFrame, test: pd.DataFrame, target: str, model: str, method: str, level: float, settings: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate on cal and return the intervals for test as (lower, upper, dict of corrections)."""
    alpha = 1.0 - level
    edges = settings["strata"].get(target)
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


def coverage_rows(y, lower, upper, subjects, edges, n_boot, seed, **labels) -> pd.DataFrame:
    """Patient-level coverage overall and by band of true SBP, at the main nominal level; the keyword arguments
    become leading columns (exploratory analyses)."""
    result = evaluate(
        y, lower, upper, subjects, {"true_sbp": band(y, edges)}, n_boot=n_boot, seed=seed, nominal=MAIN_LEVEL
    )
    for position, (name, value) in enumerate(labels.items()):
        result.insert(position, name, value)
    return result


def with_offset(table: pd.DataFrame, n_first: int, target: str = "sbp", model: str = "lightgbm") -> pd.DataFrame:
    """Use each patient's first n_first segments as the reference reading: their mean error (reference - estimate)
    is added to the patient's later estimates and quantiles, and the reference segments are dropped. Patients
    with no later segment drop out."""
    order = [c for c in ("subjectid", "caseid", "index") if c in table.columns]  # MIMIC-III has no caseid
    table = table.sort_values(order).reset_index(drop=True)
    is_reference = table.groupby("subjectid").cumcount() < n_first
    error = table[target] - table[f"{target}_{model}_mean"]
    offset = error.where(is_reference).groupby(table["subjectid"]).transform("mean")
    # a boolean row selection returns a DataFrame; pandas' type information cannot tell it from a column selection
    out = cast("pd.DataFrame", table[~is_reference]).copy()
    for column in [c for c in table.columns if c.startswith(f"{target}_{model}_")]:
        out[column] = out[column] + offset[~is_reference]
    return out


def reference_reading(
    cal: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    settings: dict,
    seed: int,
    methods: tuple[str, ...] = ("split", "cqr"),
    prepare: Callable[[pd.DataFrame], pd.DataFrame] = lambda table: table,
    target: str = "sbp",
    model: str = "lightgbm",
) -> pd.DataFrame:
    """One reference reading per patient (with_offset) from the first 1, 6 or 30 segments (10 s, 1 min, 5 min).

    prepare is applied to every table first (for example, to replace the estimates by zeros, so that the estimate
    becomes the reference reading itself). The intervals are calibrated on cal treated the same way; coverage
    by band of true SBP at the main nominal level, with the MAE and the SD of the error of the corrected estimate.
    """
    edges = settings["strata"]["sbp"]
    rows = []
    for n_first in (1, 6, 30):
        cal_o = with_offset(prepare(cal), n_first, target, model)
        for name, table in tables.items():
            corrected = with_offset(prepare(table), n_first, target, model)
            error = corrected[f"{target}_{model}_mean"] - corrected[target]
            for method in methods:
                lower, upper, _ = intervals(cal_o, corrected, target, model, method, MAIN_LEVEL, settings)
                result = coverage_rows(
                    corrected[target],
                    lower,
                    upper,
                    corrected["subjectid"],
                    edges,
                    settings["bootstrap"],
                    seed,
                    reference_segments=n_first,
                    dataset=name,
                    method=method,
                )
                result.insert(3, "mae", float(error.abs().mean()))
                result.insert(4, "sd_error", float(error.std()))
                rows.append(result)
    return pd.concat(rows, ignore_index=True)
