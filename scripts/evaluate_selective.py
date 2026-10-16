"""Selective prediction: no estimate is given when the interval is wider than a threshold (analysis plan, Section 7).

The intervals are CQR intervals, whose width varies with the input. The thresholds come from the widths of the
calibration patients alone (the widths at which 10%, 20%, ..., 100% of their segments are kept). The evaluation is in
the test patients and, if present, in the external dataset. The share kept is always reported.

usage: uv run python -X utf8 scripts/evaluate_selective.py [--dataset mimic3wdb]
output: results/conformal/selective.csv
"""

import argparse

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import calibrate, cqr_interval, cqr_scores
from ppg_bp.conformal.pipeline import quantile_column

MODELS = ("lightgbm", "cnn")
TARGETS = ("sbp", "dbp", "mbp")
RATES = np.round(np.arange(0.1, 1.01, 0.1), 2)


def cqr_bounds(table, target, model, q_lo, q_hi):
    lo, hi = (
        table[quantile_column(target, model, q_lo)].to_numpy(),
        table[quantile_column(target, model, q_hi)].to_numpy(),
    )
    return np.minimum(lo, hi), np.maximum(lo, hi)


def kept_summary(y, pred, lower, upper, subjects, keep) -> dict:
    """Accuracy and coverage of the segments kept."""
    error = pred[keep] - y[keep]
    covered = (y[keep] >= lower[keep]) & (y[keep] <= upper[keep])
    by_subject = pd.Series(covered).groupby(subjects[keep]).mean()
    width = upper - lower
    return {
        "accepted": keep.mean(),
        "subjects_with_any": by_subject.size / len(np.unique(subjects)),
        "mae": np.abs(error).mean(),
        "me": error.mean(),
        "sd": error.std(),
        "coverage_subject": by_subject.mean(),
        "width_median": float(np.median(width[keep])),
        "true_mean": y[keep].mean(),
        "true_sd": y[keep].std(),
    }


def selective_rows(cal, datasets, model, target, level, quantiles) -> list[dict]:
    q_lo, q_hi = quantiles
    lo_c, hi_c = cqr_bounds(cal, target, model, q_lo, q_hi)
    correction = calibrate(cqr_scores(cal[target], lo_c, hi_c), cal["subjectid"], 1 - level)
    width_cal = np.subtract(*cqr_interval(lo_c, hi_c, correction)[::-1])
    thresholds = np.quantile(width_cal, RATES)  # from the calibration widths only
    rows = []
    for name, table in datasets.items():
        lower, upper = cqr_interval(*cqr_bounds(table, target, model, q_lo, q_hi), correction)
        y, pred = table[target].to_numpy(), table[f"{target}_{model}_mean"].to_numpy()
        subjects = table["subjectid"].to_numpy()
        for rate, threshold in zip(RATES, thresholds, strict=True):
            keep = upper - lower <= threshold
            rows.append(
                {
                    "dataset": name,
                    "model": model,
                    "target": target,
                    "nominal_accept": rate,
                    "width_threshold": threshold,
                    **kept_summary(y, pred, lower, upper, subjects, keep),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic3wdb")
    args = parser.parse_args()
    config = load_config()
    settings = load_settings("conformal")
    level = settings["primary_level"]
    results = PROJECT_ROOT / config["paths"]["results"]
    cal = pd.read_parquet(results / "conformal" / "predictions_calibration.parquet")
    datasets = {"vitaldb_test": pd.read_parquet(results / "conformal" / "predictions_test.parquet")}
    external = results / "external" / args.dataset / "predictions.parquet"
    if external.exists():
        datasets[args.dataset] = pd.read_parquet(external)
    rows = []
    for model in MODELS:
        for target in TARGETS:
            rows += selective_rows(cal, datasets, model, target, level, settings["levels"][level])
    pd.DataFrame(rows).to_csv(results / "conformal" / "selective.csv", index=False, float_format="%.6f")
    print("done")


if __name__ == "__main__":
    main()
