"""Build the conformal prediction intervals and evaluate them in the test patients (within VitalDB).

Calibration uses only the calibration patients. The evaluation and the strata are fixed in Section 4 of the
analysis plan. Methods:
    split                  absolute residual, fixed width
    cqr                    conformalised quantile regression
    mondrian_pred          split, calibrated within bands of the estimate (group-wise)
    mondrian_sqi           split, calibrated within signal-quality tertiles
    mondrian_pred_sqi      split, calibrated within both
    cqr_mondrian_pred_sqi  cqr, calibrated within both
    split_segment          split with equal weight per segment (sensitivity analysis)

usage: uv run python -X utf8 scripts/evaluate_conformal.py [--quick]
output: results/conformal/coverage.csv (overall and by stratum), corrections.csv, calibration_size.csv,
        subject_coverage.parquet
"""

import sys

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import (
    calibrate,
    cqr_interval,
    cqr_scores,
    split_interval,
    split_scores,
)
from ppg_bp.conformal.pipeline import (
    METHODS,
    band,
    intervals,
    quantile_column,
    sqi_tertile,
)
from ppg_bp.evaluation.coverage import evaluate, per_subject

QUICK = "--quick" in sys.argv
MODELS = ("lightgbm", "cnn")
TARGETS = ("sbp", "dbp", "mbp")


def add_columns(table, settings, cuts, bmi):
    """Add the stratum of each segment for every variable of the subgroup analysis."""
    table = table.merge(bmi, on="caseid", how="left")
    table["sqi_tertile"] = sqi_tertile(table["sqi_template"], cuts)
    strata = settings["strata"]
    median = table.groupby("subjectid")["sbp"].transform("median")
    table["stratum_true_sbp"] = band(table["sbp"], strata["sbp"])
    table["stratum_true_dbp"] = band(table["dbp"], strata["dbp"])
    table["stratum_age"] = band(table["age"], strata["age"])
    table["stratum_sex"] = table["sex"]
    table["stratum_bmi"] = band(table["bmi"], strata["bmi"])
    table["stratum_pulse_rate"] = band(table["pulse_rate"], strata["pulse_rate"])
    table["stratum_sqi"] = table["sqi_tertile"]
    table["stratum_within_subject"] = band(table["sbp"] - median, strata["within_subject"])
    return table


def draw_interval(chosen, test, method, target, model, level, quantiles):
    """Calibrate on the chosen patients only and return (lower, upper, corrections) for the test patients."""
    if method == "split":
        q = calibrate(split_scores(chosen[target], chosen[f"{target}_{model}_mean"]), chosen["subjectid"], 1 - level)
        return (*split_interval(test[f"{target}_{model}_mean"], q), q)
    q_lo, q_hi = quantiles
    lo, hi = chosen[quantile_column(target, model, q_lo)], chosen[quantile_column(target, model, q_hi)]
    q = calibrate(cqr_scores(chosen[target], np.minimum(lo, hi), np.maximum(lo, hi)), chosen["subjectid"], 1 - level)
    lo_t, hi_t = test[quantile_column(target, model, q_lo)], test[quantile_column(target, model, q_hi)]
    return (*cqr_interval(np.minimum(lo_t, hi_t), np.maximum(lo_t, hi_t), q), q)


def calibration_size_experiment(cal, test, settings, seed):
    """Vary the number of calibration patients and record the patient-level coverage in the test patients
    (analysis plan, Section 5; SBP, nominal 90%, LightGBM)."""
    target, model, level = settings["primary_target"], "lightgbm", settings["primary_level"]
    rng = np.random.default_rng(seed)
    subjects = cal["subjectid"].unique()
    by_subject = dict(iter(cal.groupby("subjectid")))
    rows = []
    repeats = 5 if QUICK else settings["calibration_repeats"]
    for requested in settings["calibration_sizes"]:
        size = min(requested, len(subjects))
        for repeat in range(1 if size == len(subjects) else repeats):
            chosen = pd.concat([by_subject[s] for s in rng.choice(subjects, size=size, replace=False)])
            for method in ("split", "cqr"):
                lower, upper, q = draw_interval(chosen, test, method, target, model, level, settings["levels"][level])
                result = per_subject(test[target], lower, upper, test["subjectid"])
                rows.append(
                    {
                        "method": method,
                        "subjects": size,
                        "repeat": repeat,
                        "correction": q[None],
                        "coverage": result["covered"].mean(),
                        "width_median": result["width"].median(),
                    }
                )
    return pd.DataFrame(rows)


def strata_for(test, target, primary):
    """All subgroup variables for the primary target (SBP); only the band of the true value for the others."""
    columns = [c for c in test.columns if c.startswith("stratum_")]
    wanted = columns if primary else [c for c in columns if c == f"stratum_true_{target}"]
    return {c.removeprefix("stratum_"): test[c].to_numpy() for c in wanted}


def evaluate_all(cal, test, settings, n_boot, seed):
    """Every model, target, level and method. Returns (coverage, corrections, per-patient coverage)."""
    coverage_rows, correction_rows, subject_rows = [], [], []
    for model in MODELS:
        for target in TARGETS:
            primary = target == settings["primary_target"]
            for level in settings["levels"]:
                for method in METHODS:
                    lower, upper, corrections = intervals(cal, test, target, model, method, level, settings)
                    result = evaluate(
                        test[target],
                        lower,
                        upper,
                        test["subjectid"],
                        strata_for(test, target, primary),
                        n_boot=n_boot,
                        seed=seed,
                        nominal=level,
                    )
                    keys = {"model": model, "target": target, "level": level, "method": method}
                    for position, (name, value) in enumerate(keys.items()):
                        result.insert(position, name, value)
                    coverage_rows.append(result)
                    correction_rows += [{**keys, "group": str(g), "correction": q} for g, q in corrections.items()]
                    if primary and level == settings["primary_level"]:
                        table = per_subject(test[target], lower, upper, test["subjectid"]).reset_index()
                        table.insert(0, "method", method)
                        table.insert(0, "model", model)
                        subject_rows.append(table)
            print(f"{model} {target} done", flush=True)
    coverage = pd.concat(coverage_rows, ignore_index=True)
    coverage["reference_only"] = coverage["subjects"] < settings["min_subjects_per_stratum"]
    return coverage, pd.DataFrame(correction_rows), pd.concat(subject_rows, ignore_index=True)


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    out_dir = PROJECT_ROOT / config["paths"]["results"] / "conformal" / ("quick" if QUICK else "")
    bmi = pd.read_csv(PROJECT_ROOT / config["paths"]["processed"] / "vitaldb" / "split.csv", usecols=["caseid", "bmi"])
    cal = add_columns(pd.read_parquet(out_dir / "predictions_calibration.parquet"), settings, cuts, bmi)
    test = add_columns(pd.read_parquet(out_dir / "predictions_test.parquet"), settings, cuts, bmi)
    if set(cal["subjectid"]) & set(test["subjectid"]):
        raise AssertionError("a patient is in both calibration and test")
    print(
        f"calibration {cal['subjectid'].nunique()} patients {len(cal)} segments, "
        f"test {test['subjectid'].nunique()} patients {len(test)} segments",
        flush=True,
    )
    n_boot = 100 if QUICK else settings["bootstrap"]
    coverage, corrections, subjects = evaluate_all(cal, test, settings, n_boot, seed)
    coverage.to_csv(out_dir / "coverage.csv", index=False, float_format="%.6f")
    corrections.to_csv(out_dir / "corrections.csv", index=False, float_format="%.6f")
    subjects.to_parquet(out_dir / "subject_coverage.parquet", index=False)
    calibration_size_experiment(cal, test, settings, seed).to_csv(
        out_dir / "calibration_size.csv", index=False, float_format="%.6f"
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
