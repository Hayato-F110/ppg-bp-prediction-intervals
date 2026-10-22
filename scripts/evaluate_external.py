"""Evaluation across datasets (analysis plan, Section 6): models and intervals from VitalDB applied to external data.

6.1  applied unchanged: coverage, width and direction of misses (overall and by stratum)
6.2  correction with unlabelled target data: weighted conformal prediction (density ratio from a domain classifier)
6.3  recalibration with a few target subjects (5 to 100, 200 draws each)
6.4  sources of the shift: AUC of the classifier, distance between BP distributions, coverage with matched BP

The models are not retrained. No raw values are printed.

usage: uv run python -X utf8 scripts/evaluate_external.py [--dataset mimic3wdb]
output: results/external/<dataset>/coverage.csv, recalibration.csv, weighted.csv, shift.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import subject_weights
from ppg_bp.conformal.pipeline import METHODS, band, intervals, read_predictions
from ppg_bp.conformal.shift import (
    SBP_SHARE_EDGES,
    domain_training_sample,
    domain_weights,
    recalibration,
    sbp_shares,
    sbp_weights,
    weighted_conformal,
)
from ppg_bp.evaluation.coverage import evaluate
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import load_split

MODELS = ("lightgbm", "cnn")
TARGETS = ("sbp", "dbp", "mbp")
COLUMNS = list(FEATURE_NAMES)
# columns of recalibration.csv (the shared function also returns those of the offset correction)
RECALIBRATION_COLUMNS = [
    "model",
    "method",
    "subjects",
    "repeat",
    "correction",
    "coverage",
    "width_median",
    "coverage_sbp_ge160",
    "coverage_sbp_lt90",
    "coverage_sbp_ge160_patient",
    "patients_sbp_ge160",
]


def add_strata(table, settings):
    strata = settings["strata"]
    median = table.groupby("subjectid")["sbp"].transform("median")
    table["stratum_true_sbp"] = band(table["sbp"], strata["sbp"])
    table["stratum_true_dbp"] = band(table["dbp"], strata["dbp"])
    table["stratum_pulse_rate"] = band(table["pulse_rate"], strata["pulse_rate"])
    table["stratum_sqi"] = table["sqi_tertile"]
    table["stratum_within_subject"] = band(table["sbp"] - median, strata["within_subject"])
    return table


def direct_application(cal, ext, settings, n_boot, seed):
    """6.1: the corrections from the VitalDB calibration patients, applied unchanged to the external data."""
    strata_columns = [c for c in ext.columns if c.startswith("stratum_")]
    rows = []
    for model in MODELS:
        for target in TARGETS:
            for level in settings["levels"]:
                for method in METHODS:
                    lower, upper, _ = intervals(cal, ext, target, model, method, level, settings)
                    wanted = (
                        strata_columns
                        if target == settings["primary_target"]
                        else [c for c in strata_columns if c == f"stratum_true_{target}"]
                    )
                    strata = {c.removeprefix("stratum_"): ext[c].to_numpy() for c in wanted}
                    result = evaluate(
                        ext[target], lower, upper, ext["subjectid"], strata, n_boot=n_boot, seed=seed, nominal=level
                    )
                    for name, value in (("method", method), ("level", level), ("target", target), ("model", model)):
                        result.insert(0, name, value)
                    rows.append(result)
    out = pd.concat(rows, ignore_index=True)
    out["reference_only"] = out["subjects"] < settings["min_subjects_per_stratum"]
    return out


def point_accuracy(ext):
    rows = []
    for model in MODELS:
        for target in TARGETS:
            error = ext[f"{target}_{model}_mean"] - ext[target]
            rows.append(
                {"model": model, "target": target, "mae": error.abs().mean(), "me": error.mean(), "sd": error.std()}
            )
    return pd.DataFrame(rows)


def weighted_rows(cal, ext, ratio_cal, ratio_ext, settings, n_boot, seed) -> pd.DataFrame:
    """6.2: weighted conformal prediction for every model and level."""
    rows = []
    for model in MODELS:
        for level in settings["levels"]:
            rows += weighted_conformal(
                cal,
                ext,
                (ratio_cal, ratio_ext),
                settings["primary_target"],
                model,
                level,
                settings["levels"][level],
                (n_boot, seed),
            )
    return pd.DataFrame(rows)


def label_shift(cal, ext, settings, auc):
    """6.4: shift of the BP distribution, and coverage with the target re-weighted to the source BP (split, 90%,
    LightGBM)."""
    target, level = settings["primary_target"], settings["primary_level"]
    rows = [{"quantity": "domain_classifier_auc", "value": auc}]
    for t in TARGETS:
        rows.append({"quantity": f"wasserstein_{t}_mmHg", "value": wasserstein_distance(cal[t], ext[t])})
        rows += [
            {"quantity": f"{t}_{name}_{stat}", "value": float(getattr(table[t], stat)())}
            for name, table in (("source", cal), ("target", ext))
            for stat in ("mean", "std")
        ]
    source_shares = sbp_shares(cal[target], cal["subjectid"])
    without_target = sbp_shares(ext[target], ext["subjectid"]) == 0
    for model in MODELS:
        lower, upper, _ = intervals(cal, ext, target, model, "split", level, settings)
        covered = ((ext[target] >= lower) & (ext[target] <= upper)).to_numpy()
        weights = sbp_weights(ext[target], ext["subjectid"], source_shares, SBP_SHARE_EDGES)
        rows.append(
            {
                "quantity": f"coverage_{model}_split_as_is",
                "value": float(np.average(covered, weights=subject_weights(ext["subjectid"]))),
            }
        )
        rows.append(
            {
                "quantity": f"coverage_{model}_split_reweighted_to_source_sbp",
                "value": float(np.average(covered, weights=weights)),
            }
        )
        rows.append(
            {
                "quantity": f"source_sbp_mass_without_target_support_{model}",
                "value": float(source_shares[without_target].sum()),
            }
        )
    return pd.DataFrame(rows)


def weighting_analysis(config, cal, ext, out_dir, dataset, settings, n_boot, seed) -> float:
    """6.2: weighted conformal prediction with a classifier that tells the datasets apart. Returns its AUC."""
    cal_features = load_split(config, "calibration")
    if not (
        cal_features["index"].tolist() == cal["index"].tolist()
        and cal_features["caseid"].tolist() == cal["caseid"].tolist()
    ):
        raise AssertionError("calibration features and estimates are not in the same order")
    ext_features = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed"] / dataset / "features.parquet")
    if not (
        ext_features["index"].tolist() == ext["index"].tolist()
        and ext_features["subjectid"].tolist() == ext["subjectid"].tolist()
    ):
        raise AssertionError("external features and estimates are not in the same order")
    classifier = settings["domain_classifier"]
    train = domain_training_sample(load_split(config, "train"), classifier["segments_per_patient"], seed)
    ratio_cal, ratio_ext, auc, gain = domain_weights(
        train, cal_features, ext_features, COLUMNS, seed, classifier["params"]
    )
    gain.to_csv(out_dir / "domain_classifier_importance.csv", header=["gain_share"], float_format="%.5f")
    weighted_rows(cal, ext, ratio_cal, ratio_ext, settings, n_boot, seed).to_csv(
        out_dir / "weighted.csv", index=False, float_format="%.6f"
    )
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic3wdb")
    parser.add_argument("--quick", action="store_true", help="small run for testing; results go to a quick/ folder")
    args = parser.parse_args()
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    out_dir = results / "external" / args.dataset / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 100 if args.quick else settings["bootstrap"]
    recalibration_settings = settings["recalibration"]
    repeats = 5 if args.quick else recalibration_settings["repeats"]

    cal = read_predictions(results, "calibration", cuts)
    ext = add_strata(read_predictions(results, args.dataset, cuts), settings)
    print(
        f"calibration (source) {cal['subjectid'].nunique()} patients {len(cal)} segments, "
        f"target {ext['subjectid'].nunique()} patients {len(ext)} segments",
        flush=True,
    )
    direct_application(cal, ext, settings, n_boot, seed).to_csv(
        out_dir / "coverage.csv", index=False, float_format="%.6f"
    )
    point_accuracy(ext).to_csv(out_dir / "point_accuracy.csv", index=False, float_format="%.6f")
    print("6.1 done", flush=True)
    recalibrated = recalibration(ext, settings, seed, repeats, tuple(recalibration_settings["patients"]), MODELS)
    recalibrated[RECALIBRATION_COLUMNS].to_csv(out_dir / "recalibration.csv", index=False, float_format="%.6f")
    print("6.3 done", flush=True)
    auc = weighting_analysis(config, cal, ext, out_dir, args.dataset, settings, n_boot, seed)
    print(f"6.2 done (AUC of the classifier {auc:.3f})", flush=True)
    label_shift(cal, ext, settings, auc).to_csv(out_dir / "shift.csv", index=False, float_format="%.6f")
    print("6.4 done", flush=True)


if __name__ == "__main__":
    main()
