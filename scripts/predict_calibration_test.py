"""Estimates of the mean and the quantiles for the calibration and test patients.

- LightGBM: a model of the mean and one per quantile (configs/conformal.yaml), trained on the same training segments
  as in model development (80% of the training patients, at most 400 segments each). The number of trees is set by
  early stopping on the remaining 20% of the training patients.
- CNN: the trained model (results/waveform_model/model.pt).
- Calibration and test patients are used neither for training nor for choosing the number of trees.

Only the estimates are saved here; coverage is computed by scripts/evaluate_conformal.py.

usage: uv run python -X utf8 scripts/predict_calibration_test.py [--quick]
output: results/conformal/predictions_calibration.parquet, predictions_test.parquet, lightgbm_trees.csv
log: results/logs/phase4_predict.log
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.cnn import load_checkpoint, model_from_checkpoint
from ppg_bp.models.dataset import (
    inner_validation_subjects,
    load_split,
    load_waveforms,
    sample_per_subject,
)
from ppg_bp.models.gbm import fit_lightgbm, quantile_name
from ppg_bp.runlog import RunLog

KEEP = ["caseid", "index", "subjectid", "sbp", "dbp", "mbp", "pulse_rate", "age", "sex", "sqi_template", "sqi_skewness"]
QUICK = "--quick" in sys.argv
LOG = PROJECT_ROOT / "results" / "logs" / ("phase4_predict_quick.log" if QUICK else "phase4_predict.log")
COLUMNS = list(FEATURE_NAMES)
log = RunLog(LOG)


@torch.no_grad()
def cnn_predict(config, model, mean, std, table, chunk_subjects=40):
    """Outputs on the original scale (segments x targets x output types). Patients are read in chunks to limit "
    "memory."""
    out = np.empty((len(table), len(mean), model.head.out_features // len(mean)), dtype=np.float32)
    subjects = table["subjectid"].unique()
    positions = pd.Series(np.arange(len(table)), index=table.index)
    for chunk in np.array_split(subjects, max(1, len(subjects) // chunk_subjects)):
        part = table[table["subjectid"].isin(chunk)]
        x = load_waveforms(config, part)
        pred = np.concatenate([model(torch.from_numpy(x[i : i + 2048])).numpy() for i in range(0, len(x), 2048)])
        out[positions[part.index].to_numpy()] = pred * std[None, :, None] + mean[None, :, None]
    return out


def split_data(config, feature_settings, seed):
    """Training segments, segments for early stopping, and the calibration and test tables."""
    train = load_split(config, "train")
    val_subjects = inner_validation_subjects(train["subjectid"], feature_settings["validation_fraction"], seed)
    is_val = train["subjectid"].isin(val_subjects)
    fit = sample_per_subject(train[~is_val], feature_settings["max_segments_per_subject"], seed)
    val = train[is_val]
    tables = {name: load_split(config, name) for name in ("calibration", "test")}
    if QUICK:  # for testing; the results go to a separate folder
        fit, val = fit.iloc[:20000], val.iloc[:5000]
        tables = {
            name: t[t["subjectid"].isin(t["subjectid"].unique()[:12])].reset_index(drop=True)
            for name, t in tables.items()
        }
        feature_settings["lightgbm"] |= {"n_estimators": 30}
    held_out = set(tables["calibration"]["subjectid"]) | set(tables["test"]["subjectid"])
    if held_out & set(train["subjectid"]):
        raise AssertionError("calibration or test patients are among the training patients")
    if set(tables["calibration"]["subjectid"]) & set(tables["test"]["subjectid"]):
        raise AssertionError("a patient is in both calibration and test")
    log(
        f"training {fit['subjectid'].nunique()} patients {len(fit)} segments, "
        f"early stopping {val['subjectid'].nunique()} patients, "
        f"calibration {tables['calibration']['subjectid'].nunique()} patients {len(tables['calibration'])} segments, "
        f"test {tables['test']['subjectid'].nunique()} patients {len(tables['test'])} segments"
    )
    return fit, val, tables


def add_lightgbm(outputs, fit, val, tables, feature_settings, quantiles, seed) -> pd.DataFrame:
    """Train the mean and quantile models, add their estimates to outputs, and return the numbers of trees."""
    trees = []
    for target in feature_settings["targets"]:
        jobs = [("mean", "regression", None)] + [(quantile_name(q), "quantile", q) for q in quantiles]
        for label, objective, alpha in jobs:
            start = time.time()
            model = fit_lightgbm(
                feature_settings["lightgbm"],
                objective,
                alpha,
                fit[COLUMNS],
                fit[target],
                val[COLUMNS],
                val[target],
                seed,
            )
            for name, table in tables.items():
                outputs[name][f"{target}_lightgbm_{label}"] = model.predict(table[COLUMNS]).astype(np.float32)
            minutes = (time.time() - start) / 60
            trees.append({"target": target, "output": label, "trees": model.best_iteration_, "minutes": minutes})
            log(f"LightGBM {target} {label}: {model.best_iteration_} trees, {minutes:.1f} min")
    return pd.DataFrame(trees)


def add_cnn(outputs, config, tables, targets, quantiles) -> None:
    """Add the estimates of the trained CNN to outputs."""
    saved = load_checkpoint(PROJECT_ROOT / config["paths"]["results"] / "waveform_model" / "model.pt")
    cnn_quantiles = saved["settings"]["quantiles"]
    if not (saved["targets"] == targets and [round(q, 4) for q in cnn_quantiles] == [round(q, 4) for q in quantiles]):
        raise AssertionError("the saved CNN has other targets or quantiles than the settings")
    model = model_from_checkpoint(saved)
    torch.set_num_threads(saved["settings"]["threads"])
    for name, table in tables.items():
        start = time.time()
        pred = cnn_predict(config, model, saved["mean"], saved["std"], table)
        for t, target in enumerate(targets):
            outputs[name][f"{target}_cnn_mean"] = pred[:, t, 0]
            for k, q in enumerate(cnn_quantiles):
                outputs[name][f"{target}_cnn_{quantile_name(q)}"] = pred[:, t, 1 + k]
        log(f"CNN {name}: {len(table)} segments, {(time.time() - start) / 60:.1f} min")


def main():
    config = load_config()
    settings = load_settings("conformal")
    feature_settings = load_settings("feature_models")
    seed = config["seed"]
    out_dir = PROJECT_ROOT / config["paths"]["results"] / "conformal" / ("quick" if QUICK else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== start")
    fit, val, tables = split_data(config, feature_settings, seed)
    outputs = {name: table[KEEP].copy() for name, table in tables.items()}
    trees = add_lightgbm(outputs, fit, val, tables, feature_settings, settings["quantiles"], seed)
    trees.to_csv(out_dir / "lightgbm_trees.csv", index=False, float_format="%.2f")
    add_cnn(outputs, config, tables, feature_settings["targets"], settings["quantiles"])
    for name, table in outputs.items():
        table.to_parquet(out_dir / f"predictions_{name}.parquet", index=False)
    log("=== done")


if __name__ == "__main__":
    main()
