"""Train the feature models (median, linear regression, LightGBM) and evaluate them on the validation patients.

Test and calibration patients are not loaded. Versions with and without age and sex are both reported.

usage: uv run python -X utf8 scripts/train_feature_models.py [--quick]
output: results/feature_models/validation_metrics.csv, feature_importance.csv, validation_predictions.parquet
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.evaluation.metrics import point_metrics, subject_level_mae
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject

DEMOGRAPHICS = ["age", "sex_male"]


def fit_predict(name, params, x_fit, y_fit, x_val, y_val, seed):
    """Fit a model; return its validation estimates and, if any, the feature importance."""
    if name == "median":
        return np.full(len(x_val), np.median(y_fit)), None
    if name == "linear":
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=1.0))
        return model.fit(x_fit, y_fit).predict(x_val), None
    params = dict(params)
    rounds = params.pop("early_stopping_rounds")
    model = lgb.LGBMRegressor(**params, random_state=seed)
    model.fit(
        x_fit,
        y_fit,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric="l1",
        callbacks=[lgb.early_stopping(rounds, verbose=False)],
    )
    gain = pd.Series(model.booster_.feature_importance("gain"), index=x_fit.columns)
    return model.predict(x_val), gain / gain.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="for testing: fewer trees, nothing saved")
    args = parser.parse_args()
    config = load_config()
    settings = load_settings("feature_models")
    seed = config["seed"]
    if args.quick:
        settings["lightgbm"] |= {"n_estimators": 150, "n_jobs": 2}

    table = load_split(config, "train")
    table["sex_male"] = (table["sex"] == "M").astype(float)
    val_subjects = inner_validation_subjects(table["subjectid"], settings["validation_fraction"], seed)
    is_val = table["subjectid"].isin(val_subjects)
    fit = sample_per_subject(table[~is_val], settings["max_segments_per_subject"], seed)
    val = table[is_val]
    if set(fit["subjectid"]) & set(val["subjectid"]):
        raise AssertionError("a patient is in both training and validation")
    print(
        f"training: {fit['subjectid'].nunique()} patients {len(fit)} segments, validation: "
        f"{val['subjectid'].nunique()} patients {len(val)} segments"
    )

    rows, importances = [], {}
    predictions = val[["caseid", "index", "subjectid", *settings["targets"]]].copy()
    feature_sets = {"ppg_only": list(FEATURE_NAMES), "ppg_demographics": list(FEATURE_NAMES) + DEMOGRAPHICS}
    for target in settings["targets"]:
        for set_name, columns in feature_sets.items():
            for model in ("median", "linear", "lightgbm"):
                if model == "median" and set_name != "ppg_only":
                    continue
                pred, gain = fit_predict(
                    model, settings["lightgbm"], fit[columns], fit[target], val[columns], val[target], seed
                )
                row = {"target": target, "features": set_name if model != "median" else "none", "model": model}
                row |= point_metrics(val[target], pred) | subject_level_mae(val[target], pred, val["subjectid"])
                rows.append(row)
                predictions[f"{target}_{model}_{row['features']}"] = np.asarray(pred, dtype=np.float32)
                if gain is not None:
                    importances[(target, set_name)] = gain
                print(
                    f"{target} {row['features']:17s} {model:9s} MAE {row['mae']:.2f}  ME {row['me']:+.2f} ± "
                    f"{row['sd']:.2f}  BHS {row['bhs']}",
                    flush=True,
                )

    if args.quick:
        return
    out_dir = PROJECT_ROOT / config["paths"]["results"] / "feature_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "validation_metrics.csv", index=False, float_format="%.3f")
    pd.DataFrame(importances).to_csv(out_dir / "feature_importance.csv", float_format="%.5f")
    predictions.to_parquet(out_dir / "validation_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
