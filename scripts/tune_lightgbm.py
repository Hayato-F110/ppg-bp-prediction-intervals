"""Search LightGBM settings automatically, to see how much the validation accuracy depends on them.

The aim is to check that the settings used are not far from good ones; the models of the main analysis are not
changed (they use configs/feature_models.yaml, and settings are not swapped after seeing results). Only training
patients are used. Note: search and evaluation share the validation data, so the best value is optimistic.

usage: uv run python -X utf8 scripts/tune_lightgbm.py [--trials 40]
output: results/feature_models/tuning_trials.csv
log: results/logs/tune_lightgbm.log
"""

import argparse

import lightgbm as lgb
import numpy as np
import optuna

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject
from ppg_bp.runlog import RunLog

LOG = PROJECT_ROOT / load_config()["paths"]["results"] / "logs" / "tune_lightgbm.log"


log = RunLog(LOG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    config = load_config()
    settings = load_settings("feature_models")
    seed = config["seed"]
    LOG.parent.mkdir(parents=True, exist_ok=True)

    table = load_split(config, "train")
    is_val = table["subjectid"].isin(
        inner_validation_subjects(table["subjectid"], settings["validation_fraction"], seed)
    )
    fit = sample_per_subject(table[~is_val], settings["max_segments_per_subject"], seed)
    val = table[is_val]
    columns = list(FEATURE_NAMES)
    x_fit, y_fit, x_val, y_val = fit[columns], fit["sbp"], val[columns], val["sbp"]
    base = dict(settings["lightgbm"])
    rounds = base.pop("early_stopping_rounds")

    def run(params):
        model = lgb.LGBMRegressor(**(base | params), random_state=seed)
        model.fit(
            x_fit,
            y_fit,
            eval_X=x_val,
            eval_y=y_val,
            eval_metric="l1",
            callbacks=[lgb.early_stopping(rounds, verbose=False)],
        )
        return float(np.abs(model.predict(x_val) - y_val).mean()), model.best_iteration_

    baseline, trees = run({})
    log(f"=== start: SBP MAE with the current settings {baseline:.3f} ({trees} trees)")

    def objective(trial):
        params = {
            "objective": trial.suggest_categorical("objective", ["regression", "huber", "regression_l1"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [63, 255]),
        }
        mae, n_trees = run(params)
        trial.set_user_attr("trees", n_trees)
        log(f"trial {trial.number}: MAE {mae:.3f} ({n_trees} trees) {params}")
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=args.trials)
    trials = study.trials_dataframe()
    trials["baseline_mae"] = baseline
    trials.to_csv(PROJECT_ROOT / config["paths"]["results"] / "feature_models" / "tuning_trials.csv", index=False)
    log(
        f"=== done: best MAE {study.best_value:.3f} (current {baseline:.3f}, difference "
        f"{baseline - study.best_value:+.3f}) {study.best_params}"
    )


if __name__ == "__main__":
    main()
