"""LightGBM models survive saving and loading; the run log keeps every line."""

import threading

import numpy as np
import pandas as pd

from ppg_bp.models.gbm import COLUMNS, fit_lightgbm, load_models, output_labels, predict_all, save_models
from ppg_bp.runlog import RunLog

PARAMS = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "verbose": -1, "early_stopping_rounds": 5}
QUANTILES = (0.05, 0.95)


def make_features(n, seed):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(dict(zip(COLUMNS, rng.normal(size=(len(COLUMNS), n)), strict=True)))
    y = 120.0 + 10.0 * x[COLUMNS[0]] + rng.normal(size=n)
    return x, y


def test_saved_models_give_the_same_estimates(tmp_path):
    x, y = make_features(400, seed=0)
    models = {}
    for label, objective, alpha in [
        ("mean", "regression", None),
        ("q050", "quantile", 0.05),
        ("q950", "quantile", 0.95),
    ]:
        models[("sbp", label)] = fit_lightgbm(
            PARAMS, objective, alpha, x.iloc[:300], y.iloc[:300], x.iloc[300:], y.iloc[300:], 0
        ).booster_
    assert [label for _, label in models] == output_labels(QUANTILES)
    assert load_models(tmp_path, ["sbp"], QUANTILES) is None  # nothing saved yet
    save_models(models, tmp_path)
    loaded = load_models(tmp_path, ["sbp"], QUANTILES)
    if loaded is None:
        raise AssertionError("saved models were not found")
    before, after = predict_all(models, x), predict_all(loaded, x)
    assert list(after.columns) == ["sbp_lightgbm_mean", "sbp_lightgbm_q050", "sbp_lightgbm_q950"]
    np.testing.assert_allclose(after.to_numpy(), before.to_numpy(), rtol=1e-6)
    assert (after["sbp_lightgbm_q050"] < after["sbp_lightgbm_q950"]).mean() > 0.9


def test_run_log_keeps_whole_lines_from_threads(tmp_path):
    log = RunLog(tmp_path / "run.log")

    def write(k):
        for i in range(50):
            log(f"thread {k} line {i}")

    threads = [threading.Thread(target=write, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    assert all(line.split(" ", 1)[1].startswith("thread ") for line in lines)
