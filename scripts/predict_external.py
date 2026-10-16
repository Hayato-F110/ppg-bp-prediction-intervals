"""Features of an external dataset, and estimates of the mean and quantiles by the models trained on VitalDB.

The models are not retrained. The LightGBM models are built as in scripts/predict_calibration_test.py and saved in
results/conformal/models/ (on the first run they are trained, saved and checked against the saved calibration
estimates). The CNN is the trained model.

External data: data/processed/<dataset>/segments/<subject>.npz (output of scripts/build_mimic_segments.py).
No raw values are printed; only counts and summaries.

usage: uv run python -X utf8 scripts/predict_external.py [--dataset mimic3wdb]
output: data/processed/<dataset>/features.parquet, results/external/<dataset>/predictions.parquet
log: results/logs/phase5_predict.log
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.features.extract import FEATURE_NAMES, segment_features
from ppg_bp.models.cnn import load_checkpoint, model_from_checkpoint
from ppg_bp.models.dataset import load_split
from ppg_bp.models.gbm import (
    load_models,
    predict_all,
    quantile_name,
    save_models,
    train_models,
)
from ppg_bp.runlog import RunLog

MAX_DIFFERENCE = 0.01  # mmHg; largest allowed difference from the saved calibration estimates

CONFIG = load_config()
FS = CONFIG["segment"]["fs_hz"]
LOG = PROJECT_ROOT / CONFIG["paths"]["results"] / "logs" / "phase5_predict.log"


log = RunLog(LOG)


def subject_features(path: str) -> pd.DataFrame:
    """Reference values and features of one subject."""
    path = Path(path)
    with np.load(path) as data:
        ppg = data["ppg"]
        table = pd.DataFrame({k: data[k] for k in ("index", "sbp", "dbp", "mbp", "pulse_rate")})
    values = np.array([list(segment_features(row, FS).values()) for row in ppg], dtype=np.float32)
    table = pd.concat([table, pd.DataFrame(values, columns=FEATURE_NAMES)], axis=1)
    table.insert(0, "subjectid", path.stem)
    return table


@torch.no_grad()
def cnn_outputs(folder: Path, table: pd.DataFrame, saved) -> np.ndarray:
    targets, quantiles = saved["targets"], saved["settings"]["quantiles"]
    model = model_from_checkpoint(saved)
    mean, std = saved["mean"], saved["std"]
    out = np.empty((len(table), len(targets), 1 + len(quantiles)), dtype=np.float32)
    for subject, rows in table.groupby("subjectid", sort=False).indices.items():
        with np.load(folder / f"{subject}.npz") as data:
            ppg, index = data["ppg"], data["index"]
        if not np.array_equal(index, table["index"].to_numpy()[rows]):
            raise AssertionError("segments of the waveform file and the table do not match")
        out[rows] = model(torch.from_numpy(ppg.astype(np.float32))).numpy() * std[None, :, None] + mean[None, :, None]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic3wdb")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    quantiles = load_settings("conformal")["quantiles"]
    feature_settings = load_settings("feature_models")
    targets = feature_settings["targets"]
    results = PROJECT_ROOT / CONFIG["paths"]["results"]
    data_dir = PROJECT_ROOT / CONFIG["paths"]["processed"] / args.dataset
    out_dir = results / "external" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"=== start: {args.dataset}")

    files = sorted(str(p) for p in (data_dir / "segments").glob("*.npz"))
    with Pool(args.workers) as pool:
        table = pd.concat(pool.map(subject_features, files, chunksize=4), ignore_index=True)
    table = table.replace([np.inf, -np.inf], np.nan)
    table.to_parquet(data_dir / "features.parquet", index=False)
    log(f"features: {table['subjectid'].nunique()} subjects, {len(table)} segments")

    model_dir = results / "conformal" / "models"
    models = load_models(model_dir, targets, quantiles)
    if models is None:
        log("training and saving the LightGBM models (as for the calibration and test estimates)")
        models = train_models(CONFIG, feature_settings, quantiles, log)
        save_models(models, model_dir)
        models = load_models(model_dir, targets, quantiles)
    # check against the saved calibration estimates (the same models as in the main analysis)
    calibration = load_split(CONFIG, "calibration").iloc[:50000]
    saved = pd.read_parquet(results / "conformal" / "predictions_calibration.parquet").iloc[:50000]
    if not (
        calibration["index"].tolist() == saved["index"].tolist()
        and calibration["caseid"].tolist() == saved["caseid"].tolist()
    ):
        raise AssertionError("calibration features and saved estimates are not in the same order")
    again = predict_all(models, calibration)
    worst = max(float(np.abs(again[c].to_numpy() - saved[c].to_numpy()).max()) for c in again.columns)
    log(f"largest difference from the saved estimates: {worst:.6f} mmHg")
    if worst >= MAX_DIFFERENCE:
        raise AssertionError("the models differ from those of the main analysis")

    out = table[["subjectid", "index", "sbp", "dbp", "mbp", "pulse_rate", "sqi_template", "sqi_skewness"]].copy()
    out = pd.concat([out, predict_all(models, table)], axis=1)
    saved_cnn = load_checkpoint(results / "waveform_model" / "model.pt")
    torch.set_num_threads(2)
    pred = cnn_outputs(data_dir / "segments", table, saved_cnn)
    for t, target in enumerate(saved_cnn["targets"]):
        out[f"{target}_cnn_mean"] = pred[:, t, 0]
        for k, q in enumerate(saved_cnn["settings"]["quantiles"]):
            out[f"{target}_cnn_{quantile_name(q)}"] = pred[:, t, 1 + k]
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    log(f"=== done: estimates of {len(out)} segments saved")


if __name__ == "__main__":
    main()
