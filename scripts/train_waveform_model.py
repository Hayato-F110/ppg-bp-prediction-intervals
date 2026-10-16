"""Train the waveform model (one-dimensional CNN) and evaluate it on the validation patients within the training
patients.

The training segments and the validation patients are those of the feature model (same seed, same selection).
Test and calibration patients are not loaded.

Note: with several CPU threads, training is not exactly reproducible even with fixed seeds (validation MAE
differed by up to 0.9 mmHg between runs), because parallel sums are added in a varying order. With --deterministic
the model is trained on one thread, and two runs give identical results (checked on 2026-10-15 with --quick); this
takes about four times as long. The models of the paper were trained without --deterministic.

usage: uv run python -X utf8 scripts/train_waveform_model.py [--quick] [--model-seed N] [--deterministic]
output: results/waveform_model/model.pt, validation_metrics.csv, training_log.csv, validation_predictions.parquet
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.evaluation.metrics import point_metrics, subject_level_mae
from ppg_bp.models.cnn import WaveformNet, combined_loss, save_checkpoint
from ppg_bp.models.dataset import (
    inner_validation_subjects,
    load_labels,
    load_waveforms,
    sample_per_subject,
)


@torch.no_grad()
def predict(model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Outputs in standardised units (segments x targets x output types)."""
    model.eval()
    return np.concatenate(
        [model(torch.from_numpy(x[i : i + batch_size])).numpy() for i in range(0, len(x), batch_size)]
    )


def select_segments(config, settings, seed, quick):
    """Training segments, validation segments, and the subset of validation segments used for early stopping."""
    table = load_labels(config, "train")
    val_subjects = inner_validation_subjects(table["subjectid"], settings["validation_fraction"], seed)
    is_val = table["subjectid"].isin(val_subjects)
    fit = sample_per_subject(table[~is_val], settings["max_segments_per_subject"], seed)
    val = table[is_val]
    if quick:
        fit = fit[fit["subjectid"].isin(fit["subjectid"].unique()[:40])]
        val = val[val["subjectid"].isin(val["subjectid"].unique()[:10])]
        settings |= {"max_epochs": 2, "segments_per_epoch": 4000}
    if set(fit["subjectid"]) & set(val["subjectid"]):
        raise AssertionError("a patient is in both training and validation")
    stop = sample_per_subject(val, settings["early_stopping_per_subject"], seed)
    print(
        f"training: {fit['subjectid'].nunique()} patients {len(fit)} segments, "
        f"validation: {val['subjectid'].nunique()} patients {len(val)} segments ({len(stop)} for early stopping)",
        flush=True,
    )
    return fit, val, stop


def train(model, fit_data, stop_data, scale, settings, model_seed):
    """Train with early stopping on the SBP MAE of the stopping segments. Returns (best state, history).

    fit_data is (waveforms, standardised targets) as tensors, stop_data is (waveforms, table of the stopping
    segments) and scale is (mean, SD) of the training targets.
    """
    x_fit, y_fit_std = fit_data
    x_stop, stop = stop_data
    mean, std = scale
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    steps = settings["segments_per_epoch"] // settings["batch_size"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, settings["learning_rate"], total_steps=steps * settings["max_epochs"]
    )
    generator = torch.Generator().manual_seed(model_seed)
    best, best_state, since_best, history = np.inf, None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)[: steps * settings["batch_size"]]
        running = 0.0
        for batch in order.split(settings["batch_size"]):
            optimiser.zero_grad()
            loss = combined_loss(model(x_fit[batch]), y_fit_std[batch], model.quantiles)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += loss.item()
        stop_pred = predict(model, x_stop)[..., 0] * std + mean
        stop_mae = float(np.abs(stop_pred[:, 0] - stop["sbp"].to_numpy()).mean())
        minutes = (time.time() - start) / 60
        history.append({"epoch": epoch, "loss": running / steps, "stop_sbp_mae": stop_mae, "minutes": minutes})
        print(
            f"epoch {epoch}: loss {running / steps:.4f}, SBP MAE (stopping) {stop_mae:.2f}, {minutes:.1f} min",
            flush=True,
        )
        if stop_mae < best - 1e-3:
            best, since_best = stop_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= settings["patience"]:
                break
    if not (best_state is not None):
        raise AssertionError("the stopping MAE never improved")
    return best_state, history


def validate(model, config, val, targets, mean, std):
    """Estimates and accuracy on all validation segments (patients read in chunks to limit memory)."""
    pieces = []
    for chunk in np.array_split(val["subjectid"].unique(), max(1, val["subjectid"].nunique() // 40)):
        part = val[val["subjectid"].isin(chunk)]
        pieces.append(
            pd.DataFrame(
                predict(model, load_waveforms(config, part))[..., 0] * std + mean, index=part.index, columns=targets
            )
        )
    pred = pd.concat(pieces).loc[val.index]
    rows = []
    for target in targets:
        row = {"target": target, "features": "waveform", "model": "cnn"}
        row |= point_metrics(val[target], pred[target]) | subject_level_mae(val[target], pred[target], val["subjectid"])
        rows.append(row)
        print(f"{target} MAE {row['mae']:.2f}  ME {row['me']:+.2f} ± {row['sd']:.2f}  BHS {row['bhs']}", flush=True)
    return pred, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="for testing: a few patients, 2 epochs, nothing is saved")
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="seed of the initial weights and the training order (the segments do not change); "
        "the output goes to waveform_model_seed<N>",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="train on one CPU thread so that runs are exactly reproducible (about four times slower)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="name of the output folder under results/ (default: waveform_model, or waveform_model_seed<N>)",
    )
    args = parser.parse_args()

    config = load_config()
    settings = load_settings("waveform_model")
    if args.deterministic:
        settings["threads"] = 1
    seed = config["seed"]
    model_seed = seed if args.model_seed is None else args.model_seed
    torch.manual_seed(model_seed)
    torch.set_num_threads(settings["threads"])
    targets = settings["targets"]

    fit, val, stop = select_segments(config, settings, seed, args.quick)
    x_fit, x_stop = load_waveforms(config, fit), load_waveforms(config, stop)
    y_fit = fit[targets].to_numpy(np.float32)
    mean, std = y_fit.mean(axis=0), y_fit.std(axis=0)
    model = WaveformNet(len(targets), tuple(settings["quantiles"]), tuple(settings["channels"]))
    best_state, history = train(
        model,
        (torch.from_numpy(x_fit), torch.from_numpy((y_fit - mean) / std)),
        (x_stop, stop),
        (mean, std),
        settings,
        model_seed,
    )
    model.load_state_dict(best_state)
    pred, rows = validate(model, config, val, targets, mean, std)
    if args.quick:
        return

    folder = args.output or ("waveform_model" if args.model_seed is None else f"waveform_model_seed{args.model_seed}")
    out_dir = PROJECT_ROOT / config["paths"]["results"] / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_dir / "model.pt", best_state, mean, std, settings, targets)
    pd.DataFrame(rows).to_csv(out_dir / "validation_metrics.csv", index=False, float_format="%.3f")
    pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False, float_format="%.4f")
    predictions = val[["caseid", "index", "subjectid", *targets]].copy()
    for target in targets:
        predictions[f"{target}_cnn_waveform"] = pred[target].to_numpy(np.float32)
    predictions.to_parquet(out_dir / "validation_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
