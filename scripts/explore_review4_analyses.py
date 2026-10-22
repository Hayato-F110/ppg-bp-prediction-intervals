"""Additional analyses after the fourth (mock) review (2026-10-22). All are exploratory (not in the analysis plan).

Nominal 90% intervals, feature model (LightGBM), unless stated otherwise. Outputs go to results/review4/.

1. label_conditional.csv: conformal sets calibrated within ranges of the true SBP. A conformal predictor may take its
   groups from the label itself (Vovk 2012, label-conditional ICP, Proposition 3). For each range k of true SBP,
   margins are calibrated on the calibration segments whose true SBP lies in range k. A candidate value enters the set
   if it lies in range k and within the margins of range k around the estimate; the set is the union over the ranges.
   Coverage then holds within each range of true SBP; the cost is the size of the set, measured here.
2. groupwise_predicted_high.csv: why group-wise calibration by predicted SBP covered 82.8% at a predicted SBP of
   160 mmHg or higher in the main split. The main split with calibration and test swapped, and the margin of that
   group in the main split and in the 200 new splits of explore_review3_analyses.py (same seed, same draws).
3. carry_forward.csv: one reference reading per patient without the PPG. The mean reference over the patient's first
   1, 6 or 30 segments (10 s, 1 min, 5 min) is used as the estimate for all later segments, and split conformal
   intervals are calibrated on calibration patients treated the same way. Compare offset_calibration.csv of review3.
4. mimic_without_newborn.csv: coverage on MIMIC-III without the patient whose median pulse rate exceeded 130 per
   minute and median SBP was below 80 mmHg, values typical of a newborn (Section S1.12). Age was not checked: the
   clinical records of MIMIC-III are not used.
5. hypotension.csv: intervals for mean pressure (MBP) against the threshold of intraoperative hypotension, a true MBP
   below 65 mmHg. Coverage below 65 mmHg, and for each segment whether the interval lay wholly below 65 mmHg, wholly
   at or above it, or contained it. Shares are averaged first within each patient, as for coverage.
6. fit_accuracy.csv (added 2026-10-22, owner's item 44): SBP error of the feature model on the segments it was fitted
   on, on all segments of the fitting patients, on the validation patients and on the test patients, to show the gap
   between training and new patients (overfitting).
7. patient_coverage.csv (added 2026-10-22, owner's request for histograms): coverage of each patient, VitalDB test
   patients and MIMIC-III, split conformal and CQR, for the distribution in fig. S9.

usage: uv run python -X utf8 scripts/explore_review4_analyses.py [step ...]
"""

import sys

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import conformal_quantile, subject_weights
from ppg_bp.conformal.pipeline import band, coverage_rows, intervals, read_predictions, reference_reading
from ppg_bp.constants import FEW_PATIENTS, MAIN_LEVEL, RESPLITS
from ppg_bp.evaluation.coverage import threshold_sides
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject

LEVEL = MAIN_LEVEL
TARGET, MODEL = "sbp", "lightgbm"
SBP_THRESHOLD = 140.0  # mmHg, as in explore_clinical_threshold.py
MAP_THRESHOLD = 65.0  # mmHg, intraoperative hypotension
NEWBORN_PULSE_RATE, NEWBORN_SBP = 130.0, 80.0  # per minute, mmHg; medians per patient (Section S1.12)
GAP = 1e-6  # mmHg; tolerance when checking whether the pieces of a set touch


def per_patient_mean(values, subjects) -> float:
    return float(pd.Series(np.asarray(values, dtype=np.float64)).groupby(np.asarray(subjects)).mean().mean())


def label_margins(cal: pd.DataFrame, column: str, labels: list[str], edges, symmetric: bool) -> list:
    """Margins (below, above) of the estimate for each range of true SBP; error = true − estimate.

    As for group-wise calibration, a range with fewer than FEW_PATIENTS calibration patients uses the margins of
    all calibration segments.
    """
    true_band = band(cal[TARGET], edges)
    error = (cal[TARGET] - cal[column]).to_numpy()
    subjects = cal["subjectid"].to_numpy()

    def margins(mask: np.ndarray) -> tuple:
        weights = subject_weights(subjects[mask])
        if symmetric:
            half = conformal_quantile(np.abs(error[mask]), weights, 1 - LEVEL)
            return half, half
        return (
            conformal_quantile(-error[mask], weights, (1 - LEVEL) / 2),
            conformal_quantile(error[mask], weights, (1 - LEVEL) / 2),
        )

    overall = margins(np.ones(len(cal), dtype=bool))
    out = []
    for name in labels:
        mask = true_band == name
        out.append(margins(mask) if pd.unique(subjects[mask]).size >= FEW_PATIENTS else overall)
    return out


def label_conditional(cal, test, settings) -> pd.DataFrame:
    edges = settings["strata"]["sbp"]
    labels = list(band(np.array([edges[0] - 1, *edges]), edges))
    # ranges are [edge, next edge); the outer ranges are open (true SBP was limited to 20–300 mmHg by rule 2)
    lower_edge = np.array([-np.inf, *edges])
    upper_edge = np.array([*edges, np.inf])
    y, subjects, true_band = test[TARGET].to_numpy(), test["subjectid"].to_numpy(), band(test[TARGET], edges)
    rows = []
    for model in ("lightgbm", "cnn"):
        column = f"{TARGET}_{model}_mean"
        pred = test[column].to_numpy()
        for symmetric in (True, False):
            margin = np.array(label_margins(cal, column, labels, edges, symmetric))
            # one piece [lower, upper] per range; a piece is empty when lower > upper
            lower = np.maximum(lower_edge[None, :], pred[:, None] - margin[None, :, 0])
            upper = np.minimum(upper_edge[None, :], pred[:, None] + margin[None, :, 1])
            present = lower <= upper
            size = np.where(present, upper - lower, 0.0).sum(axis=1)
            covered = ((lower <= y[:, None]) & (y[:, None] <= upper)).any(axis=1)
            lowest = np.where(present, lower, np.inf).min(axis=1)
            highest = np.where(present, upper, -np.inf).max(axis=1)
            # pieces of neighbouring ranges touch at the edges; a gap means the set is not one interval
            gap = (highest - lowest) - size > GAP
            high = y >= SBP_THRESHOLD
            empty = ~present.any(axis=1)  # an empty set is neither wholly above nor wholly below 140 mmHg
            row = {
                "model": model,
                "margins": "symmetric" if symmetric else "asymmetric",
                "coverage_all": per_patient_mean(covered, subjects),
                # properties of the sets (size, empty, pieces) per segment; coverage and the sorting per patient
                "median_size": float(np.median(size)),
                "share_empty": float(empty.mean()),
                "share_not_one_interval": float(gap.mean()),
                "true_ge140_wholly_above": per_patient_mean(
                    ~empty[high] & (lowest[high] >= SBP_THRESHOLD), subjects[high]
                ),
                "true_ge140_wholly_below": per_patient_mean(
                    ~empty[high] & (highest[high] < SBP_THRESHOLD), subjects[high]
                ),
            }
            for k, name in enumerate(labels):
                mask = true_band == name
                row[f"margin_below_{name}"], row[f"margin_above_{name}"] = margin[k]
                row[f"coverage_{name}"] = per_patient_mean(covered[mask], subjects[mask])
                row[f"median_size_{name}"] = float(np.median(size[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def groupwise_predicted_high(cal, test, settings, seed) -> pd.DataFrame:
    edges = settings["strata"]["sbp"]
    column = f"{TARGET}_{MODEL}_mean"

    def one(c, t, split, repeat=None):
        lower, upper, corrections = intervals(c, t, TARGET, MODEL, "mondrian_pred", LEVEL, settings)
        covered = ((t[TARGET] >= lower) & (t[TARGET] <= upper)).to_numpy()
        mask = band(t[column], edges) == "≥160"
        return {
            "split": split,
            "repeat": repeat,
            "coverage_predicted_ge160": per_patient_mean(covered[mask], t["subjectid"].to_numpy()[mask]),
            "test_patients": t.loc[mask, "subjectid"].nunique(),
            "calibration_patients": c.loc[band(c[column], edges) == "≥160", "subjectid"].nunique(),
            "margin": corrections.get("≥160", corrections[None]),
            "overall_margin": corrections[None],
        }

    rows = [one(cal, test, "main"), one(test, cal, "main, calibration and test swapped")]
    # the same draws as resplit() of explore_review3_analyses.py
    pooled = pd.concat([cal, test], ignore_index=True)
    subjects = pooled["subjectid"].unique()
    rng = np.random.default_rng(seed)
    for repeat in range(RESPLITS):
        chosen = set(rng.choice(subjects, size=len(subjects) // 2, replace=False))
        in_cal = pooled["subjectid"].isin(chosen).to_numpy()
        rows.append(one(pooled[in_cal], pooled[~in_cal], "new split", repeat))
    return pd.DataFrame(rows)


def without_ppg(table: pd.DataFrame) -> pd.DataFrame:
    """Estimates of zero, so that with_offset() turns them into the mean reference of the first segments."""
    out = table.copy()
    for column in [c for c in table.columns if c.startswith(f"{TARGET}_{MODEL}_")]:
        out[column] = 0.0
    return out


def carry_forward(cal, test, ext, settings, seed) -> pd.DataFrame:
    """The reference reading alone, without the PPG (split conformal; compare offset_calibration of review3)."""
    tables = {"VitalDB test": test, "MIMIC-III": ext}
    return reference_reading(cal, tables, settings, seed, methods=("split",), prepare=without_ppg)


def newborn_like(ext: pd.DataFrame) -> list:
    medians = ext.groupby("subjectid")[["pulse_rate", TARGET]].median()
    chosen = medians[(medians["pulse_rate"] > NEWBORN_PULSE_RATE) & (medians[TARGET] < NEWBORN_SBP)]
    return chosen.index.tolist()


def mimic_without_newborn(cal, ext, settings, seed) -> pd.DataFrame:
    edges = settings["strata"]["sbp"]
    removed = newborn_like(ext)
    rows = []
    for name, table in (("all patients", ext), ("without newborn-like", ext[~ext["subjectid"].isin(removed)])):
        for method in ("split", "cqr"):
            lower, upper, _ = intervals(cal, table, TARGET, MODEL, method, LEVEL, settings)
            rows.append(
                coverage_rows(
                    table[TARGET],
                    lower,
                    upper,
                    table["subjectid"],
                    edges,
                    settings["bootstrap"],
                    seed,
                    patients=name,
                    removed=len(removed) if name != "all patients" else 0,
                    method=method,
                )
            )
    return pd.concat(rows, ignore_index=True)


def hypotension(cal, test, ext, settings) -> pd.DataFrame:
    rows = []
    for name, table in (("VitalDB test", test), ("MIMIC-III", ext)):
        y, subjects = table["mbp"].to_numpy(), table["subjectid"].to_numpy()
        low = y < MAP_THRESHOLD
        for model in ("lightgbm", "cnn"):
            pred = table[f"mbp_{model}_mean"].to_numpy()
            for method in ("split", "cqr"):
                lower, upper, _ = intervals(cal, table, "mbp", model, method, LEVEL, settings)
                lower, upper = np.asarray(lower), np.asarray(upper)
                covered = (lower <= y) & (y <= upper)
                above, below = threshold_sides(lower, upper, MAP_THRESHOLD)
                rows.append(
                    {
                        "dataset": name,
                        "model": model,
                        "method": method,
                        "patients_with_low": int(pd.unique(subjects[low]).size),
                        "share_segments_low": float(low.mean()),
                        "coverage_all": per_patient_mean(covered, subjects),
                        "coverage_low": per_patient_mean(covered[low], subjects[low]),
                        "median_width": float(np.median(upper - lower)),
                        # true MBP < 65: interval wholly below (hypotension confirmed), containing 65, wholly above
                        "low_wholly_below": per_patient_mean(below[low], subjects[low]),
                        "low_undecided": per_patient_mean(~below[low] & ~above[low], subjects[low]),
                        "low_wholly_above": per_patient_mean(above[low], subjects[low]),
                        # true MBP >= 65: interval wholly below 65 would be a false alarm
                        "not_low_wholly_below": per_patient_mean(below[~low], subjects[~low]),
                        "low_estimate_below": per_patient_mean(pred[low] < MAP_THRESHOLD, subjects[low]),
                        "not_low_estimate_below": per_patient_mean(pred[~low] < MAP_THRESHOLD, subjects[~low]),
                    }
                )
    return pd.DataFrame(rows)


def fit_accuracy(config) -> pd.DataFrame:
    """MAE of the SBP mean model on the data it was fitted on and on new patients (as train_feature_models.py)."""
    import lightgbm as lgb  # noqa: PLC0415  only this step needs it

    settings = load_settings("feature_models")
    seed = config["seed"]
    train = load_split(config, "train")
    validation = inner_validation_subjects(train["subjectid"], settings["validation_fraction"], seed)
    fitting = train[~train["subjectid"].isin(validation)]
    parts = {
        "fitted segments": sample_per_subject(fitting, settings["max_segments_per_subject"], seed),
        "all segments of the fitting patients": fitting,
        "validation patients": train[train["subjectid"].isin(validation)],
        "test patients": load_split(config, "test"),
    }
    model = lgb.Booster(
        model_file=str(PROJECT_ROOT / config["paths"]["results"] / "conformal" / "models" / "sbp_mean.txt")
    )
    rows = []
    for name, part in parts.items():
        error = model.predict(part[model.feature_name()]) - part[TARGET].to_numpy()
        rows.append(
            {"data": name, "patients": part["subjectid"].nunique(), "segments": len(part), "mae": np.abs(error).mean()}
        )
    return pd.DataFrame(rows)


def patient_coverage(cal, test, ext, settings) -> pd.DataFrame:
    """Coverage of each patient (share of the patient's segments covered), nominal 90%, SBP, feature model."""
    rows = []
    for dataset, table in (("VitalDB test", test), ("MIMIC-III", ext)):
        for method in ("split", "cqr"):
            lower, upper, _ = intervals(cal, table, TARGET, MODEL, method, LEVEL, settings)
            covered = (table[TARGET].to_numpy() >= lower) & (table[TARGET].to_numpy() <= upper)
            per = pd.Series(covered.astype(float)).groupby(table["subjectid"].to_numpy()).mean()
            rows += [{"dataset": dataset, "method": method, "subject": s, "coverage": c} for s, c in per.items()]
    return pd.DataFrame(rows)


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    out = results / "review4"
    out.mkdir(parents=True, exist_ok=True)
    cal, test, ext = (read_predictions(results, name, cuts) for name in ("calibration", "test", "mimic3wdb"))

    steps = {
        "label_conditional": lambda: label_conditional(cal, test, settings),
        "groupwise_predicted_high": lambda: groupwise_predicted_high(cal, test, settings, seed),
        "carry_forward": lambda: carry_forward(cal, test, ext, settings, seed),
        "mimic_without_newborn": lambda: mimic_without_newborn(cal, ext, settings, seed),
        "hypotension": lambda: hypotension(cal, test, ext, settings),
        "fit_accuracy": lambda: fit_accuracy(config),
        "patient_coverage": lambda: patient_coverage(cal, test, ext, settings),
    }
    wanted = sys.argv[1:] or list(steps)  # optional: names of the steps to run
    for name, step in steps.items():
        if name not in wanted:
            continue
        step().to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
