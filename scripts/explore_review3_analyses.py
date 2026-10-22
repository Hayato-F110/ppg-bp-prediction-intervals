"""Additional analyses requested by the third review (2026-10-15). All are exploratory (not in the analysis plan).

Nominal 90% SBP intervals, feature model (LightGBM), unless stated otherwise. Outputs go to results/review3/.

1. median_interval.csv: split conformal interval around the training median, which uses no PPG (the reference
   width quoted in the text). The median is that of the training segments used to fit the models.
2. cnn_seeds.csv (C1): validation accuracy of the waveform model trained with five other seeds.
3. pulse_pressure.csv (C2): share of segments with a pulse pressure above 100 mmHg (possible overshoot of the
   arterial line) by band of true SBP, and coverage when such segments are removed from calibration and test.
   excluded_by_rules.csv: true SBP of the segments of calibration and test cases that the pulse-rate,
   beat-variability and pulse-pressure rules excluded, against those kept.
4. label_shift.csv (C3): coverage on MIMIC-III re-weighted to the SBP distribution of VitalDB, with a 95% CI from
   a bootstrap over MIMIC-III patients; and the reverse, VitalDB test patients re-weighted to MIMIC-III.
5. resplit.csv (C4): the calibration and test patients pooled and split again at random 200 times, to show how
   much the coverage varies with the calibration patients (the main analysis bootstraps only the test patients).
6. offset_calibration.csv (C5): one reference reading per patient. The mean error (reference - estimate) of each
   patient's first 1, 6 or 30 segments (10 s, 1 min, 5 min) is added to the estimates of the patient's later
   segments; the interval is then calibrated on the corrected residuals of the calibration patients.
7. weighting_subsets.csv (C6): weighted conformal prediction with the domain classifier restricted to features less
   tied to the recording device, and with tighter clipping of the density ratio.
8. within_patient_correlation.csv (Section S2.5): rank correlation of APG times and ratios with SBP within each
   validation patient, also after removing the part explained by the pulse interval (partial rank correlation).
9. outlying_band.csv (Section S2.11): the band of points away from the main cloud in the DBP Bland-Altman plot of the
   feature model (validation patients): error above 25 mmHg and mean of estimate and reference below 55 mmHg.

usage: uv run python -X utf8 scripts/explore_review3_analyses.py [step ...]
"""

import sys

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import calibrate, split_interval, split_scores
from ppg_bp.conformal.pipeline import band, coverage_rows, intervals, read_predictions, reference_reading
from ppg_bp.conformal.shift import domain_training_sample, domain_weights, sbp_shares, sbp_weights, weighted_conformal
from ppg_bp.constants import EXTRA_SEEDS, MAIN_LEVEL, RESPLITS
from ppg_bp.evaluation.metrics import point_metrics
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject

LEVEL = MAIN_LEVEL
TARGET, MODEL = "sbp", "lightgbm"
HIGH_PULSE_PRESSURE = 100.0  # mmHg
RHYTHM = ["ibi_median", "ibi_sd", "ibi_rmssd", "ibi_cv"]
APG_FEATURES = ["t_ab", "t_ac", "t_ad", "t_ae", "t_af", "t_be", "t_de", "crest_time", "width50_rel", "d_a", "apg_index"]
MIN_SEGMENTS_FOR_CORRELATION, MIN_DISTINCT_VALUES = 50, 6  # per patient
SEED_RUNS = ("waveform_model", *(f"waveform_model_seed{seed}" for seed in EXTRA_SEEDS))  # table S8
BAND_ERROR, BAND_MEAN = 25.0, 55.0  # mmHg; outlying band of the DBP Bland-Altman plot


def median_interval(config, cal, test, settings, seed) -> pd.DataFrame:
    feature_settings = load_settings("feature_models")
    train = load_split(config, "train")
    held_out = inner_validation_subjects(train["subjectid"], feature_settings["validation_fraction"], seed)
    fit = sample_per_subject(
        train[~train["subjectid"].isin(held_out)], feature_settings["max_segments_per_subject"], seed
    )
    median = float(np.median(fit[TARGET]))
    q = calibrate(split_scores(cal[TARGET], np.full(len(cal), median)), cal["subjectid"], 1 - LEVEL)
    lower, upper = split_interval(np.full(len(test), median), q)
    out = coverage_rows(
        test[TARGET],
        lower,
        upper,
        test["subjectid"],
        settings["strata"]["sbp"],
        settings["bootstrap"],
        seed,
        predictor="training median",
    )
    out.insert(1, "median", median)
    return out


def cnn_seeds(results) -> pd.DataFrame:
    """Validation accuracy of each run, recomputed from its saved validation estimates at full precision
    (validation_metrics.csv keeps only 3 decimals)."""
    rows = []
    # the runs named explicitly, so that other folders (such as the single-thread check) do not enter the table
    for folder in (results / name for name in SEED_RUNS):
        table = pd.read_parquet(folder / "validation_predictions.parquet")
        rows += [
            {"run": folder.name, "target": target, **point_metrics(table[target], table[f"{target}_cnn_waveform"])}
            for target in ("sbp", "dbp", "mbp")
        ]
    return pd.DataFrame(rows)


def pulse_pressure(cal, test, settings, seed) -> pd.DataFrame:
    edges = settings["strata"]["sbp"]
    rows = []
    share = (test["sbp"] - test["dbp"] > HIGH_PULSE_PRESSURE).groupby(band(test["sbp"], edges)).mean()
    for stratum, value in share.items():
        rows.append(
            pd.DataFrame([{"analysis": "share_above_100", "variable": "true_sbp", "stratum": stratum, "value": value}])
        )
    keep_cal = (cal["sbp"] - cal["dbp"]) <= HIGH_PULSE_PRESSURE
    keep_test = (test["sbp"] - test["dbp"]) <= HIGH_PULSE_PRESSURE
    for method in ("split", "cqr"):
        c, t = cal[keep_cal], test[keep_test]
        lower, upper, _ = intervals(c, t, TARGET, MODEL, method, LEVEL, settings)
        rows.append(
            coverage_rows(
                t[TARGET],
                lower,
                upper,
                t["subjectid"],
                edges,
                settings["bootstrap"],
                seed,
                analysis="without_pulse_pressure_above_100",
                method=method,
            )
        )
    return pd.concat(rows, ignore_index=True)


def excluded_by_rules(processed, edges) -> pd.DataFrame:
    """Segments of the calibration and test cases (data/processed/vitaldb_sensitivity) kept or excluded by the
    pulse-rate, beat-variability and pulse-pressure rules of the main analysis (is_base of
    build_sensitivity_segments.py; segments excluded by the earlier rules are not stored there)."""
    tables = []
    for path in sorted((processed / "vitaldb_sensitivity").glob("*.npz")):
        with np.load(path) as data:
            if len(data["index"]) == 0:
                continue
            tables.append(pd.DataFrame({"sbp": data["sbp"], "is_base": data["is_base"]}))
    table = pd.concat(tables, ignore_index=True)
    table["band"] = band(table["sbp"], edges)
    out = table.groupby(["is_base", "band"]).size().rename("segments").reset_index()
    out["share"] = out["segments"] / out.groupby("is_base")["segments"].transform("sum")
    out["status"] = np.where(out["is_base"], "kept", "excluded by pulse-rate, beat-variability or pulse-pressure rules")
    return out.drop(columns="is_base")


def reweighted_coverage(covered, values, subjects, shares) -> float:
    """Coverage with each segment weighted so that the SBP distribution matches shares (sbp_shares() of the
    reference; 5-mmHg bins, equal weight per patient on both sides)."""
    return float(np.average(covered, weights=sbp_weights(values, subjects, shares)))


def label_shift(cal, test, ext, settings, seed) -> pd.DataFrame:
    n_boot = settings["bootstrap"]
    vitaldb_shares = sbp_shares(cal[TARGET], cal["subjectid"])  # the same in every bootstrap draw
    rows = []
    lower, upper, _ = intervals(cal, ext, TARGET, MODEL, "split", LEVEL, settings)
    covered = ((ext[TARGET] >= lower) & (ext[TARGET] <= upper)).to_numpy()
    y, subjects = ext[TARGET].to_numpy(), ext["subjectid"].to_numpy()
    point = reweighted_coverage(covered, y, subjects, vitaldb_shares)
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(subjects).apply(np.asarray)
    boots = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups.iloc[k] for k in chosen])
        names = np.concatenate([np.full(len(groups.iloc[k]), j) for j, k in enumerate(chosen)])
        boots.append(reweighted_coverage(covered[index], y[index], names, vitaldb_shares))
    rows.append(
        {
            "analysis": "MIMIC-III re-weighted to the SBP of VitalDB calibration patients",
            "coverage": point,
            "ci_low": np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5),
        }
    )
    lower, upper, _ = intervals(cal, test, TARGET, MODEL, "split", LEVEL, settings)
    covered_test = ((test[TARGET] >= lower) & (test[TARGET] <= upper)).to_numpy()
    rows.append(
        {
            "analysis": "VitalDB test re-weighted to the SBP of MIMIC-III",
            "coverage": reweighted_coverage(
                covered_test,
                test[TARGET].to_numpy(),
                test["subjectid"].to_numpy(),
                sbp_shares(ext[TARGET], ext["subjectid"]),
            ),
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    )
    return pd.DataFrame(rows)


def resplit(cal, test, settings, seed) -> pd.DataFrame:
    pooled = pd.concat([cal, test], ignore_index=True)
    subjects = pooled["subjectid"].unique()
    edges = settings["strata"]["sbp"]
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(RESPLITS):
        chosen = set(rng.choice(subjects, size=len(subjects) // 2, replace=False))
        in_cal = pooled["subjectid"].isin(chosen).to_numpy()
        c, t = pooled[in_cal], pooled[~in_cal]
        true_band = band(t[TARGET], edges)
        predicted_band = band(t[f"{TARGET}_{MODEL}_mean"], edges)
        for method in ("split", "cqr", "mondrian_pred"):
            lower, upper, _ = intervals(c, t, TARGET, MODEL, method, LEVEL, settings)
            covered = ((t[TARGET] >= lower) & (t[TARGET] <= upper)).to_numpy()
            row = {"repeat": repeat, "method": method}
            row["all"] = pd.Series(covered).groupby(t["subjectid"].to_numpy()).mean().mean()
            for name, labels in (("true", true_band), ("predicted", predicted_band)):
                for stratum in ("140–<160", "≥160", "<90"):
                    mask = labels == stratum
                    row[f"{name} {stratum}"] = (
                        pd.Series(covered[mask]).groupby(t["subjectid"].to_numpy()[mask]).mean().mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def offset_calibration(cal, test, ext, settings, seed) -> pd.DataFrame:
    """One reference reading per patient: the first segment (10 s), or the mean over the first 6 (1 min) or
    30 segments (5 min)."""
    return reference_reading(cal, {"VitalDB test": test, "MIMIC-III": ext}, settings, seed)


def weighting_subsets(config, cal, ext, settings, seed) -> pd.DataFrame:
    cal_features = load_split(config, "calibration")
    ext_features = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed"] / "mimic3wdb" / "features.parquet")
    # the same checks as weighting_analysis() of evaluate_external.py
    if not (
        cal_features["index"].tolist() == cal["index"].tolist()
        and cal_features["caseid"].tolist() == cal["caseid"].tolist()
    ):
        raise AssertionError("calibration features and estimates are not in the same order")
    if not (
        ext_features["index"].tolist() == ext["index"].tolist()
        and ext_features["subjectid"].tolist() == ext["subjectid"].tolist()
    ):
        raise AssertionError("external features and estimates are not in the same order")
    classifier = settings["domain_classifier"]  # the same training sample as weighting_analysis() of evaluate_external
    train = domain_training_sample(load_split(config, "train"), classifier["segments_per_patient"], seed)
    all_columns = list(FEATURE_NAMES)
    variants = {
        "all features (main analysis)": (all_columns, (1e-3, 1e3)),
        "without b/a and upslope time": ([c for c in all_columns if c not in ("b_a", "upslope_time")], (1e-3, 1e3)),
        "rhythm features only": (RHYTHM, (1e-3, 1e3)),
        "all features, ratio clipped to 0.1-10": (all_columns, (0.1, 10.0)),
    }
    rows = []
    for name, (columns, clip) in variants.items():
        ratio_cal, ratio_ext, auc, _ = domain_weights(
            train, cal_features, ext_features, columns, seed, classifier["params"], clip
        )
        rows += [
            {"variant": name, "auc": auc, **row}
            for row in weighted_conformal(
                cal,
                ext,
                (ratio_cal, ratio_ext),
                TARGET,
                MODEL,
                LEVEL,
                settings["levels"][LEVEL],
                (settings["bootstrap"], seed),
            )
        ]
        print(f"weighting: {name}: AUC {auc:.3f}", flush=True)
    return pd.DataFrame(rows)


def validation_patients(config) -> pd.DataFrame:
    """Segments (with features) of the validation patients held out from the training patients."""
    feature_settings = load_settings("feature_models")
    train = load_split(config, "train")
    held_out = inner_validation_subjects(train["subjectid"], feature_settings["validation_fraction"], config["seed"])
    return train[train["subjectid"].isin(held_out)]


def _rank_correlations(part: pd.DataFrame, feature: str) -> tuple[float, float]:
    """Rank correlation with SBP within one patient, and the same after removing the linear effect of the rank of
    the pulse interval from both ranks. NaN for patients with too few segments or values."""
    part = part[[feature, "sbp", "ibi_median"]].dropna()
    if len(part) < MIN_SEGMENTS_FOR_CORRELATION or part[feature].nunique() < MIN_DISTINCT_VALUES:
        return np.nan, np.nan
    ranks = part.rank()
    plain = np.corrcoef(ranks[feature], ranks["sbp"])[0, 1]

    def residual(values: pd.Series) -> np.ndarray:
        return values - np.polyval(np.polyfit(ranks["ibi_median"], values, 1), ranks["ibi_median"])

    return plain, np.corrcoef(residual(ranks[feature]), residual(ranks["sbp"]))[0, 1]


def within_patient_correlation(config) -> pd.DataFrame:
    validation = validation_patients(config)
    rows = []
    for feature in APG_FEATURES:
        values = np.array([_rank_correlations(part, feature) for _, part in validation.groupby("subjectid")])
        plain, partial = values[:, 0], values[:, 1]
        rows.append(
            {
                "feature": feature,
                "patients": int(np.isfinite(plain).sum()),
                "median": np.nanmedian(plain),
                "share_negative": np.mean(plain[np.isfinite(plain)] < 0),
                "median_without_pulse_interval": np.nanmedian(partial),
                "share_negative_without_pulse_interval": np.mean(partial[np.isfinite(partial)] < 0),
            }
        )
    return pd.DataFrame(rows)


def outlying_band(results) -> pd.DataFrame:
    table = pd.read_parquet(results / "feature_models" / "validation_predictions.parquet")
    estimate = table["dbp_lightgbm_ppg_only"]
    error, mean = estimate - table["dbp"], (estimate + table["dbp"]) / 2
    inside = (error > BAND_ERROR) & (mean < BAND_MEAN)
    part = table[inside]
    return pd.DataFrame(
        [
            {
                "segments": int(inside.sum()),
                "share": inside.mean(),
                "patients": part["subjectid"].nunique(),
                "largest_patient_segments": int(part["subjectid"].value_counts().iloc[0]),
                "dbp_min": part["dbp"].min(),
                "dbp_max": part["dbp"].max(),
                "sbp_median": part["sbp"].median(),
                "pulse_pressure_median": (part["sbp"] - part["dbp"]).median(),
                "pulse_pressure_median_all": (table["sbp"] - table["dbp"]).median(),
            }
        ]
    )


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    out = results / "review3"
    out.mkdir(parents=True, exist_ok=True)
    cal, test, ext = (read_predictions(results, name, cuts) for name in ("calibration", "test", "mimic3wdb"))

    steps = {
        "median_interval": lambda: median_interval(config, cal, test, settings, seed),
        "cnn_seeds": lambda: cnn_seeds(results),
        "pulse_pressure": lambda: pulse_pressure(cal, test, settings, seed),
        "excluded_by_rules": lambda: excluded_by_rules(processed, settings["strata"]["sbp"]),
        "label_shift": lambda: label_shift(cal, test, ext, settings, seed),
        "offset_calibration": lambda: offset_calibration(cal, test, ext, settings, seed),
        "weighting_subsets": lambda: weighting_subsets(config, cal, ext, settings, seed),
        "resplit": lambda: resplit(cal, test, settings, seed),
        "within_patient_correlation": lambda: within_patient_correlation(config),
        "outlying_band": lambda: outlying_band(results),
    }
    wanted = sys.argv[1:] or list(steps)  # optional: names of the steps to run
    for name, step in steps.items():
        if name not in wanted:
            continue
        step().to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
