"""Analyses requested by the first review (2026-10-14). All exploratory (not in the analysis plan).

No original result file is changed. Output: results/review/.

1. test_accuracy.csv: accuracy of the point estimate in the 627 test patients. The training-median reference is the
   median of the training segments, as in train_feature_models.py.
2. shrinkage.csv: coverage by true SBP band against a simple model. The estimate shrinks linearly
   (estimate = a + b x true + error) with normal errors of constant SD; a, b and SD come from the calibration
   patients (equal weight per patient). With the split conformal margin q, each segment is covered with probability
   Phi((y + q - m)/s) - Phi((y - q - m)/s), m = a + b y, averaged per patient as for the observed coverage.
   Coverage by predicted SBP band (what the intervals actually condition on) is given as well.
3. recalibration_shift.csv: MIMIC-III recalibration that also corrects the offset of the estimate: the mean of
   true - estimate in the chosen patients (equal weight per patient) is added to estimates and quantiles first.
   Patients are drawn in the same random order as evaluate_external.py; without the offset the results must match.
4. differences.csv: 95% CIs of the main differences between methods (paired bootstrap over patients).

usage: uv run python -X utf8 scripts/explore_review_analyses.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import (
    calibrate,
    cqr_interval,
    cqr_scores,
    split_interval,
    split_scores,
    subject_weights,
)
from ppg_bp.conformal.pipeline import band, intervals, quantile_column, read_predictions
from ppg_bp.constants import SBP_HIGH
from ppg_bp.evaluation.coverage import paired_difference
from ppg_bp.evaluation.metrics import point_metrics
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject

RECALIBRATION_SIZES = (5, 10, 20, 50, 100)  # as in evaluate_external.py
REPEATS = 200


def test_accuracy(config, test):
    settings = load_settings("feature_models")
    seed = config["seed"]
    train = load_split(config, "train")
    validation = inner_validation_subjects(train["subjectid"], settings["validation_fraction"], seed)
    fit = sample_per_subject(train[~train["subjectid"].isin(validation)], settings["max_segments_per_subject"], seed)
    rows = []
    for target in ("sbp", "dbp", "mbp"):
        predictions = {
            "Training median": np.full(len(test), np.median(fit[target])),
            "Feature model": test[f"{target}_lightgbm_mean"],
            "Waveform model": test[f"{target}_cnn_mean"],
        }
        for name, pred in predictions.items():
            rows.append(
                {
                    "target": target,
                    "predictor": name,
                    "patients": test["subjectid"].nunique(),
                    **point_metrics(test[target], pred),
                }
            )
    return pd.DataFrame(rows)


def patient_mean(values, subjects, labels, n_boot=2000, seed=0):
    """Per stratum: mean per patient, then mean over patients (as for coverage in this study), with a 95% CI (lo, hi)
    from a bootstrap over patients."""
    table = pd.DataFrame({"value": values, "subject": subjects, "label": labels}).dropna(subset=["label"])
    per = table.groupby(["label", "subject"])["value"].mean()
    rng = np.random.default_rng(seed)
    rows = {}
    for label, part in per.groupby("label"):
        v = part.to_numpy()
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        rows[label] = {
            "mean": v.mean(),
            "count": len(v),
            "lo": np.percentile(boots, 2.5),
            "hi": np.percentile(boots, 97.5),
        }
    return pd.DataFrame(rows).T


def shrinkage(cal, test, settings, level=0.9):
    target, model = "sbp", "lightgbm"
    y_cal, p_cal = cal[target].to_numpy(), cal[f"{target}_{model}_mean"].to_numpy()
    w = subject_weights(cal["subjectid"])
    # weighted least squares in closed form (np.polyfit was ill-conditioned with small weights and ignored them)
    mean_y, mean_p = np.average(y_cal, weights=w), np.average(p_cal, weights=w)
    slope = float(
        np.average((y_cal - mean_y) * (p_cal - mean_p), weights=w) / np.average((y_cal - mean_y) ** 2, weights=w)
    )
    intercept = float(mean_p - slope * mean_y)
    residual = p_cal - (intercept + slope * y_cal)
    sd = float(np.sqrt(np.average(residual**2, weights=w)))
    q = calibrate(split_scores(y_cal, p_cal), cal["subjectid"].to_numpy(), 1 - level)[None]

    y, p = test[target].to_numpy(), test[f"{target}_{model}_mean"].to_numpy()
    m = intercept + slope * y
    expected = norm.cdf((y + q - m) / sd) - norm.cdf((y - q - m) / sd)
    covered = (np.abs(y - p) <= q).astype(float)
    edges = settings["strata"]["sbp"]
    true_band, predicted_band = band(y, edges), band(p, edges)
    # coverage of group-wise calibration by predicted SBP band (mondrian_pred), by predicted band (nominal by design)
    lower, upper, _ = intervals(cal, test, target, model, "mondrian_pred", level, settings)
    covered_group = ((y >= lower) & (y <= upper)).astype(float)
    out = pd.concat(
        {
            "observed_by_true": patient_mean(covered, test["subjectid"], true_band)["mean"],
            "expected_by_true": patient_mean(expected, test["subjectid"], true_band)["mean"],
            "patients_true": patient_mean(covered, test["subjectid"], true_band)["count"],
            "observed_by_predicted": patient_mean(covered, test["subjectid"], predicted_band)["mean"],
            "patients_predicted": patient_mean(covered, test["subjectid"], predicted_band)["count"],
            "groupwise_by_predicted": patient_mean(covered_group, test["subjectid"], predicted_band)["mean"],
            "groupwise_by_predicted_lo": patient_mean(covered_group, test["subjectid"], predicted_band)["lo"],
            "groupwise_by_predicted_hi": patient_mean(covered_group, test["subjectid"], predicted_band)["hi"],
            "observed_by_predicted_lo": patient_mean(covered, test["subjectid"], predicted_band)["lo"],
            "observed_by_predicted_hi": patient_mean(covered, test["subjectid"], predicted_band)["hi"],
        },
        axis=1,
    )
    out.attrs = {"slope": slope, "intercept": intercept, "residual_sd": sd, "q": q}
    print(
        f"shrinkage model: estimate = {intercept:.1f} + {slope:.3f} x true, SD of error {sd:.1f}, split margin {q:.1f} "
        "mmHg",
        flush=True,
    )
    return out


def recalibration_shift(ext, settings, seed):
    """With and without the offset correction, in the same random order as evaluate_external (LightGBM only)."""
    target, level = settings["primary_target"], settings["primary_level"]
    q_lo, q_hi = settings["levels"][level]
    rng = np.random.default_rng(seed)
    subjects = ext["subjectid"].unique()
    code = pd.Series(np.arange(len(subjects)), index=subjects)[ext["subjectid"]].to_numpy()
    high = (ext[target] >= SBP_HIGH).to_numpy()
    model = "lightgbm"  # the first model in the original function, so the random draws match
    mean = ext[f"{target}_{model}_mean"].to_numpy()
    lo, hi = ext[quantile_column(target, model, q_lo)].to_numpy(), ext[quantile_column(target, model, q_hi)].to_numpy()
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    y = ext[target].to_numpy()
    rows = []
    for size in RECALIBRATION_SIZES:
        for repeat in range(REPEATS):
            chosen = np.zeros(len(subjects), dtype=bool)
            chosen[rng.choice(len(subjects), size=size, replace=False)] = True
            cal, out = chosen[code], ~chosen[code]
            residual = y[cal] - mean[cal]
            offset = float(np.average(residual, weights=subject_weights(code[cal])))
            for shifted in (False, True):
                d = offset if shifted else 0.0
                for method in ("split", "cqr"):
                    if method == "split":
                        q = calibrate(split_scores(y[cal], mean[cal] + d), code[cal], 1 - level)
                        lower, upper = split_interval(mean[out] + d, q)
                    else:
                        q = calibrate(cqr_scores(y[cal], lo[cal] + d, hi[cal] + d), code[cal], 1 - level)
                        lower, upper = cqr_interval(lo[out] + d, hi[out] + d, q)
                    covered = (y[out] >= lower) & (y[out] <= upper)
                    is_high = high[out]
                    per_high = pd.Series(covered[is_high]).groupby(code[out][is_high]).mean()
                    rows.append(
                        {
                            "method": method,
                            "shifted": shifted,
                            "subjects": size,
                            "repeat": repeat,
                            "offset": offset,
                            "coverage": pd.Series(covered).groupby(code[out]).mean().mean(),
                            "width_median": float(np.median(upper - lower)),
                            "coverage_sbp_ge160": covered[high[out]].mean() if high[out].any() else np.nan,
                            "coverage_sbp_ge160_patient": per_high.mean() if len(per_high) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def differences(cal, test, ext, settings, seed, level=0.9):
    target, model = "sbp", "lightgbm"
    rows = []

    def covered(table, method, source):
        lower, upper, _ = intervals(source, table, target, model, method, level, settings)
        return ((table[target] >= lower) & (table[target] <= upper)).to_numpy().astype(float)

    vit = {m: covered(test, m, cal) for m in ("split", "cqr", "mondrian_pred_sqi", "cqr_mondrian_pred_sqi")}
    high = (test[target] >= SBP_HIGH).to_numpy()
    for a, b in (("cqr", "split"), ("mondrian_pred_sqi", "split"), ("cqr_mondrian_pred_sqi", "cqr")):
        rows.append(
            {
                "dataset": "VitalDB test",
                "stratum": "SBP ≥ 160",
                "comparison": f"{a} − {b}",
                **paired_difference(
                    vit[a][high], vit[b][high], test["subjectid"].to_numpy()[high], settings["bootstrap"], seed
                ),
            }
        )
    mim = {m: covered(ext, m, cal) for m in ("split", "cqr")}
    rows.append(
        {
            "dataset": "MIMIC-III",
            "stratum": "all",
            "comparison": "cqr − split",
            **paired_difference(mim["cqr"], mim["split"], ext["subjectid"].to_numpy(), settings["bootstrap"], seed),
        }
    )
    rows.append(
        {
            "dataset": "MIMIC-III − VitalDB test",
            "stratum": "all",
            "comparison": "split (unpaired; see coverage.csv)",
            "patients": np.nan,
            "difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    )
    return pd.DataFrame(rows[:-1])


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    out_dir = results / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal, test, ext = (read_predictions(results, name, cuts) for name in ("calibration", "test", "mimic3wdb"))

    test_accuracy(config, test).to_csv(out_dir / "test_accuracy.csv", index=False, float_format="%.6f")
    print("1. accuracy in the test patients done", flush=True)
    shrink = shrinkage(cal, test, settings)
    shrink.to_csv(out_dir / "shrinkage.csv", float_format="%.6f")
    pd.Series(shrink.attrs).to_csv(out_dir / "shrinkage_model.csv", header=["value"], float_format="%.6f")
    print("2. shrinkage model done", flush=True)
    shift = recalibration_shift(ext, settings, seed)
    shift.to_csv(out_dir / "recalibration_shift.csv", index=False, float_format="%.6f")
    original = pd.read_csv(results / "external" / "mimic3wdb" / "recalibration.csv")
    original = original[original.model == "lightgbm"].set_index(["method", "subjects", "repeat"])["coverage"]
    mine = shift[~shift.shifted].set_index(["method", "subjects", "repeat"])["coverage"]
    gap = (mine - original.reindex(mine.index)).abs().max()
    print(f"3. offset correction done (largest difference from the original without correction {gap:.2e})", flush=True)
    differences(cal, test, ext, settings, seed).to_csv(out_dir / "differences.csv", index=False, float_format="%.6f")
    print("4. differences between methods done", flush=True)


if __name__ == "__main__":
    main()
