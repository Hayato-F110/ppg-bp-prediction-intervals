"""Analyses requested by the second review (2026-10-14). All exploratory (not in the analysis plan).

No original result file is changed. Output: results/review2/. Feature model (LightGBM), SBP, nominal 90%.

1. extremes.csv: coverage by true SBP band, counted in different ways: as in the main analysis (patients with at
   least one segment in the band, equal weight), patients with at least 6 segments (1 min) in the band, by segment.
2. mbp_bands.csv: coverage by true MBP band (MBP is less affected by resonance of the arterial line).
3. extreme_reference.csv: pulse pressure at extreme SBP (a hint of resonance or damping), segments per patient.
4. slope_curve.csv: expected coverage under the shrinkage model for other slopes b.
5. recalibration_below90.csv: share of single MIMIC-III recalibrations with coverage below 90%.
6. pulse_groupwise.csv: group-wise calibration by bands of the pulse rate from the PPG.
7. deshrink.csv: split conformal after correcting the estimate for shrinkage.

usage: uv run python -X utf8 scripts/explore_review2_analyses.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import calibrate, split_interval, split_scores, subject_weights
from ppg_bp.conformal.pipeline import band, intervals, sqi_tertile
from ppg_bp.constants import MAIN_LEVEL, SBP_HIGH, SBP_LOW
from ppg_bp.evaluation.coverage import evaluate

MIN_SEGMENTS_IN_BAND = 6  # one minute of segments
HIGH_PULSE_PRESSURE = 100  # mmHg; possible overshoot of the arterial line

LEVEL = MAIN_LEVEL
SBP_EDGES = [90, 100, 120, 140, 160]
MBP_EDGES = [70, 80, 90, 100, 110]
PULSE_EDGES = [60, 80, 100]


def add_ppg_pulse(table, features, keys):
    """Add the pulse rate from the PPG (60 / median pulse interval), which is known at test time."""
    merged = table.merge(features[[*keys, "ibi_median"]], on=keys, how="left")
    merged["ppg_pulse_rate"] = 60.0 / merged["ibi_median"]
    return merged


def coverage_by(y, lower, upper, subjects, labels, n_boot, seed):
    return evaluate(y, lower, upper, subjects, {"band": labels}, n_boot=n_boot, seed=seed, nominal=LEVEL)


def extremes(test, cal, settings):
    rows = []
    y = test["sbp"].to_numpy()
    labels = band(y, SBP_EDGES)
    for method in ("split", "cqr"):
        lower, upper, _ = intervals(cal, test, "sbp", "lightgbm", method, LEVEL, settings)
        covered = (y >= lower) & (y <= upper)
        frame = pd.DataFrame({"subject": test["subjectid"].to_numpy(), "band": labels, "covered": covered})
        per = frame.groupby(["band", "subject"])["covered"].agg(["mean", "size"]).reset_index()
        for b, part in per.groupby("band"):
            many = part[part["size"] >= MIN_SEGMENTS_IN_BAND]
            rows.append(
                {
                    "method": method,
                    "band": b,
                    "patients": len(part),
                    "coverage_patient": part["mean"].mean(),
                    "patients_6plus": len(many),
                    "coverage_patient_6plus": many["mean"].mean() if len(many) else np.nan,
                    "segments": int(part["size"].sum()),
                    "coverage_segment": frame[frame.band == b]["covered"].mean(),
                    "segments_per_patient_median": part["size"].median(),
                    "segments_per_patient_q25": part["size"].quantile(0.25),
                    "segments_per_patient_q75": part["size"].quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def mbp_bands(test, cal, settings, seed):
    rows = []
    for method in ("split", "cqr"):
        lower, upper, _ = intervals(cal, test, "mbp", "lightgbm", method, LEVEL, settings)
        result = coverage_by(
            test["mbp"], lower, upper, test["subjectid"], band(test["mbp"], MBP_EDGES), settings["bootstrap"], seed
        )
        rows.append(result.assign(method=method))
    return pd.concat(rows, ignore_index=True)


def extreme_reference(test):
    pp = test["sbp"] - test["dbp"]
    groups = {
        "SBP ≥ 160": test["sbp"] >= SBP_HIGH,
        "SBP < 90": test["sbp"] < SBP_LOW,
        "SBP 90–159": (test["sbp"] >= SBP_LOW) & (test["sbp"] < SBP_HIGH),
    }
    rows = []
    for name, mask in groups.items():
        p = pp[mask]
        rows.append(
            {
                "group": name,
                "segments": int(mask.sum()),
                "patients": test.loc[mask, "subjectid"].nunique(),
                "pulse_pressure_median": p.median(),
                "pulse_pressure_q25": p.quantile(0.25),
                "pulse_pressure_q75": p.quantile(0.75),
                "share_pp_over_100": (p > HIGH_PULSE_PRESSURE).mean(),
                "dbp_median": test.loc[mask, "dbp"].median(),
                "pulse_rate_median": test.loc[mask, "pulse_rate"].median(),
            }
        )
    return pd.DataFrame(rows)


def slope_curve(cal, test):
    """Shrinkage model: estimate = a + b x true value + error (SD s). b is varied and a is set so that the mean
    estimate stays at its observed value (the estimates are on average below the true values; until 2026-10-15 the
    code centred them on the true mean instead, contrary to the description). The margin q is found under the same
    model from the calibration patients (equal weight per patient)."""
    y_cal, w = cal["sbp"].to_numpy(), subject_weights(cal["subjectid"])
    mean = np.average(y_cal, weights=w)
    offset = np.average(cal["sbp_lightgbm_mean"].to_numpy(), weights=w) - mean  # mean estimate - mean true value
    y_test, subjects = test["sbp"].to_numpy(), test["subjectid"].to_numpy()
    high = y_test >= SBP_HIGH
    grid = np.round(np.arange(0.3, 1.0001, 0.05), 2)
    rows = []
    for s in (9.6, 5.0):
        for b in grid:
            # Calibration score |y - estimate| with y - estimate = (1 - b)(y - mean) - offset - e;
            # its quantile is found numerically (equal weight per patient)
            shift = (1 - b) * (y_cal - mean) - offset
            target = LEVEL * (
                1 + 1 / len(np.unique(cal["subjectid"]))
            )  # quantile with one new patient added (as in the main analysis)

            def share_within(q, shift=shift, s=s):
                return np.average(norm.cdf((q - shift) / s) - norm.cdf((-q - shift) / s), weights=w)

            lo, hi = 0.0, 200.0  # bisection (trying all candidates took hours)
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if share_within(mid) >= target else (mid, hi)
            q = hi
            m = mean + offset + b * (y_test - mean)
            p = norm.cdf((y_test + q - m) / s) - norm.cdf((y_test - q - m) / s)
            frame = pd.DataFrame({"s": subjects, "p": p, "high": high})
            overall = frame.groupby("s")["p"].mean().mean()
            at_high = frame[frame.high].groupby("s")["p"].mean().mean()
            rows.append(
                {
                    "slope": b,
                    "residual_sd": s,
                    "margin": q,
                    "width": 2 * q,
                    "coverage_all": overall,
                    "coverage_sbp_ge160": at_high,
                }
            )
    return pd.DataFrame(rows)


def recalibration_below90(results):
    recal = pd.read_csv(results / "external" / "mimic3wdb" / "recalibration.csv")
    recal = recal[recal.model == "lightgbm"]
    return (
        recal.groupby(["method", "subjects"])
        .coverage.agg(share_below_90=lambda c: (c < MAIN_LEVEL).mean(), draws="size")
        .reset_index()
    )


def pulse_groupwise(cal, test, ext, settings, seed):
    """Group-wise calibration by bands of the PPG pulse rate; bands with fewer than 30 patients use the overall
    correction."""
    scores = split_scores(cal["sbp"], cal["sbp_lightgbm_mean"])
    groups_cal = band(cal["ppg_pulse_rate"], PULSE_EDGES)
    corrections = calibrate(
        scores, cal["subjectid"].to_numpy(), 1 - LEVEL, groups_cal, settings["min_subjects_per_stratum"]
    )
    rows = []
    for name, table in (("VitalDB test", test), ("MIMIC-III", ext)):
        groups = band(table["ppg_pulse_rate"], PULSE_EDGES)
        lower, upper = split_interval(table["sbp_lightgbm_mean"], corrections, groups)
        base_lower, base_upper, _ = intervals(cal, table, "sbp", "lightgbm", "split", LEVEL, settings)
        for method, (lo, hi) in (("split", (base_lower, base_upper)), ("group-wise (PPG pulse rate)", (lower, upper))):
            strata = {"true_sbp": band(table["sbp"], SBP_EDGES), "ppg_pulse_rate": groups}
            result = evaluate(
                table["sbp"], lo, hi, table["subjectid"], strata, n_boot=settings["bootstrap"], seed=seed, nominal=LEVEL
            )
            rows.append(result.assign(dataset=name, method=method))
    out = pd.concat(rows, ignore_index=True)
    out.attrs["corrections"] = corrections
    return out


def deshrink(cal, test, ext, settings, seed):
    """Correct the estimate by (estimate - a)/b (a and b from the calibration patients) and apply split conformal."""
    y, p, w = cal["sbp"].to_numpy(), cal["sbp_lightgbm_mean"].to_numpy(), subject_weights(cal["subjectid"])
    my, mp = np.average(y, weights=w), np.average(p, weights=w)
    b = np.average((y - my) * (p - mp), weights=w) / np.average((y - my) ** 2, weights=w)
    a = mp - b * my
    corrected_cal = (p - a) / b
    corrections = calibrate(split_scores(y, corrected_cal), cal["subjectid"].to_numpy(), 1 - LEVEL)
    rows = []
    for name, table in (("VitalDB test", test), ("MIMIC-III", ext)):
        corrected = (table["sbp_lightgbm_mean"].to_numpy() - a) / b
        lower, upper = split_interval(corrected, corrections)
        result = evaluate(
            table["sbp"],
            lower,
            upper,
            table["subjectid"],
            {"true_sbp": band(table["sbp"], SBP_EDGES)},
            n_boot=settings["bootstrap"],
            seed=seed,
            nominal=LEVEL,
        )
        rows.append(result.assign(dataset=name, a=a, b=b, margin=corrections[None]))
    return pd.concat(rows, ignore_index=True)


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    out = results / "review2"
    out.mkdir(parents=True, exist_ok=True)

    vit_features = pd.read_parquet(
        processed / "vitaldb" / "features.parquet", columns=["caseid", "index", "ibi_median"]
    )
    mim_features = pd.read_parquet(
        processed / "mimic3wdb" / "features.parquet", columns=["subjectid", "index", "ibi_median"]
    )
    cal = add_ppg_pulse(
        pd.read_parquet(results / "conformal" / "predictions_calibration.parquet"), vit_features, ["caseid", "index"]
    )
    test = add_ppg_pulse(
        pd.read_parquet(results / "conformal" / "predictions_test.parquet"), vit_features, ["caseid", "index"]
    )
    ext = add_ppg_pulse(
        pd.read_parquet(results / "external" / "mimic3wdb" / "predictions.parquet"),
        mim_features,
        ["subjectid", "index"],
    )
    for table in (cal, test, ext):
        table["sqi_tertile"] = sqi_tertile(table["sqi_template"], cuts)

    extremes(test, cal, settings).to_csv(out / "extremes.csv", index=False, float_format="%.6f")
    print("1. ways of counting done", flush=True)
    mbp_bands(test, cal, settings, seed).to_csv(out / "mbp_bands.csv", index=False, float_format="%.6f")
    print("2. MBP bands done", flush=True)
    extreme_reference(test).to_csv(out / "extreme_reference.csv", index=False, float_format="%.6f")
    print("3. pulse pressure at extreme SBP done", flush=True)
    slope_curve(cal, test).to_csv(out / "slope_curve.csv", index=False, float_format="%.6f")
    print("4. slope curve done", flush=True)
    recalibration_below90(results).to_csv(out / "recalibration_below90.csv", index=False, float_format="%.6f")
    print("5. recalibrations below 90% done", flush=True)
    pulse = pulse_groupwise(cal, test, ext, settings, seed)
    pulse.to_csv(out / "pulse_groupwise.csv", index=False, float_format="%.6f")
    print(
        "6. pulse-rate group-wise done (corrections: "
        + ", ".join(f"{k}: {v:.1f}" for k, v in pulse.attrs["corrections"].items())
        + "）",
        flush=True,
    )
    deshrink(cal, test, ext, settings, seed).to_csv(out / "deshrink.csv", index=False, float_format="%.6f")
    print("7. shrinkage correction done", flush=True)


if __name__ == "__main__":
    main()
