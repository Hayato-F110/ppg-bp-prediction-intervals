"""Sensitivity analysis (analysis plan, Section 2.1): do the main results hold under other exclusion rules?

The model is not changed (the LightGBM models of the main analysis). Only the segments of the calibration and test
patients that enter the evaluation change:
    base        pulse rate 40-180 per minute, SD of beat-to-beat SBP at most 15 mmHg (main analysis)
    rate_wide   pulse rate 20-200 per minute
    sd10        SD of beat-to-beat SBP at most 10 mmHg
    sd20        SD of beat-to-beat SBP at most 20 mmHg
All other rules, such as a pulse pressure of at least 20 mmHg, are those of the main analysis.

input: data/processed/vitaldb_sensitivity/ (output of scripts/build_sensitivity_segments.py)
usage: uv run python -X utf8 scripts/evaluate_sensitivity.py
output: results/conformal/sensitivity.csv
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import band, intervals, sqi_tertile
from ppg_bp.data.segmentation import MAX_BEAT_SBP_SD, MIN_PULSE_PRESSURE, PULSE_RATE_RANGE
from ppg_bp.evaluation.coverage import evaluate
from ppg_bp.features.extract import FEATURE_NAMES, segment_features
from ppg_bp.models.gbm import load_models, predict_all

VARIANTS = {  # name: (pulse rate range per minute, maximum SD of beat-to-beat SBP in mmHg)
    "base": (PULSE_RATE_RANGE, MAX_BEAT_SBP_SD),
    "rate_wide": ((20.0, 200.0), MAX_BEAT_SBP_SD),
    "sd10": (PULSE_RATE_RANGE, 10.0),
    "sd20": (PULSE_RATE_RANGE, 20.0),
}
BANDS = ["<90", "90–<100", "100–<120", "120–<140", "140–<160", "≥160"]
META_COLUMNS = ("index", "pulse_rate", "sbp_sd", "sbp", "dbp", "mbp", "is_base", "is_new")


def read_resampled_cases(folder, fs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every re-cut case: the quantities the rules use (all segments), and the features of the segments that
    only the wider rules admit."""
    meta, new_features = [], []
    for path in sorted(folder.glob("*.npz")):
        case_id = int(path.stem)
        with np.load(path) as data:
            if len(data["index"]) == 0:
                continue
            table = pd.DataFrame({k: data[k] for k in META_COLUMNS})
            ppg_new = data["ppg_new"]
        table.insert(0, "caseid", case_id)
        meta.append(table)
        if len(ppg_new):
            values = np.array([list(segment_features(row, fs).values()) for row in ppg_new], dtype=np.float32)
            features = pd.DataFrame(values, columns=FEATURE_NAMES)
            features.insert(0, "index", table.loc[table["is_new"], "index"].to_numpy())
            features.insert(0, "caseid", case_id)
            new_features.append(features)
    new = pd.concat(new_features, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return pd.concat(meta, ignore_index=True), new


def segment_table(meta, new_features, split, models, conformal_dir, cuts) -> pd.DataFrame:
    """One row per segment with its estimates: those of the main analysis for segments it kept, and new ones for
    the segments that only the wider rules admit."""
    meta["subjectid"] = split.loc[meta["caseid"], "subjectid"].to_numpy()
    meta["part"] = split.loc[meta["caseid"], "split"].to_numpy()
    new_pred = pd.concat([new_features[["caseid", "index", "sqi_template"]], predict_all(models, new_features)], axis=1)
    old = pd.concat(
        [pd.read_parquet(conformal_dir / f"predictions_{name}.parquet") for name in ("calibration", "test")],
        ignore_index=True,
    )
    columns = ["caseid", "index", "sqi_template"] + [c for c in old.columns if "_lightgbm_" in c]
    pred = pd.concat([old[columns], new_pred[columns]], ignore_index=True)
    table = meta.merge(pred, on=["caseid", "index"], how="left", validate="one_to_one")
    in_any = table["is_base"] | table["is_new"]
    if not (table.loc[in_any, "sbp_lightgbm_mean"].notna().all()):
        raise AssertionError("some segments have no estimate")
    print(
        f"cases {table['caseid'].nunique()}, segments judged {len(table)}, "
        f"in the main analysis {int(table['is_base'].sum())}, newly admitted {int(table['is_new'].sum())}",
        flush=True,
    )
    table["sqi_tertile"] = sqi_tertile(table["sqi_template"], cuts)
    return table


def evaluate_variant(name, cal, test, settings, seed) -> list[pd.DataFrame]:
    rows = []
    for method in ("split", "cqr"):
        lower, upper, corrections = intervals(cal, test, "sbp", "lightgbm", method, settings["primary_level"], settings)
        strata = {"true_sbp": band(test["sbp"], settings["strata"]["sbp"])}
        result = evaluate(
            test["sbp"],
            lower,
            upper,
            test["subjectid"],
            strata,
            n_boot=settings["bootstrap"],
            seed=seed,
            nominal=settings["primary_level"],
        )
        result.insert(0, "correction", corrections[None])
        result.insert(0, "mae", float(np.abs(test["sbp_lightgbm_mean"] - test["sbp"]).mean()))
        result.insert(0, "cal_segments", len(cal))
        result.insert(0, "method", method)
        result.insert(0, "variant", name)
        rows.append(result)
    return rows


def print_summary(out: pd.DataFrame) -> None:
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    overall = out[out.variable == "all"][
        ["variant", "method", "segments", "mae", "coverage", "coverage_lo", "coverage_hi", "width_median"]
    ]
    print(overall.to_string(index=False))
    by_band = (
        out[(out.variable == "true_sbp") & (out.method == "split")]  # noqa: PD010 (one value per cell; pivot fails loudly on duplicates)
        .pivot(index="stratum", columns="variant", values="coverage")
        .reindex(BANDS)
    )
    print(by_band[list(VARIANTS)].to_string())


def main():
    config = load_config()
    settings = load_settings("conformal")
    feature_settings = load_settings("feature_models")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    results = PROJECT_ROOT / config["paths"]["results"] / "conformal"
    split = pd.read_csv(processed / "vitaldb" / "split.csv").set_index("caseid")
    models = load_models(results / "models", feature_settings["targets"], settings["quantiles"])

    meta, new_features = read_resampled_cases(processed / "vitaldb_sensitivity", config["segment"]["fs_hz"])
    table = segment_table(meta, new_features, split, models, results, cuts)
    pressure_ok = table["sbp"] - table["dbp"] >= MIN_PULSE_PRESSURE
    rows = []
    for name, ((low, high), max_sd) in VARIANTS.items():
        keep = (table["pulse_rate"] >= low) & (table["pulse_rate"] <= high) & (table["sbp_sd"] <= max_sd) & pressure_ok
        cal, test = table[keep & (table["part"] == "calibration")], table[keep & (table["part"] == "test")]
        rows += evaluate_variant(name, cal, test, settings, config["seed"])
        print(f"{name}: calibration {len(cal)} segments, test {len(test)} segments", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(results / "sensitivity.csv", index=False, float_format="%.6f")
    print_summary(out)


if __name__ == "__main__":
    main()
