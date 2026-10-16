"""Planned sensitivity analysis (analysis plan, Section 4.4): coverage by tertile of PPG skewness (sqi_skewness).

It was in the plan (approved 2026-10-13) but had been missed; it was run on 2026-10-14.
As for the main index (sqi_template), the cut points come from all training segments; evaluation is in the test
patients.
Models and intervals are those of the main analysis.

usage: uv run python -X utf8 scripts/evaluate_skewness_strata.py
output: results/conformal/skewness_strata.csv (with the cut points as columns)
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import intervals, read_predictions
from ppg_bp.evaluation.coverage import evaluate


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    results = PROJECT_ROOT / config["paths"]["results"]
    processed = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"

    features = pd.read_parquet(processed / "features.parquet", columns=["caseid", "sqi_skewness"])
    split = pd.read_csv(processed / "split.csv", usecols=["caseid", "split"])
    train = features.merge(split, on="caseid")
    train = train[train.split == "train"]["sqi_skewness"].dropna()
    lower, upper = np.quantile(train, [1 / 3, 2 / 3])
    print(f"skewness tertile cut points ({len(train):,} training segments): {lower:.4f}, {upper:.4f}", flush=True)

    cal, test = (read_predictions(results, name, cuts) for name in ("calibration", "test"))
    skew = test["sqi_skewness"].to_numpy()
    tertile = np.where(np.isnan(skew), None, np.select([skew < lower, skew < upper], ["low", "middle"], "high")).astype(
        object
    )

    rows = []
    for method in ("split", "cqr"):
        low, high, _ = intervals(cal, test, "sbp", "lightgbm", method, 0.9, settings)
        result = evaluate(
            test["sbp"],
            low,
            high,
            test["subjectid"],
            {"skewness_tertile": tertile},
            n_boot=settings["bootstrap"],
            seed=config["seed"],
            nominal=0.9,
        )
        result.insert(0, "method", method)
        rows.append(result[result.variable != "all"])
    out = pd.concat(rows, ignore_index=True).assign(cut_lower=lower, cut_upper=upper)
    out.to_csv(results / "conformal" / "skewness_strata.csv", index=False, float_format="%.6f")
    print(
        out[["method", "stratum", "subjects", "coverage", "coverage_lo", "coverage_hi", "width_median"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
