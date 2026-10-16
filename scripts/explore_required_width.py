"""Exploratory (not in the analysis plan): how wide would the interval need to be for nominal coverage by true BP?

The true BP band is unknown at test time, so this cannot be used in practice. It shows why group-wise
calibration cannot remove the bias: the width needed for nominal 90% in each band (split conformal calibrated on
the calibration segments of that band only), and the margins needed above and below for one-sided corrections.

usage: uv run python -X utf8 scripts/explore_required_width.py
output: results/conformal/exploratory_required_width.csv
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.core import conformal_quantile, subject_weights
from ppg_bp.conformal.pipeline import band
from ppg_bp.constants import MAIN_LEVEL

LEVEL = MAIN_LEVEL


def main():
    config = load_config()
    settings = load_settings("conformal")
    results = PROJECT_ROOT / config["paths"]["results"] / "conformal"
    cal, test = (
        pd.read_parquet(results / "predictions_calibration.parquet"),
        pd.read_parquet(results / "predictions_test.parquet"),
    )
    edges = settings["strata"]["sbp"]
    rows = []
    for model in ("lightgbm", "cnn"):
        column = f"sbp_{model}_mean"
        for table in (cal, test):
            table["band"] = band(table["sbp"], edges)
            table["error"] = table["sbp"] - table[column]  # positive: true value above the estimate
        for name in pd.unique(cal["band"]):
            c, t = cal[cal["band"] == name], test[test["band"] == name]
            weights = subject_weights(c["subjectid"])
            half = conformal_quantile(c["error"].abs().to_numpy(), weights, 1 - LEVEL)
            # margins above and below separately, each for 5% misses (asymmetric interval)
            up = conformal_quantile(c["error"].to_numpy(), weights, (1 - LEVEL) / 2)
            down = conformal_quantile(-c["error"].to_numpy(), weights, (1 - LEVEL) / 2)
            covered_sym = t["error"].abs() <= half
            covered_asym = (t["error"] <= up) & (-t["error"] <= down)
            rows.append(
                {
                    "model": model,
                    "true_sbp": name,
                    "cal_subjects": c["subjectid"].nunique(),
                    "test_subjects": t["subjectid"].nunique(),
                    "symmetric_width": 2 * half,
                    "test_coverage_symmetric": covered_sym.groupby(t["subjectid"]).mean().mean(),
                    "margin_above": up,
                    "margin_below": down,
                    "asymmetric_width": up + down,
                    "test_coverage_asymmetric": covered_asym.groupby(t["subjectid"]).mean().mean(),
                    "median_error": t["error"].median(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(results / "exploratory_required_width.csv", index=False, float_format="%.6f")
    order = ["<90", "90–<100", "100–<120", "120–<140", "140–<160", "≥160"]
    pd.set_option("display.width", 220, "display.float_format", "{:.1f}".format)
    for model in ("lightgbm", "cnn"):
        print(model)
        print(out[out.model == model].set_index("true_sbp").reindex(order).drop(columns="model").to_string())


if __name__ == "__main__":
    main()
