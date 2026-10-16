"""Exploratory (not in the analysis plan; added on 2026-10-16 after the results were known): can the intervals sort
a reading against the clinical threshold of 140 mmHg for SBP?

For each segment, the nominal 90% SBP interval (calibrated on the VitalDB calibration patients as in the main
analysis) lies wholly at or above 140 mmHg ("above"), wholly below it ("below"), or contains it ("undecided").
Shares are patient-level: first within each patient, then averaged over patients, as for coverage. They are
reported for all segments and separately for segments with a true SBP of 140 mmHg or higher and below 140 mmHg
(only patients with at least one such segment count).

usage: uv run python -X utf8 scripts/explore_clinical_threshold.py
output: results/conformal/exploratory_clinical_threshold.csv
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import intervals, read_predictions
from ppg_bp.constants import MAIN_LEVEL

THRESHOLD = 140.0  # mmHg; lower limit of hypertension for office SBP


def patient_share(flag: pd.Series, subjects: pd.Series) -> float:
    return float(flag.groupby(subjects).mean().mean())


def summarise(table: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> list[dict]:
    above = pd.Series(lower >= THRESHOLD, index=table.index)
    below = pd.Series(upper < THRESHOLD, index=table.index)
    undecided = ~(above | below)
    covered = pd.Series((lower <= table["sbp"]) & (table["sbp"] <= upper), index=table.index)
    rows = []
    for subset, mask in (
        ("all", np.ones(len(table), dtype=bool)),
        ("true_sbp_>=140", (table["sbp"] >= THRESHOLD).to_numpy()),
        ("true_sbp_<140", (table["sbp"] < THRESHOLD).to_numpy()),
    ):
        part = table[mask]
        subjects = part["subjectid"]
        rows.append(
            {
                "subset": subset,
                "subjects": int(subjects.nunique()),
                "segments": len(part),
                "coverage": patient_share(covered[mask], subjects),
                "above": patient_share(above[mask], subjects),
                "below": patient_share(below[mask], subjects),
                "undecided": patient_share(undecided[mask], subjects),
                "undecided_segment_level": float(undecided[mask].mean()),
            }
        )
    return rows


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    results = PROJECT_ROOT / config["paths"]["results"]
    cal = read_predictions(results, "calibration", cuts)
    rows = []
    for dataset in ("test", "mimic3wdb"):
        table = read_predictions(results, dataset, cuts)
        for model in ("lightgbm", "cnn"):
            for method in ("split", "cqr"):
                lower, upper, _ = intervals(cal, table, "sbp", model, method, MAIN_LEVEL, settings)
                rows.extend(
                    {"dataset": dataset, "model": model, "method": method, **row}
                    for row in summarise(table, np.asarray(lower), np.asarray(upper))
                )
    out = pd.DataFrame(rows)
    out.to_csv(results / "conformal" / "exploratory_clinical_threshold.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
