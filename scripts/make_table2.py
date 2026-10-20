"""Table 2 of the paper: patient-level coverage of nominal 90% SBP intervals by method, overall and at the extremes
of true SBP, with the median interval width (main split; VitalDB test patients and MIMIC-III).

usage: uv run python -X utf8 scripts/make_table2.py
input: results/conformal/coverage.csv, results/external/mimic3wdb/coverage.csv
output: results/tables/table2_methods.csv (with 95% confidence intervals), results/tables/table2_methods.md
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.constants import MAIN_LEVEL, SBP_HIGH
from ppg_bp.evaluation.plots import half_up

SBP_LOW_BAND, SBP_HIGH_BAND = "<90", f"≥{SBP_HIGH}"
METHODS = {
    "split": "Split conformal",
    "cqr": "CQR",
    "mondrian_pred": "Group-wise, predicted SBP",
    "mondrian_pred_sqi": "Group-wise, predicted SBP and signal quality",
    "cqr_mondrian_pred_sqi": "CQR and group-wise, predicted SBP and signal quality",
}
# (block of the table, coverage table, model, methods)
BLOCKS = (
    ("VitalDB, feature model", "vitaldb", "lightgbm", tuple(METHODS)),
    ("VitalDB, waveform model", "vitaldb", "cnn", ("split", "cqr")),
    ("MIMIC-III, feature model", "mimic3wdb", "lightgbm", ("split", "cqr")),
)


def coverage_row(table: pd.DataFrame, model: str, method: str, variable: str, stratum: str) -> pd.Series:
    row = table[
        (table.model == model)
        & (table.target == "sbp")
        & np.isclose(table.level, MAIN_LEVEL)
        & (table.method == method)
        & (table.variable == variable)
        & (table.stratum == stratum)
    ]
    if len(row) != 1:
        raise SystemExit(f"expected one row for {model} {method} {variable} {stratum}, found {len(row)}")
    return row.iloc[0]


def main() -> None:
    results = PROJECT_ROOT / load_config()["paths"]["results"]
    tables = {
        "vitaldb": pd.read_csv(results / "conformal" / "coverage.csv"),
        "mimic3wdb": pd.read_csv(results / "external" / "mimic3wdb" / "coverage.csv"),
    }
    rows = []
    for block, dataset, model, methods in BLOCKS:
        for method in methods:
            parts = {
                "all": coverage_row(tables[dataset], model, method, "all", "all"),
                "below_90": coverage_row(tables[dataset], model, method, "true_sbp", SBP_LOW_BAND),
                "at_or_above_160": coverage_row(tables[dataset], model, method, "true_sbp", SBP_HIGH_BAND),
            }
            row = {"block": block, "method": METHODS[method]}
            for name, part in parts.items():
                row |= {
                    f"coverage_{name}": part.coverage,
                    f"lo_{name}": part.coverage_lo,
                    f"hi_{name}": part.coverage_hi,
                }
            rows.append(row | {"width_median": parts["all"].width_median})
    table = pd.DataFrame(rows)
    out_dir = results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table2_methods.csv", index=False, float_format="%.6f")

    lines = ["| Model and method | All | < 90 | ≥ 160 | Median width (mmHg) |", "|---|---|---|---|---|"]
    for block, part in table.groupby("block", sort=False):
        lines.append(f"| *{block}* | | | | |")
        for r in part.itertuples():
            coverages = (r.coverage_all, r.coverage_below_90, r.coverage_at_or_above_160)
            values = [half_up(100 * c, 1) for c in coverages] + [half_up(r.width_median, 1)]
            lines.append(f"| {r.method} | " + " | ".join(values) + " |")
    (out_dir / "table2_methods.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
