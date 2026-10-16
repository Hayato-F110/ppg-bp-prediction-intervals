"""Table 1 of the paper: the datasets and patients (VitalDB training, calibration and test; MIMIC-III).

A description, not a comparison; kept segments only (after exclusions). Age, sex and BMI per patient (first
operation for patients with several); pressures and pulse rate as mean +/- SD over segments. MIMIC-III clinical
records (age etc.) were not used, so those cells are left empty.

usage: uv run python -X utf8 scripts/make_characteristics.py
output: results/tables/characteristics.csv, results/tables/table_characteristics.md (copied into the paper)
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.constants import SBP_HIGH, SBP_LOW

SPLITS = {"train": "VitalDB: training", "calibration": "VitalDB: calibration", "test": "VitalDB: test"}


def mean_sd(values) -> str:
    return f"{values.mean():.1f} ± {values.std():.1f}"


def median_iqr(values) -> str:
    q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{q2:.0f} ({q1:.0f}–{q3:.0f})"


def describe(segments: pd.DataFrame, patients: pd.DataFrame | None) -> dict:
    return {
        "Patients": f"{segments.subjectid.nunique():,}",
        "Segments (10 s)": f"{len(segments):,}",
        "Age (years), median (IQR)": median_iqr(patients.age) if patients is not None else "not used",
        "Female, n (%)": f"{(patients.sex == 'F').sum():,} ({100 * (patients.sex == 'F').mean():.0f}%)"
        if patients is not None
        else "not used",
        "Body mass index (kg/m²), median (IQR)": median_iqr(patients.bmi.dropna())
        if patients is not None
        else "not used",
        "SBP (mmHg), mean ± SD": mean_sd(segments.sbp),
        "DBP (mmHg), mean ± SD": mean_sd(segments.dbp),
        "Segments with SBP < 90 / ≥ 160 mmHg (%)": f"{100 * (segments.sbp < SBP_LOW).mean():.1f} / "
        f"{100 * (segments.sbp >= SBP_HIGH).mean():.1f}",
        "Pulse rate (per minute), mean ± SD": mean_sd(segments.pulse_rate),
    }


def main():
    config = load_config()
    root = PROJECT_ROOT / config["paths"]["processed"]
    labels = pd.read_parquet(
        root / "vitaldb" / "segment_labels.parquet",
        columns=["caseid", "subjectid", "split", "sbp", "dbp", "pulse_rate"],
    )
    cases = pd.read_csv(root / "vitaldb" / "split.csv")
    # per patient: values of the first operation with at least one kept segment
    cases = cases[cases.caseid.isin(labels.caseid.unique())].sort_values("caseid").drop_duplicates("subjectid")

    columns = {}
    for split, name in SPLITS.items():
        columns[name] = describe(labels[labels.split == split], cases[cases.split == split])
    mimic = pd.read_parquet(root / "mimic3wdb" / "features.parquet", columns=["subjectid", "sbp", "dbp", "pulse_rate"])
    columns["MIMIC-III"] = describe(mimic, None)

    table = pd.DataFrame(columns)
    table.loc["Setting"] = ["surgery", "surgery", "surgery", "intensive care"]
    rate = "PPG sampling rate as recorded (Hz)"  # all analyses use 125 Hz
    table.loc[rate] = ["500", "500", "500", "125"]
    table = table.loc[["Setting", rate, *[i for i in table.index if i not in ("Setting", rate)]]]

    out_dir = PROJECT_ROOT / config["paths"]["results"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "characteristics.csv")
    lines = ["| | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {index} | " + " | ".join(row) + " |" for index, row in table.iterrows()]
    (PROJECT_ROOT / config["paths"]["results"] / "tables" / "table_characteristics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
