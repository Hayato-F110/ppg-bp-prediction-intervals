"""Fig. 1: the design of the study (split of the data, models, intervals, evaluation on the second dataset).

The numbers of patients and segments are counted from the data (as in Section 2 of the paper and Table S1).

usage: uv run python -X utf8 scripts/plot_study_design.py
input: data/processed/vitaldb/split.csv, results/external/mimic3wdb/predictions.parquet
output: results/figures/study_design (.png, .svg, .pdf and .eps)
"""

import matplotlib.pyplot as plt
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.evaluation.plots import (
    FLOW_NEUTRAL,
    FLOW_SOURCE,
    FLOW_TARGET,
    INK,
    INK_SECONDARY,
    MIN_TEXT,
    PAGE_WIDTH,
    apply_style,
    flow_arrow,
    flow_box,
    save_figure,
)


def vitaldb_row(ax, counts: dict):
    """Upper row: VitalDB, split by patient, models, intervals and evaluation."""
    vitaldb = flow_box(
        ax,
        0.00,
        0.58,
        0.19,
        0.98,
        "VitalDB",
        [
            "surgery",
            "PPG 500 Hz,",
            "analysed at 125 Hz",
            f"{counts['patients']:,} patients",
            f"{counts['segments']:,} segments",
        ],
        FLOW_SOURCE,
    )
    train = flow_box(ax, 0.25, 0.86, 0.46, 0.98, "Training", [f"{counts['train']:,} patients"], FLOW_SOURCE)
    calib = flow_box(ax, 0.25, 0.72, 0.46, 0.84, "Calibration", [f"{counts['calibration']:,} patients"], FLOW_SOURCE)
    test = flow_box(ax, 0.25, 0.56, 0.46, 0.68, "Test", [f"{counts['test']:,} patients"], FLOW_SOURCE)
    models = flow_box(
        ax,
        0.50,
        0.80,
        0.78,
        0.98,
        "Two PPG-only models",
        ["feature model (LightGBM)", "waveform model (CNN)"],
        FLOW_NEUTRAL,
    )
    intervals = flow_box(ax, 0.50, 0.655, 0.78, 0.775, "Prediction intervals", ["split, CQR, group-wise"], FLOW_NEUTRAL)
    flow_box(
        ax,
        0.82,
        0.56,
        1.00,
        0.98,
        "Evaluation",
        ["coverage and", "width per patient,", "by true SBP and", "subgroups;", "withholding", "uncertain estimates"],
        FLOW_SOURCE,
    )
    for target in (train, calib, test):
        flow_arrow(ax, vitaldb["right"], target["left"])
    flow_arrow(ax, train["right"], models["left"])
    flow_arrow(ax, calib["right"], intervals["left"])
    flow_arrow(ax, models["bottom"], intervals["top"])
    flow_arrow(ax, intervals["right"], (0.82, 0.72))
    flow_arrow(ax, test["right"], (0.82, 0.62))
    ax.text(
        0.22, 0.555, "split by patient", ha="center", va="top", fontsize=MIN_TEXT, color=INK_SECONDARY, style="italic"
    )


def mimic_row(ax, counts: dict):
    """Lower row: MIMIC-III and the four ways of applying the intervals. "Applied unchanged" uses the models and
    intervals of the upper row; this is written in the box rather than drawn, so that no lines cross."""
    mimic = flow_box(
        ax,
        0.00,
        -0.12,
        0.19,
        0.42,
        "MIMIC-III",
        [
            "intensive care,",
            "125 Hz",
            f"{counts['mimic_patients']:,} patients",
            f"{counts['mimic_segments']:,} segments",
        ],
        FLOW_TARGET,
    )
    unchanged = flow_box(
        ax, 0.25, 0.30, 0.77, 0.42, "Applied unchanged", ["models and intervals from VitalDB"], FLOW_TARGET
    )
    weighted = flow_box(
        ax,
        0.25,
        0.16,
        0.77,
        0.28,
        "Weighting with unlabelled data",
        ["VitalDB calibration data re-weighted"],
        FLOW_TARGET,
    )
    recal = flow_box(
        ax,
        0.25,
        0.02,
        0.77,
        0.14,
        "Recalibration with labelled data",
        ["5 to 100 MIMIC-III patients; models not retrained"],
        FLOW_TARGET,
    )
    reference = flow_box(
        ax,
        0.25,
        -0.12,
        0.77,
        0.00,
        "One reference reading per patient",
        ["offset from the first 10 s to 5 min; also without the PPG"],
        FLOW_TARGET,
    )
    eval_m = flow_box(
        ax,
        0.82,
        -0.12,
        1.00,
        0.42,
        "Evaluation",
        ["coverage and", "width per patient,", "by true SBP;", "withholding", "uncertain estimates"],
        FLOW_TARGET,
    )
    for target in (unchanged, weighted, recal, reference):
        flow_arrow(ax, mimic["right"], target["left"])
        flow_arrow(ax, target["right"], eval_m["left"])


def count(config: dict) -> dict:
    """Patients and kept segments of VitalDB (by part of the split) and of MIMIC-III."""
    split = pd.read_csv(PROJECT_ROOT / config["paths"]["processed"] / "vitaldb" / "split.csv")
    split = split[split["ok"] > 0]
    mimic = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["results"] / "external" / "mimic3wdb" / "predictions.parquet",
        columns=["subjectid"],
    )
    return {
        "patients": split["subjectid"].nunique(),
        "segments": int(split["ok"].sum()),
        **split.groupby("split")["subjectid"].nunique().to_dict(),
        "mimic_patients": mimic["subjectid"].nunique(),
        "mimic_segments": len(mimic),
    }


def main():
    config = load_config()
    counts = count(config)
    apply_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 4.1))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)  # no margins: use the full text width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.13, 1.07)
    ax.axis("off")
    # the two rows answer the two questions of the study; a heading names each
    heading = {"ha": "left", "fontsize": 9, "fontweight": "bold", "color": INK}
    ax.text(0.0, 1.065, "Within a dataset", va="top", **heading)
    ax.text(0.0, 0.445, "Across datasets", va="bottom", **heading)
    vitaldb_row(ax, counts)
    mimic_row(ax, counts)
    save_figure(fig, PROJECT_ROOT / config["paths"]["results"] / "figures" / "study_design.png")
    plt.close(fig)
    print("Saved the figure")


if __name__ == "__main__":
    main()
