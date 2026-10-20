"""Fig. 1: the design of the study (split of the data, models, intervals, evaluation on the second dataset).

The numbers are those of Section 2 of the paper (patients and segments as in Table S1); no result file is read.

usage: uv run python -X utf8 scripts/plot_study_design.py
output: results/figures/study_design (.png and .svg)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.evaluation.plots import (
    AXIS,
    INK,
    INK_SECONDARY,
    MIN_TEXT,
    PAGE_WIDTH,
    apply_style,
    save_figure,
)

VITALDB, MIMIC, NEUTRAL = "#e3eefb", "#fbe7dd", "#f1f0ec"  # source dataset, second dataset, shared


def box(ax, x0, y0, x1, y1, title, lines, color):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=color,
            edgecolor=AXIS,
            linewidth=0.8,
        )
    )
    ax.text((x0 + x1) / 2, y1 - 0.02, title, ha="center", va="top", fontsize=MIN_TEXT, fontweight="bold", color=INK)
    ax.text(
        (x0 + x1) / 2,
        y1 - 0.075,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=MIN_TEXT,
        color=INK_SECONDARY,
        linespacing=1.3,
    )
    return {
        "left": (x0, (y0 + y1) / 2),
        "right": (x1, (y0 + y1) / 2),
        "top": ((x0 + x1) / 2, y1),
        "bottom": ((x0 + x1) / 2, y0),
    }


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK_SECONDARY,
            "linewidth": 0.8,
            "shrinkA": 1,
            "shrinkB": 1,
            "mutation_scale": 7,
        },
    )


def vitaldb_row(ax):
    """Upper row: VitalDB, split by patient, models, intervals and evaluation."""
    vitaldb = box(
        ax,
        0.00,
        0.58,
        0.19,
        0.98,
        "VitalDB",
        ["surgery", "PPG 500 Hz,", "analysed at 125 Hz", "3,134 patients", "3,653,817 segments"],
        VITALDB,
    )
    train = box(ax, 0.25, 0.86, 0.46, 0.98, "Training", ["1,880 patients"], VITALDB)
    calib = box(ax, 0.25, 0.72, 0.46, 0.84, "Calibration", ["627 patients"], VITALDB)
    test = box(ax, 0.25, 0.56, 0.46, 0.68, "Test", ["627 patients"], VITALDB)
    models = box(
        ax, 0.50, 0.80, 0.78, 0.98, "Two PPG-only models", ["feature model (LightGBM)", "waveform model (CNN)"], NEUTRAL
    )
    intervals = box(ax, 0.50, 0.655, 0.78, 0.775, "Prediction intervals", ["split, CQR, group-wise"], NEUTRAL)
    box(
        ax,
        0.82,
        0.58,
        1.00,
        0.90,
        "Evaluation",
        ["coverage and", "width per patient,", "by true SBP and", "subgroups"],
        VITALDB,
    )
    for target in (train, calib, test):
        arrow(ax, vitaldb["right"], target["left"])
    arrow(ax, train["right"], models["left"])
    arrow(ax, calib["right"], intervals["left"])
    arrow(ax, models["bottom"], intervals["top"])
    arrow(ax, intervals["right"], (0.82, 0.72))
    arrow(ax, test["right"], (0.82, 0.62))
    ax.text(
        0.22, 0.555, "split by patient", ha="center", va="top", fontsize=MIN_TEXT, color=INK_SECONDARY, style="italic"
    )


def mimic_row(ax):
    """Lower row: MIMIC-III and the three ways of applying the intervals. "Applied unchanged" uses the models and
    intervals of the upper row; this is written in the box rather than drawn, so that no lines cross."""
    mimic = box(
        ax, 0.00, 0.02, 0.19, 0.42, "MIMIC-III", ["intensive care,", "125 Hz", "416 patients", "53,928 segments"], MIMIC
    )
    unchanged = box(ax, 0.25, 0.30, 0.77, 0.42, "Applied unchanged", ["models and intervals from VitalDB"], MIMIC)
    weighted = box(
        ax, 0.25, 0.16, 0.77, 0.28, "Weighting with unlabelled data", ["VitalDB calibration data re-weighted"], MIMIC
    )
    recal = box(
        ax,
        0.25,
        0.02,
        0.77,
        0.14,
        "Recalibration with labelled data",
        ["5 to 100 MIMIC-III patients; models not retrained"],
        MIMIC,
    )
    eval_m = box(ax, 0.82, 0.02, 1.00, 0.42, "Evaluation", ["coverage and", "width per patient,", "by true SBP"], MIMIC)
    for target in (unchanged, weighted, recal):
        arrow(ax, mimic["right"], target["left"])
        arrow(ax, target["right"], eval_m["left"])


def main():
    config = load_config()
    apply_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)  # no margins: use the full text width
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    vitaldb_row(ax)
    mimic_row(ax)
    save_figure(fig, PROJECT_ROOT / config["paths"]["results"] / "figures" / "study_design.png")
    plt.close(fig)
    print("Saved the figure")


if __name__ == "__main__":
    main()
