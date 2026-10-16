"""Fig. 3 (coverage and direction of misses by true SBP) and Fig. S2 (number of calibration patients).

usage: uv run python -X utf8 scripts/plot_conformal.py
input: results/conformal/coverage.csv, calibration_size.csv
output: results/figures/phase4_coverage_by_sbp, phase4_calibration_size (.png and .svg)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.evaluation.plots import (
    AXIS,
    INK,
    MUTED,
    PAGE_WIDTH,
    SURFACE,
    apply_style,
    dot,
    half_up,
    label,
    save_figure,
)

BANDS = ["<90", "90–<100", "100–<120", "120–<140", "140–<160", "≥160"]
BAND_LABELS = ["<90", "90–99", "100–119", "120–139", "140–159", "≥160"]
# fixed colour order (blue, orange, teal, yellow); marker shapes also differ, so colour is not the only cue
METHODS = [
    ("split", "Split conformal", "#2a78d6", "o"),
    ("cqr", "CQR", "#eb6834", "s"),
    ("mondrian_pred_sqi", "Group-wise (predicted SBP × signal quality)", "#1baf7a", "^"),
    ("cqr_mondrian_pred_sqi", "CQR + group-wise", "#eda100", "D"),
]


def coverage_panel(ax, rows, split, level, out_path):
    """Left panel: coverage by band of true SBP for each method, with the shrinkage model for split conformal."""
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for (method, name, color, marker), dx in zip(METHODS, offsets, strict=True):
        r = rows[rows.method == method].set_index("stratum").reindex(BANDS)
        x = np.arange(len(BANDS)) + dx
        ax.vlines(x, 100 * r.coverage_lo, 100 * r.coverage_hi, color=color, linewidth=2.2)
        dot(ax, x, 100 * r.coverage, color, marker, size=3.8, linestyle="none", label=name)
    ax.axhline(100 * level, color=INK, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"nominal {100 * level:.0f}%",
        (1.0, 100 * level),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=INK,
    )
    # Coverage of split conformal expected if the estimate only shrinks linearly towards the mean
    # (scripts/explore_review_analyses.py)
    model_path = Path(out_path).parents[1] / "review" / "shrinkage.csv"
    if model_path.exists():
        expected = pd.read_csv(model_path, index_col=0).reindex(BANDS)["expected_by_true"]
        ax.plot(
            np.arange(len(BANDS)) + offsets[0],
            100 * expected,
            "_",
            color=INK,
            markersize=11,
            markeredgewidth=1.6,
            label="Split conformal, shrinkage model",
        )
    for k, value in enumerate(100 * split.coverage):
        label(ax, half_up(value), (k + offsets[0], value), (-5, -3), ha="right")
    ax.set_xticks(range(len(BANDS)), BAND_LABELS)
    ax.set_ylim(0, 105)
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("Patient-level coverage (%)")
    ax.set_title("Coverage by true SBP", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower center")


def miss_panel(ax, split):
    """Right panel: share of segments above and below the split conformal interval, by band of true SBP."""
    x = np.arange(len(BANDS))
    above, below = 100 * split.miss_above.to_numpy(), 100 * split.miss_below.to_numpy()
    ax.bar(x, above, width=0.6, color="#2a78d6", label="True value above")
    ax.bar(x, -below, width=0.6, color="#86b6ef", label="True value below")
    for k in range(len(BANDS)):
        if above[k] >= 1:
            label(ax, half_up(above[k]), (k, above[k]), (0, 2))
        if below[k] >= 1:
            label(ax, half_up(below[k]), (k, -below[k]), (0, -8))
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x, BAND_LABELS, rotation=30)
    ax.set_ylim(-40, 85)
    ax.set_yticks([-25, 0, 25, 50, 75], ["−25", "0", "25", "50", "75"])
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("Outside the interval (%)\nabove (+), below (−)")
    ax.set_title("Misses, split conformal", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")


def coverage_by_band(coverage, out_path, level=0.9, model="lightgbm"):
    """Fig. 3: coverage and misses by band of true SBP, VitalDB test patients."""
    rows = coverage[
        (coverage.target == "sbp")
        & (coverage.level == level)
        & (coverage.model == model)
        & (coverage.variable == "true_sbp")
    ]
    split = rows[rows.method == "split"].set_index("stratum").reindex(BANDS)
    fig, axes = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.3), constrained_layout=True, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    coverage_panel(axes[0], rows, split, level, out_path)
    miss_panel(axes[1], split)
    save_figure(fig, out_path)
    plt.close(fig)


def calibration_size(table, out_path, level=0.9):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8), constrained_layout=True)
    for ax, column, scale, ylabel, title in (
        (axes[0], "coverage", 100.0, "Patient-level coverage on test (%)", "Coverage"),
        (axes[1], "width_median", 1.0, "Median interval width (mmHg)", "Interval width"),
    ):
        for method, name, color, marker in METHODS[:2]:
            grouped = table[table.method == method].groupby("subjects")[column]
            x = np.array(sorted(grouped.groups))
            mean, lo, hi = (
                scale * grouped.mean().to_numpy(),
                scale * grouped.quantile(0.05).to_numpy(),
                scale * grouped.quantile(0.95).to_numpy(),
            )
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                x,
                mean,
                marker=marker,
                color=color,
                linewidth=1.4,
                markersize=4,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=name,
            )
        ax.set_xscale("log")
        sizes = np.array(sorted(table.subjects.unique()))
        ax.set_xticks(sizes, [str(v) for v in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Calibration patients")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)
    axes[0].axhline(100 * level, color=INK, linewidth=1.0, linestyle="--")
    axes[0].annotate(
        f"nominal {100 * level:.0f}%",
        (1.0, 100 * level),
        xycoords=("axes fraction", "data"),
        xytext=(-4, -9),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=INK,
    )
    axes[0].annotate(
        "45: closest prior study",
        (45, axes[0].get_ylim()[0]),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=6.5,
        color=MUTED,
        ha="center",
    )
    save_figure(fig, out_path)
    plt.close(fig)


def main():
    config = load_config()
    results = PROJECT_ROOT / config["paths"]["results"]
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    coverage_by_band(pd.read_csv(results / "conformal" / "coverage.csv"), out_dir / "phase4_coverage_by_sbp.png")
    calibration_size(
        pd.read_csv(results / "conformal" / "calibration_size.csv"), out_dir / "phase4_calibration_size.png"
    )
    print("Saved the figures")


if __name__ == "__main__":
    main()
