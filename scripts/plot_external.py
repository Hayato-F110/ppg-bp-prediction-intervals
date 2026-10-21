"""Figures of the cross-dataset analysis: coverage by band of true SBP in VitalDB and in the external dataset
(Fig. 5), and recalibration with patients of the external dataset (Fig. 6).

usage: uv run python -X utf8 scripts/plot_external.py [--dataset mimic3wdb]
output: results/figures/phase5_coverage_source_vs_target, phase5_recalibration (.png, .svg, .pdf and .eps)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.evaluation.plots import INK, MIN_TEXT, PAGE_WIDTH, SURFACE, apply_style, half_up, save_figure

BANDS = ["<90", "90–<100", "100–<120", "120–<140", "140–<160", "≥160"]
BAND_LABELS = ["<90", "90–99", "100–119", "120–139", "140–159", "≥160"]
METHODS = [("split", "Split conformal", "#2a78d6", "o"), ("cqr", "CQR", "#eb6834", "s")]
NAMES = {"mimic3wdb": "MIMIC-III"}
RECALIBRATION_PANELS = (  # column, scale, y label, title
    ("coverage", 100.0, "Patient-level coverage (%)", "All segments"),
    ("coverage_sbp_ge160_patient", 100.0, "Patient-level coverage (%)", "True SBP ≥ 160 mmHg"),
    ("width_median", 1.0, "Median width (mmHg)", "Interval width"),
)


def rows_for(coverage, method, variable, level=0.9, model="lightgbm"):
    return coverage[
        (coverage.target == "sbp")
        & (coverage.level == level)
        & (coverage.model == model)
        & (coverage.variable == variable)
        & (coverage.method == method)
    ]


def coverage_figure(source, target, name, out_path, level=0.9):
    """Fig. 5: one panel per method; filled markers VitalDB, open markers the external dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0), constrained_layout=True, sharey=True)
    for ax, (method, method_name, color, marker) in zip(axes, METHODS, strict=True):
        for table, dataset, filled, dx in ((source, "VitalDB test", True, -0.12), (target, name, False, 0.12)):
            r = rows_for(table, method, "true_sbp").set_index("stratum").reindex(BANDS)
            x = np.arange(len(BANDS)) + dx
            ax.vlines(x, 100 * r.coverage_lo, 100 * r.coverage_hi, color=color, linewidth=2.2)
            overall = float(rows_for(table, method, "all").coverage.iloc[0])
            ax.plot(
                x,
                100 * r.coverage,
                marker,
                color=color,
                markerfacecolor=color if filled else SURFACE,
                markeredgecolor=color if not filled else SURFACE,
                markeredgewidth=1.2,
                markersize=4,
                linestyle="none",
                label=f"{dataset}: {half_up(100 * overall)}% overall",
            )
        ax.axhline(100 * level, color=INK, linewidth=1.0, linestyle="--")
        ax.set_xticks(range(len(BANDS)), BAND_LABELS, rotation=30)
        ax.set_ylim(0, 106)
        ax.set_xlabel("True SBP (mmHg)")
        ax.set_title(method_name, loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="lower center")
    axes[0].set_ylabel("Patient-level coverage (%)")
    axes[1].annotate(
        f"nominal {100 * level:.0f}%",
        (1.0, 100 * level),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        fontsize=MIN_TEXT,
        color=INK,
    )
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


def recalibration_lines(ax, table, column, scale):
    """Mean over the random draws (the spread between draws is in the supplementary data, sheet recalibration)."""
    for method, method_name, color, marker in METHODS:
        grouped = table[table.method == method].groupby("subjects")[column]
        x = np.array(sorted(grouped.groups))
        ax.plot(
            x,
            scale * grouped.mean().to_numpy(),
            marker=marker,
            color=color,
            linewidth=1.4,
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            label=method_name,
        )


def offset_corrected_lines(axes, shift_path: Path):
    """Recalibration that also corrects the offset of the estimate (dashed; explore_review_analyses.py)."""
    shifted = pd.read_csv(shift_path)
    shifted = shifted[shifted.shifted & np.isfinite(shifted.width_median)]
    for ax, (column, scale, _, _) in zip(axes, RECALIBRATION_PANELS, strict=True):
        for method, method_name, color, marker in METHODS:
            mean = shifted[shifted.method == method].groupby("subjects")[column].mean()
            ax.plot(
                mean.index,
                scale * mean.to_numpy(),
                linestyle="--",
                color=color,
                linewidth=1.1,
                marker=marker,  # open markers tell the methods apart in greyscale
                markersize=3.5,
                markerfacecolor=SURFACE,
                label=f"{method_name}, offset corrected",
            )


def source_width_lines(ax, source):
    """Median width in VitalDB for comparison."""
    for method, method_name, color, _ in METHODS:
        width_source = float(rows_for(source, method, "all").width_median.iloc[0])
        ax.axhline(width_source, color=color, linewidth=0.9, linestyle="-.")
        ax.annotate(
            f"VitalDB, {method_name}: {half_up(width_source, 1)}",
            (1.0, width_source),
            xycoords=("axes fraction", "data"),
            xytext=(-4, -9 if method == "cqr" else 3),
            textcoords="offset points",
            ha="right",
            fontsize=MIN_TEXT,
            color=INK,
        )


def recalibration_figure(table, source, name, out_path, level=0.9, model="lightgbm"):
    """Fig. 6: coverage overall, coverage at true SBP >= 160 and width against the number of calibration patients."""
    table = table[(table.model == model) & np.isfinite(table.width_median)]
    sizes = np.array(sorted(table.subjects.unique()))
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.9), constrained_layout=True)
    for ax, (column, scale, ylabel, title) in zip(axes, RECALIBRATION_PANELS, strict=True):
        recalibration_lines(ax, table, column, scale)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(v) for v in sizes])
        ax.minorticks_off()
        ax.set_xlabel(f"{name} patients")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    for ax in axes[:2]:
        ax.axhline(100 * level, color=INK, linewidth=1.0, linestyle="--")
        ax.set_ylim(40, 105)  # all values lie above 40%; a 0-100 axis left most of the panel empty
    shift_path = Path(out_path).parents[1] / "review" / "recalibration_shift.csv"
    if shift_path.exists():
        offset_corrected_lines(axes, shift_path)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4, fontsize=MIN_TEXT)
    source_width_lines(axes[2], source)
    axes[2].set_ylim(0, None)
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic3wdb")
    args = parser.parse_args()
    config = load_config()
    results = PROJECT_ROOT / config["paths"]["results"]
    name = NAMES.get(args.dataset, args.dataset)
    apply_style()
    source = pd.read_csv(results / "conformal" / "coverage.csv")
    target = pd.read_csv(results / "external" / args.dataset / "coverage.csv")
    coverage_figure(source, target, name, results / "figures" / "phase5_coverage_source_vs_target.png")
    recalibration_figure(
        pd.read_csv(results / "external" / args.dataset / "recalibration.csv"),
        source,
        name,
        results / "figures" / "phase5_recalibration.png",
    )
    print("Saved the figures")


if __name__ == "__main__":
    main()
