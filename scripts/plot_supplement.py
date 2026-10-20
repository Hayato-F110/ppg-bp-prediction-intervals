"""Supplementary figures that replace wide tables of coverage values (the exact values are in the supplementary data).

- supp_coverage_by_band: coverage by band of true pressure in VitalDB, for the waveform model and for DBP
- supp_true_vs_predicted: coverage by band of true and of predicted SBP, with the shrinkage model
- supp_differences: paired differences in coverage between methods and between models (forest plot)
- supp_reference_reading: one reference reading per patient

usage: uv run python -X utf8 scripts/plot_supplement.py
input: results/conformal/coverage.csv, results/external/mimic3wdb/coverage.csv, results/review/shrinkage.csv,
       results/review/differences.csv, results/review2/model_comparison.csv, results/review3/offset_calibration.csv
output: results/figures/supp_*.png (.svg, .pdf, .eps)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT
from ppg_bp.constants import MAIN_LEVEL
from ppg_bp.evaluation.plots import INK, MIN_TEXT, MUTED, PAGE_WIDTH, apply_style, dot, save_figure

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
SBP_BANDS = ["<90", "90–<100", "100–<120", "120–<140", "140–<160", "≥160"]
DBP_BANDS = ["<50", "50–<60", "60–<70", "70–<80", "80–<90", "≥90"]
# colours and markers as in Fig. 3
SPLIT, CQR = ("Split conformal", "#2a78d6", "o"), ("CQR", "#eb6834", "s")


def band_label(band: str) -> str:
    """ "90–<100" -> "90–99"."""
    if "–<" not in band:
        return band
    low, high = band.split("–<")
    return f"{low}–{int(high) - 1}"


def nominal_line(ax) -> None:
    ax.axhline(100 * MAIN_LEVEL, color=INK, linewidth=1.0, linestyle="--")


def coverage_by_band() -> None:
    # the feature model for SBP is in Fig. 3 (VitalDB) and Fig. 5 (MIMIC-III) of the main text
    source = pd.read_csv(RESULTS / "conformal" / "coverage.csv")
    panels = [
        ("Waveform model, SBP", source, "cnn", "sbp", SBP_BANDS),
        ("Feature model, DBP", source, "lightgbm", "dbp", DBP_BANDS),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8), sharey=True, constrained_layout=True)
    for ax, (title, coverage, model, pressure, bands) in zip(axes, panels, strict=True):
        rows = coverage[
            (coverage.model == model)
            & (coverage.target == pressure)
            & np.isclose(coverage.level, MAIN_LEVEL)
            & (coverage.variable == f"true_{pressure}")
        ]
        for (name, color, marker), method, dx in ((SPLIT, "split", -0.12), (CQR, "cqr", 0.12)):
            r = rows[rows.method == method].set_index("stratum").reindex(bands)
            x = np.arange(len(bands)) + dx
            ax.vlines(x, 100 * r.coverage_lo, 100 * r.coverage_hi, color=color, linewidth=2)
            dot(ax, x, 100 * r.coverage, color, marker, size=3.6, linestyle="none", label=name)
        nominal_line(ax)
        ax.set_xticks(range(len(bands)), [band_label(b) for b in bands], rotation=30)
        ax.set_xlabel(f"True {pressure.upper()} (mmHg)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Patient-level coverage (%)")
    axes[0].legend(frameon=False, loc="lower center")
    save_figure(fig, FIGURES / "supp_coverage_by_band.png", panels=axes)
    plt.close(fig)


def true_vs_predicted() -> None:
    shrink = pd.read_csv(RESULTS / "review" / "shrinkage.csv", index_col=0).reindex(SBP_BANDS)
    x = np.arange(len(SBP_BANDS))
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.6, 2.8), constrained_layout=True)
    ax.plot(x, 100 * shrink.observed_by_true, "-o", color="#2a78d6", markersize=4, label="By true SBP, split conformal")
    ax.plot(x, 100 * shrink.expected_by_true, "--", color=INK, linewidth=1.2, label="By true SBP, shrinkage model")
    for column, name, color, marker, dx in (
        ("observed_by_predicted", "By predicted SBP, split conformal", "#1baf7a", "^", -0.08),
        ("groupwise_by_predicted", "By predicted SBP, group-wise", "#eda100", "D", 0.08),
    ):
        ax.vlines(x + dx, 100 * shrink[f"{column}_lo"], 100 * shrink[f"{column}_hi"], color=color, linewidth=2)
        dot(ax, x + dx, 100 * shrink[column], color, marker, size=3.6, linestyle="none", label=name)
    nominal_line(ax)
    ax.set_xticks(x, [band_label(b) for b in SBP_BANDS])
    ax.set_xlabel("SBP band (mmHg)")
    ax.set_ylabel("Patient-level coverage (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", fontsize=MIN_TEXT)
    save_figure(fig, FIGURES / "supp_true_vs_predicted.png")
    plt.close(fig)


def forest(ax, rows: pd.DataFrame, title: str) -> None:
    y = np.arange(len(rows))[::-1]
    ax.hlines(y, 100 * rows.ci_low, 100 * rows.ci_high, color=INK, linewidth=1.6)
    dot(ax, 100 * rows.difference, y, INK, "o", size=3.6, linestyle="none")
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.set_yticks(y, rows.label)
    ax.set_xlabel("Difference in coverage (percentage points)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def differences() -> None:
    names = {
        "cqr − split": "CQR − split",
        "mondrian_pred_sqi − split": "Group-wise − split",
        "cqr_mondrian_pred_sqi − cqr": "CQR + group-wise − CQR",
    }
    short = {"VitalDB test": "VitalDB", "MIMIC-III": "MIMIC-III"}
    methods = pd.read_csv(RESULTS / "review" / "differences.csv")
    methods["label"] = [
        f"{short[d]}, {s}: {names.get(c, c)}"
        for d, s, c in zip(methods.dataset, methods.stratum, methods.comparison, strict=True)
    ]
    models = pd.read_csv(RESULTS / "review2" / "model_comparison.csv")
    models = models[models.measure.str.startswith("coverage")]  # MAE is in mmHg and is given in the text
    strata = {"all": "all", "≥160": "SBP ≥ 160", "<90": "SBP < 90"}
    measures = {"coverage split": "split conformal", "coverage cqr": "CQR"}
    models["label"] = [
        f"{short[d]}, {strata[s]}: {measures[m]}"
        for d, s, m in zip(models.dataset, models.stratum, models.measure, strict=True)
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(PAGE_WIDTH * 0.75, 0.22 * (len(methods) + len(models)) + 1.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [len(methods), len(models)]},
    )
    forest(axes[0], methods, "Between methods, feature model")
    forest(axes[1], models, "Waveform model minus feature model")
    save_figure(fig, FIGURES / "supp_differences.png", panels=axes)
    plt.close(fig)


def reference_reading() -> None:
    offset = pd.read_csv(RESULTS / "review3" / "offset_calibration.csv")
    offset = offset[(offset.variable == "all") & (offset.stratum == "all")]
    # without a reference reading: the main analysis (feature model, SBP, nominal 90%)
    no_reference_mae = {
        "VitalDB test": pd.read_csv(RESULTS / "review" / "test_accuracy.csv")
        .set_index(["target", "predictor"])
        .loc[("sbp", "Feature model"), "mae"],
        "MIMIC-III": pd.read_csv(RESULTS / "external" / "mimic3wdb" / "point_accuracy.csv")
        .set_index(["model", "target"])
        .loc[("lightgbm", "sbp"), "mae"],
    }
    no_reference_coverage = {
        "VitalDB test": pd.read_csv(RESULTS / "conformal" / "coverage.csv"),
        "MIMIC-III": pd.read_csv(RESULTS / "external" / "mimic3wdb" / "coverage.csv"),
    }
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6), constrained_layout=True)
    for dataset, style in (("VitalDB test", "-"), ("MIMIC-III", ":")):
        part = offset[offset.dataset == dataset]
        mae = [no_reference_mae[dataset], *part[part.method == "split"].sort_values("reference_segments").mae]
        axes[0].plot(x, mae, style + "o", color=INK, markersize=4, label=dataset)
        plain = no_reference_coverage[dataset]
        plain = plain[
            (plain.model == "lightgbm")
            & (plain.target == "sbp")
            & np.isclose(plain.level, MAIN_LEVEL)
            & (plain.variable == "all")
        ].set_index("method")
        for (name, color, marker), method, dx in ((SPLIT, "split", -0.06), (CQR, "cqr", 0.06)):
            r = pd.concat([plain.loc[[method]], part[part.method == method].sort_values("reference_segments")])
            axes[1].vlines(x + dx, 100 * r.coverage_lo, 100 * r.coverage_hi, color=color, linewidth=2)
            axes[1].plot(x + dx, 100 * r.coverage, style + marker, color=color, markersize=4, label=name)
    for ax in axes:
        ax.set_xticks(x, ["none", "10 s", "1 min", "5 min"])
        ax.set_xlabel("Reference reading at the start")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE of SBP (mmHg)")
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, loc="lower left")
    nominal_line(axes[1])
    axes[1].set_ylabel("Patient-level coverage (%)")
    handles, names = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], names[:2], frameon=False, loc="lower right")  # colour = method; line style = dataset
    save_figure(fig, FIGURES / "supp_reference_reading.png", panels=axes)
    plt.close(fig)


def main() -> None:
    apply_style()
    coverage_by_band()
    true_vs_predicted()
    differences()
    reference_reading()
    print("Saved the figures")


if __name__ == "__main__":
    main()
