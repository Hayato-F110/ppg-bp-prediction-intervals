"""Figures of the main text that show comparisons (Figs. 3, 5 and 8) and two supplementary figures.

- main_accuracy (Fig. 3): accuracy of the point estimate in the 627 test patients. The estimate against the true SBP
  (a) shows the shrinkage directly. PPG-only models are only a little better than returning the training median (b).
  Estimates are pulled towards the centre of the training data, so low pressures are overestimated and high
  pressures underestimated (c).
- main_subgroups (Fig. 5): coverage of split conformal intervals (nominal 90%) in subgroups other than true pressure.
- main_withholding (Fig. 8): withholding estimates with wide intervals. Keeping fewer estimates does not bring the SD
  of the error down to 8 mmHg (left). Thresholds set on VitalDB withhold most segments of MIMIC-III (right).
- supp_bland_altman_test, supp_slope_curve: supplementary figures.

usage: uv run python -X utf8 scripts/plot_main_figures.py
input: results/review/test_accuracy.csv, results/conformal/predictions_test.parquet, results/conformal/coverage.csv,
       results/conformal/skewness_strata.csv, results/conformal/selective.csv, results/review2/slope_curve.csv
output: results/figures/main_accuracy, main_subgroups, main_withholding, supp_bland_altman_test, supp_slope_curve
        (.png, .svg, .pdf and .eps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import band_labels
from ppg_bp.constants import FEW_PATIENTS  # groups with fewer patients are marked † (reference only)
from ppg_bp.evaluation.plots import (
    AXIS,
    DENSITY,
    INK,
    INK_SECONDARY,
    MIN_TEXT,
    PAGE_WIDTH,
    SURFACE,
    TARGET_LABELS,
    apply_style,
    bland_altman,
    dot,
    half_up,
    label,
    save_figure,
    signed,
)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a0"
PREDICTORS = [("Training median", GREY), ("Feature model", BLUE), ("Waveform model", ORANGE)]
BAR_HATCHES = ("..", "//", "\\\\")  # a pattern per bar series, so that the bars stay apart in greyscale
SBP_EDGES = load_settings("conformal")["strata"]["sbp"]
SBP_BAND_LABELS = band_labels(SBP_EDGES)
MODELS = (("lightgbm", "Feature model", BLUE), ("cnn", "Waveform model", ORANGE))
CUFF_SD = 8.0  # SD limit of the error for cuff devices (mmHg)


def error_by_true_band(test: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles of estimate − true SBP by band of true SBP, for both models."""
    which = np.digitize(test["sbp"], SBP_EDGES)
    rows = []
    for model, _, _ in MODELS:
        error = test[f"sbp_{model}_mean"] - test["sbp"]
        for k, band in enumerate(SBP_BAND_LABELS):
            q25, q50, q75 = np.percentile(error[which == k], [25, 50, 75])
            rows.append({"model": model, "band": band, "median": q50, "q25": q25, "q75": q75})
    return pd.DataFrame(rows)


def mae_panel(ax, metrics: pd.DataFrame):
    rows = {name: part.set_index("target") for name, part in metrics.groupby("predictor")}
    targets = ["sbp", "dbp"]
    width = 0.26
    for k, ((name, color), hatch) in enumerate(zip(PREDICTORS, BAR_HATCHES, strict=True)):
        values = [rows[name].loc[t, "mae"] for t in targets]
        lo, hi = ([rows[name].loc[t, c] for t in targets] for c in ("mae_lo", "mae_hi"))
        x = np.arange(len(targets)) + (k - 1) * width
        ax.bar(x, values, width=width * 0.92, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0, label=name)
        # 95% CI from a bootstrap over patients
        ax.vlines(x, lo, hi, color=INK, linewidth=1.0)
        for xi, v, top in zip(x, values, hi, strict=True):
            label(ax, half_up(v, 1), (xi, top), (0, 2))
    ax.set_xticks(range(len(targets)), ["SBP", "DBP"])
    ax.set_ylabel("Mean absolute error (mmHg)")
    ax.set_title("Accuracy in test patients", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, loc="upper right")


def error_panel(ax, bands: pd.DataFrame):
    labels = list(dict.fromkeys(bands.band))
    for dx, (model, name, color) in zip((-0.12, 0.12), MODELS, strict=True):
        part = bands[bands.model == model].set_index("band").reindex(labels)
        x = np.arange(len(labels)) + dx
        ax.vlines(x, part.q25, part.q75, color=color, linewidth=1.4)
        # marker shape as well as colour, so that the models stay apart in greyscale
        dot(ax, x, part["median"], color, "o" if model == "lightgbm" else "s", label=name)
    feature_part = bands[bands.model == "lightgbm"].set_index("band").reindex(labels)
    for k, v in enumerate(feature_part["median"]):
        label(ax, signed(v, 0), (k - 0.12, v), (-5, 0), ha="right", va="center")
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("Estimate − true SBP (mmHg)")
    ax.set_title("Error by true SBP (median and IQR)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")


SCATTER_RANGE = (50.0, 210.0)  # mmHg, both axes of Fig. 3(a); the 0.2-99.8th percentiles of true SBP lie inside


def scatter_panel(ax, test: pd.DataFrame, shrinkage_path: Path):
    """Fig. 3(a): estimate against true SBP (feature model) as a density, with the line of identity and the line of
    the shrinkage model fitted to the calibration patients (explore_review_analyses.py)."""
    low, high = SCATTER_RANGE
    mesh = ax.hexbin(
        test["sbp"],
        test["sbp_lightgbm_mean"],
        gridsize=60,
        extent=(low, high, low, high),
        cmap=DENSITY,
        mincnt=1,
        bins="log",
        linewidths=0,
    )
    x = np.array([low, high])
    # a light halo keeps the lines visible over the darkest bins, also in greyscale
    halo = {"path_effects": [patheffects.withStroke(linewidth=3.2, foreground=SURFACE)]}
    ax.plot(x, x, color=INK, linewidth=1.0, linestyle="--", label="identity", **halo)
    model = pd.read_csv(shrinkage_path, index_col=0)["value"]
    # bands around identity: the SD limit for cuff devices (reference only), and the 90% split conformal margin q,
    # inside which a segment is covered
    for offset, style, name in (
        (CUFF_SD, ":", f"± {CUFF_SD:.0f} mmHg (SD limit, cuff devices)"),
        (model["q"], "-.", f"± {model['q']:.1f} mmHg (90% split conformal)"),
    ):
        for sign in (1, -1):
            ax.plot(
                x,
                x + sign * offset,
                color=INK,
                linewidth=0.8,
                linestyle=style,
                label=name if sign == 1 else None,
                **halo,
            )
    ax.plot(
        x,
        model["intercept"] + model["slope"] * x,
        color=INK,
        linewidth=1.4,
        label=f"shrinkage model, slope {model['slope']:.2f}",
        **halo,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("Estimated SBP (mmHg)")
    ax.set_title("Feature model, test patients", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1)  # below the panel, off the data
    return mesh


def accuracy(results: Path, out_path: Path):
    """Fig. 3."""
    metrics = pd.read_csv(results / "review" / "test_accuracy.csv")  # explore_review_analyses.py
    test = pd.read_parquet(
        results / "conformal" / "predictions_test.parquet", columns=["sbp", "sbp_lightgbm_mean", "sbp_cnn_mean"]
    )
    fig = plt.figure(figsize=(PAGE_WIDTH, 5.9), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.15, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    mesh = scatter_panel(axes[0], test, results / "review" / "shrinkage_model.csv")
    colorbar = fig.colorbar(mesh, ax=axes[0], shrink=0.8, aspect=25, pad=0.02, label="Segments per bin (log)")
    colorbar.outline.set_visible(False)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    mae_panel(axes[1], metrics)
    error_panel(axes[2], error_by_true_band(test))
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


SUBGROUPS = [
    (
        "Within patient",
        "within_subject",
        [
            ("<-20", "> 20 mmHg below own median"),
            ("-20–<20", "within 20 mmHg of own median"),
            ("≥20", "≥ 20 mmHg above own median"),
        ],
    ),
    (
        "Pulse rate",
        "pulse_rate",
        [("<60", "< 60 per minute"), ("60–<80", "60–79"), ("80–<100", "80–99"), ("≥100", "≥ 100")],
    ),
    ("Signal quality", "sqi", [("low", "lowest tertile"), ("middle", "middle tertile"), ("high", "highest tertile")]),
    (
        "PPG skewness",
        "skewness",
        [("low", "lowest tertile"), ("middle", "middle tertile"), ("high", "highest tertile")],
    ),
    ("Age", "age", [("<40", "< 40 years"), ("40–<60", "40–59"), ("60–<75", "60–74"), ("≥75", "≥ 75")]),
    ("Sex", "sex", [("F", "female"), ("M", "male")]),
    (
        "Body mass index",
        "bmi",
        [("<18.5", "< 18.5 kg/m²"), ("18.5–<25", "18.5–24.9"), ("25–<30", "25–29.9"), ("≥30", "≥ 30")],
    ),
]


def subgroup_coverage(results: Path, level: float) -> pd.DataFrame:
    """Split conformal coverage of SBP in all subgroups, including the PPG skewness tertiles
    (planned sensitivity analysis, evaluate_skewness_strata.py)."""
    coverage = pd.read_csv(results / "conformal" / "coverage.csv")
    rows = coverage[
        (coverage.target == "sbp")
        & (coverage.level == level)
        & (coverage.model == "lightgbm")
        & (coverage.method == "split")
    ]
    skew = pd.read_csv(results / "conformal" / "skewness_strata.csv")
    return pd.concat([rows, skew[skew.method == "split"].assign(variable="skewness")], ignore_index=True)


def draw_subgroups(ax, rows: pd.DataFrame, level: float) -> float:
    """One row per subgroup, a bold heading per variable. Returns the y of the last row."""
    y, ticks, tick_labels, headings = 0, [], [], []
    for group, variable, strata in SUBGROUPS:
        headings.append((group, y))
        y -= 1
        for stratum, name in strata:
            r = rows[(rows.variable == variable) & (rows.stratum == stratum)].iloc[0]
            # highlighted when the whole 95% confidence interval lies below nominal (not an arbitrary margin)
            low = r.coverage_hi < level
            color = ORANGE if low else BLUE
            ax.hlines(y, 100 * r.coverage_lo, 100 * r.coverage_hi, color=color, linewidth=1.4)
            dot(ax, 100 * r.coverage, y, color, "s" if low else "o")  # the shortfall shows in greyscale too
            ax.annotate(
                f"{half_up(100 * r.coverage, 1)}  (n = {int(r.subjects)}){' †' if r.subjects < FEW_PATIENTS else ''}",
                (101, y),
                xycoords="data",
                va="center",
                fontsize=MIN_TEXT,
                color=INK if low else INK_SECONDARY,
                annotation_clip=False,
            )
            ticks.append(y)
            tick_labels.append(name)
            y -= 1
        y -= 0.4
    for group, gy in headings:
        ax.annotate(
            group,
            (0, gy),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=MIN_TEXT,
            fontweight="bold",
            color=INK,
            annotation_clip=False,
        )
    ax.set_yticks(ticks, tick_labels)
    return y


def subgroups(results: Path, out_path: Path, level=0.9):
    """Fig. 5."""
    rows = subgroup_coverage(results, level)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 4.6), constrained_layout=True)
    y = draw_subgroups(ax, rows, level)
    ax.axvline(100 * level, color=INK, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"nominal {100 * level:.0f}%",
        (100 * level, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=MIN_TEXT,
        color=INK,
    )
    # the meaning of orange (confidence interval wholly below nominal) is given in the caption
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(60, 100)
    ax.set_xlabel("Patient-level coverage (%), split conformal, 95% CI")
    ax.grid(axis="y", visible=False)
    save_figure(fig, out_path)
    plt.close(fig)


def kept_error_panel(ax, vitaldb: pd.DataFrame):
    """Fig. 8(a): error of the kept estimates (their coverage is in the supplementary data, withholding.csv)."""
    for column, name, marker in (("sd", "SD of error", "o"), ("mae", "Mean absolute error", "s")):
        ax.plot(
            100 * vitaldb.accepted,
            vitaldb[column],
            marker=marker,
            color=BLUE if column == "sd" else ORANGE,
            linewidth=1.4,
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            label=name,
        )
    ax.axhline(CUFF_SD, color=INK, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"SD limit for cuff devices ({CUFF_SD:.0f} mmHg)",
        (1.0, CUFF_SD),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        fontsize=MIN_TEXT,
        color=INK,
    )
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 19)
    ax.set_xlabel("VitalDB test segments kept (%)")
    ax.set_ylabel("SBP error (mmHg)")
    ax.set_title("Error of the kept estimates", loc="left")
    ax.legend(frameon=False, loc="lower right")


def kept_share_panel(ax, vitaldb: pd.DataFrame, mimic: pd.DataFrame):
    """Right panel of Fig. 8: share kept on MIMIC-III with the thresholds set on VitalDB."""
    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=1.0, linestyle=":")
    ax.plot(
        100 * vitaldb.accepted.to_numpy(),
        100 * mimic.accepted.to_numpy(),
        marker="o",
        color=BLUE,
        linewidth=1.4,
        markersize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
    )
    label(ax, "same share kept", (70, 70), (-4, 4), ha="right", size=MIN_TEXT, color=INK_SECONDARY, rotation=38)
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 103)
    ax.set_xlabel("VitalDB test segments kept (%)")
    ax.set_ylabel("MIMIC-III segments kept (%)")
    ax.set_title("Same thresholds on MIMIC-III", loc="left")


def withholding(results: Path, out_path: Path):
    """Fig. 8."""
    d = pd.read_csv(results / "conformal" / "selective.csv")
    d = d[(d.target == "sbp") & (d.model == "lightgbm")]
    vitaldb = d[d.dataset == "vitaldb_test"].sort_values("nominal_accept")
    mimic = d[d.dataset == "mimic3wdb"].sort_values("nominal_accept")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.7), constrained_layout=True, gridspec_kw={"wspace": 0.12})
    kept_error_panel(axes[0], vitaldb)
    kept_share_panel(axes[1], vitaldb, mimic)
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


def supp_bland_altman(results: Path, out_path: Path):
    """Supplementary figure: Bland-Altman plots in the test patients (both models; SBP, DBP and MBP)."""
    targets = ("sbp", "dbp", "mbp")
    test = pd.read_parquet(
        results / "conformal" / "predictions_test.parquet",
        columns=[*targets, *[f"{t}_{m}_mean" for t in targets for m in ("lightgbm", "cnn")]],
    )
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 5.8), constrained_layout=True)
    mesh = None
    for row, (model, name, _) in enumerate(MODELS):
        for column, (ax, target) in enumerate(zip(axes[row], targets, strict=True)):
            mesh, _ = bland_altman(ax, test[target], test[f"{target}_{model}_mean"], f"{name}: {TARGET_LABELS[target]}")
            # axis titles only on the outer panels, so that the panels keep their width
            if column > 0:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
    if mesh is None:
        raise AssertionError("no panel was drawn")
    colorbar = fig.colorbar(
        mesh, ax=axes, location="bottom", shrink=0.5, aspect=40, pad=0.02, label="Segments per bin (log)"
    )
    colorbar.outline.set_visible(False)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


def supp_slope_curve(results: Path, out_path: Path):
    """Supplementary figure: under the shrinkage model, expected coverage at true SBP >= 160 and the interval width
    against the slope b of the estimate (explore_review2_analyses.py). The slope and residual SD of this study come
    from results/review/shrinkage_model.csv (explore_review_analyses.py), rounded as in the curve."""
    curve = pd.read_csv(results / "review2" / "slope_curve.csv")
    fitted = pd.read_csv(results / "review" / "shrinkage_model.csv", index_col=0)["value"]
    residual_sd, slope = round(float(fitted["residual_sd"]), 1), round(float(fitted["slope"]), 2)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6), constrained_layout=True)
    for s, color, marker in ((residual_sd, BLUE, "o"), (5.0, ORANGE, "s")):
        part = curve[curve.residual_sd == s]
        name = f"residual SD {s:g} mmHg" + (" (this study)" if s == residual_sd else "")
        style = {"marker": marker, "color": color, "markersize": 3.5, "linewidth": 1.3, "label": name}
        axes[0].plot(part.slope, 100 * part.coverage_sbp_ge160, **style)
        axes[1].plot(part.slope, part.width, **style)
    axes[0].axhline(90, color=INK, linewidth=1.0, linestyle="--")
    axes[0].axvline(slope, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    label(axes[0], f"this study (b = {slope:.2f})", (slope, 50), (3, 0), ha="left", color=INK_SECONDARY)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Expected coverage, true SBP ≥ 160 (%)")
    axes[0].set_title("Coverage at high pressure", loc="left")
    axes[1].set_ylabel("Interval width (mmHg)")
    axes[1].set_title("Width of the split conformal interval", loc="left")
    axes[1].set_ylim(0, None)
    for ax in axes:
        ax.set_xlabel("Slope b of the estimate on the true SBP")
    axes[1].legend(frameon=False, loc="lower left")  # the coverage panel has no free corner
    save_figure(fig, out_path, panels=axes)
    plt.close(fig)


def main():
    config = load_config()
    results = PROJECT_ROOT / config["paths"]["results"]
    apply_style()
    out = results / "figures"
    accuracy(results, out / "main_accuracy.png")
    subgroups(results, out / "main_subgroups.png")
    withholding(results, out / "main_withholding.png")
    supp_bland_altman(results, out / "supp_bland_altman_test.png")
    supp_slope_curve(results, out / "supp_slope_curve.png")
    print("Saved the figures")


if __name__ == "__main__":
    main()
