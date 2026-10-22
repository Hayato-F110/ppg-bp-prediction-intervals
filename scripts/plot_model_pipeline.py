"""Fig. 2: from one 10-s PPG segment to a prediction interval (inputs and outputs of the two models, and the conformal
step). Added 2026-10-22 at the owner's request (item 42), so that readers see the inputs and outputs at a glance.

The settings are those of table S3 and section 2.4 of the paper (configs/feature_models.yaml,
configs/waveform_model.yaml, src/ppg_bp/models/cnn.py); no result file is read.

usage: uv run python -X utf8 scripts/plot_model_pipeline.py
output: results/figures/model_pipeline (.png, .svg, .pdf and .eps)
"""

import itertools

import matplotlib.pyplot as plt

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.evaluation.plots import (
    FLOW_NEUTRAL,
    FLOW_SOURCE,
    INK,
    PAGE_WIDTH,
    apply_style,
    flow_arrow,
    flow_box,
    save_figure,
)

INTERVAL = "#e6f2e6"  # the conformal step
ROWS = {"feature": (0.70, 0.95), "waveform": (0.37, 0.62), "interval": (0.00, 0.25)}  # y range of each row
COLUMNS = ((0.14, 0.34), (0.38, 0.58), (0.62, 0.80))  # x ranges of the three steps of each model
LEFT = COLUMNS[0][0]


def model_row(ax, row, steps):
    y0, y1 = ROWS[row]
    boxes = [
        flow_box(ax, x0, y0, x1, y1, title, lines, FLOW_NEUTRAL)
        for (x0, x1), (title, lines) in zip(COLUMNS, steps, strict=True)
    ]
    for a, b in itertools.pairwise(boxes):
        flow_arrow(ax, a["right"], b["left"])
    return boxes


def models(ax):
    """Upper two rows: the input segment, the two models and their outputs."""
    segment = flow_box(
        ax,
        0.00,
        ROWS["waveform"][0],
        0.10,
        ROWS["feature"][1],
        "Input",
        ["10-s PPG", "segment,", "125 Hz,", "1,250", "samples"],
        FLOW_SOURCE,
    )
    feature = model_row(
        ax,
        "feature",
        [
            ("APG, fiducial points", ["band-pass filter,", "2nd derivative (APG),", "waves a–f per beat"]),
            ("42 features", ["APG ratios and times,", "pulse shape, rhythm,", "signal quality"]),
            ("Boosted trees", ["a model for the mean", "and one for each", "quantile"]),
        ],
    )
    waveform = model_row(
        ax,
        "waveform",
        [
            ("Convolution", ["standardised input;", "15 taps, stride 2"]),
            ("Residual blocks", ["4 stages of 2 blocks,", "kernel 7, 32 to 96", "channels"]),
            ("Pooling, linear", ["21 outputs:", "3 pressures × (mean", "+ 6 quantiles)"]),
        ],
    )
    for first in (feature[0], waveform[0]):
        flow_arrow(ax, segment["right"], first["left"])
    outputs = flow_box(
        ax,
        0.84,
        ROWS["waveform"][0],
        1.00,
        ROWS["feature"][1],
        "Per segment",
        [
            "estimate $\\hat{y}$",
            "(mean),",
            "lower and upper",
            "quantiles",
            "$\\hat{y}_{\\mathrm{lo}}$, $\\hat{y}_{\\mathrm{hi}}$",
        ],
        FLOW_NEUTRAL,
    )
    flow_arrow(ax, feature[-1]["right"], (0.84, sum(ROWS["feature"]) / 2))
    flow_arrow(ax, waveform[-1]["right"], (0.84, sum(ROWS["waveform"]) / 2))
    return outputs


def conformal(ax, outputs):
    """Lower row: margin from the calibration patients, and the interval around the outputs."""
    y0, y1 = ROWS["interval"]
    calibration = flow_box(
        ax,
        LEFT,
        y0,
        0.46,
        y1,
        "Calibration patients",
        [
            "627 patients with known $y$; score",
            "$|y - \\hat{y}|$ (split) or",
            "$\\max(\\hat{y}_{\\mathrm{lo}} - y,\\ y - \\hat{y}_{\\mathrm{hi}})$ (CQR)",
        ],
        INTERVAL,
    )
    margin = flow_box(
        ax,
        0.50,
        y0,
        0.80,
        y1,
        "Margin $q$",
        [
            "weighted quantile of the scores,",
            "weight $1/n_i$ per segment",
            "of patient $i$; group-wise:",
            "one $q$ per group",
        ],
        INTERVAL,
    )
    interval = flow_box(
        ax,
        0.84,
        y0,
        1.00,
        y1,
        "Interval",
        ["split: $\\hat{y} \\pm q$", "CQR:", "$[\\hat{y}_{\\mathrm{lo}} - q,\\ \\hat{y}_{\\mathrm{hi}} + q]$"],
        INTERVAL,
    )
    flow_arrow(ax, calibration["right"], margin["left"])
    flow_arrow(ax, margin["right"], interval["left"])
    flow_arrow(ax, outputs["bottom"], interval["top"])


def main():
    config = load_config()
    apply_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.9))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)  # no margins: use the full text width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.03)
    ax.axis("off")
    heading = {"ha": "left", "va": "bottom", "fontsize": 9, "fontweight": "bold", "color": INK}
    ax.text(LEFT, ROWS["feature"][1] + 0.01, "Feature model (LightGBM)", **heading)
    ax.text(LEFT, ROWS["waveform"][1] + 0.01, "Waveform model (one-dimensional residual CNN)", **heading)
    ax.text(LEFT, ROWS["interval"][1] + 0.01, "Prediction interval (conformal prediction)", **heading)
    conformal(ax, models(ax))
    save_figure(fig, PROJECT_ROOT / config["paths"]["results"] / "figures" / "model_pipeline.png")
    plt.close(fig)
    print("Saved the figure")


if __name__ == "__main__":
    main()
