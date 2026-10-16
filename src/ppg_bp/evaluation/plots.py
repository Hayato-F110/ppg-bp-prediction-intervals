"""Figures for the evaluation. Text in the figures is in English for the paper.

Colour: density (amount) uses shades of one hue; lines and notes are neutral. Meaning never depends on colour alone.
"""

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

MIN_SHARE_NOTED = 0.05  # % of points beyond the axis limits above which a note is added

SURFACE = "#fcfcfb"
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
DENSITY = LinearSegmentedColormap.from_list("density", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

TARGET_LABELS = {"sbp": "SBP", "dbp": "DBP", "mbp": "MBP"}
# Text width of the paper (A4 with 2.5 cm side margins, about 16 cm). Figures are drawn at this width and not scaled, so
# text is 7-8 pt
PAGE_WIDTH = 6.5


def half_up(value: float, digits: int = 0) -> str:
    """Round half up to a string (26.5 -> "27"). Python formatting rounds half to even (26.5 -> "26"), which could
    disagree with the text."""
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def signed(value: float, digits: int = 1) -> str:
    """Signed number with a true minus sign; −0.0 is written as 0.0."""
    rounded = round(float(value), digits) + 0.0
    text = f"{abs(rounded):.{digits}f}"
    return ("−" if rounded < 0 else "+" if rounded > 0 else "") + text


def save_figure(fig: Figure, path: Path) -> None:
    """Save the figure as PNG (200 dpi) and as an SVG (vector) file of the same name.

    The .docx of the paper uses the SVG (sharp at any size). The PNG is for checking and for tools without SVG support.
    """
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".svg"))


def label(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    offset: tuple[float, float] = (0, 0),
    *,
    ha: str = "center",
    va: str = "baseline",
    size: float = 6.5,
    color: str = INK,
    **kwargs,
) -> None:
    """Small text next to a point; offset is in points."""
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=size, color=color, **kwargs)


def dot(ax: Axes, x: ArrayLike, y: ArrayLike, color: str, marker: str = "o", size: float = 4.5, **kwargs) -> None:
    """Markers with a light edge, the style of the point estimates in all figures."""
    ax.plot(x, y, marker, color=color, markersize=size, markeredgecolor=SURFACE, markeredgewidth=0.8, **kwargs)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def bland_altman(
    ax: Axes, reference: ArrayLike, estimate: ArrayLike, title: str, limit: float = 60.0
) -> tuple[PolyCollection, tuple[float, float, float]]:
    """Bland–Altman plot: mean of the two values against estimate − reference, drawn as a density because of the many
    points.

    Returns (mean error, lower limit of agreement, upper limit).
    """
    reference, estimate = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    diff, mean = estimate - reference, (estimate + reference) / 2.0
    bias, sd = diff.mean(), diff.std(ddof=1)
    lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
    extent = (*np.percentile(mean, [0.2, 99.8]), -limit, limit)
    mesh = ax.hexbin(mean, diff, gridsize=70, extent=extent, cmap=DENSITY, mincnt=1, bins="log", linewidths=0)
    for value, style, text in (
        (bias, "-", f"mean {signed(bias)}"),
        (upper, "--", f"+1.96 SD: {signed(upper)}"),
        (lower, "--", f"−1.96 SD: {signed(lower)}"),
    ):
        ax.axhline(value, color=INK, linewidth=0.8, linestyle=style)
        ax.annotate(
            text,
            (1.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(-limit, limit)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Mean of the two (mmHg)")
    ax.set_ylabel("Estimate − reference (mmHg)")
    outside = 100.0 * np.mean(np.abs(diff) > limit)
    if outside > MIN_SHARE_NOTED:
        ax.annotate(
            f"{outside:.1f}% beyond ±{limit:.0f} mmHg not shown",
            (0.0, 0.0),
            xycoords="axes fraction",
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color=INK_SECONDARY,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
    return mesh, (float(bias), float(lower), float(upper))


def error_by_band(
    ax: Axes, reference: ArrayLike, estimate: ArrayLike, edges: Sequence[float], title: str
) -> list[tuple]:
    """Median and interquartile range of the error (estimate − reference) by band of the reference; shows
    underestimation of high pressures."""
    reference, estimate = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    diff = estimate - reference
    band = np.digitize(reference, edges)
    labels = (
        [f"<{edges[0]:.0f}"] + [f"{a:.0f}–{b - 1:.0f}" for a, b in itertools.pairwise(edges)] + [f"≥{edges[-1]:.0f}"]
    )
    rows = []
    for k, label in enumerate(labels):
        values = diff[band == k]
        if len(values) == 0:
            continue
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        rows.append((label, q50, q25, q75, 100.0 * len(values) / len(diff)))
        ax.plot([k, k], [q25, q75], color=SERIES, linewidth=1.4, solid_capstyle="round")
        ax.plot(k, q50, "o", color=SERIES, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.annotate(
            signed(q50, 0), (k, q75), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, color=INK
        )
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Reference (mmHg)")
    ax.set_ylabel("Estimate − reference (mmHg)")
    ax.grid(axis="x", visible=False)
    return rows
