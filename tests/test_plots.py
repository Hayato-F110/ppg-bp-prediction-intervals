"""Figure functions: they return the expected values and write files."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppg_bp.evaluation.plots import apply_style, bland_altman, error_by_band


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    reference = rng.normal(120, 20, 5000)
    return reference, reference + rng.normal(2.0, 10.0, 5000)


def test_bland_altman_returns_bias_and_limits(data, tmp_path):
    reference, estimate = data
    apply_style()
    fig, ax = plt.subplots()
    _, (bias, lower, upper) = bland_altman(ax, reference, estimate, "SBP")
    assert bias == pytest.approx(2.0, abs=0.5)
    assert upper - lower == pytest.approx(2 * 1.96 * 10.0, rel=0.05)
    fig.savefig(tmp_path / "ba.png")
    plt.close(fig)
    assert (tmp_path / "ba.png").stat().st_size > 1000


def test_error_by_band_covers_all_segments(data):
    reference, _ = data
    estimate = 0.5 * reference + 60.0  # regression to the mean: high pressures underestimated, low ones overestimated
    fig, ax = plt.subplots()
    rows = error_by_band(ax, reference, estimate, [90, 100, 120, 140, 160], "SBP")
    plt.close(fig)
    assert sum(share for *_, share in rows) == pytest.approx(100.0)
    medians = [median for _, median, *_ in rows]
    assert medians[0] > 0 > medians[-1]
    assert medians == sorted(medians, reverse=True)
