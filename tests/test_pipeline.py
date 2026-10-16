"""Building the intervals (shared within and across datasets): bands, signal-quality tertiles, each method."""

import numpy as np
import pandas as pd
import pytest

from ppg_bp.conformal.pipeline import (
    METHODS,
    band,
    intervals,
    quantile_column,
    sqi_tertile,
)
from ppg_bp.evaluation.coverage import per_subject
from ppg_bp.models.gbm import output_labels, quantile_name

SETTINGS = {
    "strata": {"sbp": [90, 100, 120, 140, 160], "dbp": [50, 60, 70, 80, 90]},
    "levels": {0.9: [0.05, 0.95]},
    "min_subjects_per_stratum": 30,
}


def test_band_puts_edges_in_the_upper_band_and_keeps_missing():
    out = band([89.9, 90.0, 119.9, 120.0, 160.0, np.nan], [90, 100, 120, 140, 160])
    assert out.tolist() == ["<90", "90–<100", "100–<120", "120–<140", "≥160", None]


def test_sqi_tertile_uses_the_fixed_cuts():
    assert sqi_tertile([0.5, 0.99, 0.999], {"lower": 0.9894, "upper": 0.9971}).tolist() == ["low", "middle", "high"]


def test_quantile_names_match_between_modules():
    assert quantile_name(0.025) == "q025" and quantile_name(0.95) == "q950"
    assert output_labels([0.05, 0.95]) == ["mean", "q050", "q950"]
    assert quantile_column("sbp", "lightgbm", 0.05) == "sbp_lightgbm_q050"


def make_table(n_subjects, seed):
    """Data whose estimates shrink towards the centre (high pressures are underestimated)."""
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(n_subjects), 60)
    y = rng.normal(118, 20, len(subjects))
    mean = 118 + 0.4 * (y - 118) + rng.normal(0, 8, len(subjects))
    return pd.DataFrame(
        {
            "subjectid": subjects,
            "sbp": y,
            "sbp_m_mean": mean,
            "sbp_m_q050": mean - 20,
            "sbp_m_q950": mean + 20,
            "sqi_tertile": rng.choice(["low", "middle", "high"], len(subjects)),
        }
    )


@pytest.mark.parametrize("method", METHODS)
def test_every_method_reaches_nominal_coverage_overall(method):
    cal, test = make_table(300, 0), make_table(300, 1)
    lower, upper, corrections = intervals(cal, test, "sbp", "m", method, 0.9, SETTINGS)
    coverage = per_subject(test["sbp"], lower, upper, test["subjectid"])["covered"].mean()
    assert coverage == pytest.approx(0.90, abs=0.03)
    assert None in corrections and np.all(upper > lower)


def test_fixed_width_intervals_miss_high_values_from_below():
    """With shrinking estimates, fixed-width intervals miss high values above them (the main finding in miniature)."""
    cal, test = make_table(300, 0), make_table(300, 1)
    lower, upper, _ = intervals(cal, test, "sbp", "m", "split", 0.9, SETTINGS)
    high = (test["sbp"] >= 150).to_numpy()
    above, below = (test["sbp"] > upper)[high].mean(), (test["sbp"] < lower)[high].mean()
    covered_mid = ((test["sbp"] >= lower) & (test["sbp"] <= upper))[(test["sbp"].between(110, 126)).to_numpy()].mean()
    assert above > 0.3 and below < 0.01
    assert covered_mid > 0.95
