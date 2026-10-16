"""Weighted conformal prediction under covariate shift: on synthetic data, the properties follow the theory."""

import numpy as np
import pandas as pd
import pytest

from ppg_bp.conformal.core import conformal_quantile, split_scores, subject_weights
from ppg_bp.conformal.pipeline import quantile_column
from ppg_bp.conformal.shift import domain_weights, patient_effective_share, weighted_conformal

TARGET, MODEL, LEVEL = "sbp", "m", 0.9
QUANTILES = (0.05, 0.95)


def make_table(n_subjects, shift, seed):
    """Segments whose error grows with |x|; x is shifted by `shift` in the target data. Quantile columns are exact."""
    rng = np.random.default_rng(seed)
    # unequal counts, so that no cumulative weight falls exactly on the quantile level (a tie decided by rounding)
    subjects = np.repeat(np.arange(n_subjects), rng.integers(5, 16, size=n_subjects))
    x = rng.normal(shift, 1.0, size=len(subjects))
    sd = 1.0 + np.abs(x)
    mean = 120.0 + 5.0 * x
    table = pd.DataFrame({"subjectid": subjects, "x": x, TARGET: mean + rng.normal(size=len(x)) * sd})
    table[f"{TARGET}_{MODEL}_mean"] = mean
    for q in QUANTILES:
        z = 1.6448536269514722 * (1 if q > 0.5 else -1)  # standard normal quantile of 0.95 / 0.05
        table[quantile_column(TARGET, MODEL, q)] = mean + z * sd
    return table


def test_effective_share_is_one_for_equal_patients_and_small_for_one_dominant_patient():
    subjects = np.repeat(np.arange(10), 5)
    assert patient_effective_share(subjects, np.ones(len(subjects))) == pytest.approx(1.0)
    weights = np.where(subjects == 0, 1000.0, 1.0)
    assert patient_effective_share(subjects, weights) < 0.15


def test_unit_ratios_give_the_unweighted_patient_quantile():
    cal = make_table(200, 0.0, seed=1)
    ext = make_table(50, 0.0, seed=2)
    rows = weighted_conformal(
        cal, ext, (np.ones(len(cal)), np.ones(len(ext))), TARGET, MODEL, LEVEL, QUANTILES, (50, 0)
    )
    split = next(r for r in rows if r["method"] == "weighted_split")
    scores = split_scores(cal[TARGET], cal[f"{TARGET}_{MODEL}_mean"])
    base = subject_weights(cal["subjectid"])
    q = conformal_quantile(scores, base / base.sum() * cal["subjectid"].nunique(), 1 - LEVEL)
    assert split["width_median"] == pytest.approx(2 * q)
    assert split["infinite_share"] == 0.0


def test_true_density_ratios_restore_coverage_under_shift():
    shift = 1.0
    cal = make_table(400, 0.0, seed=3)
    ext = make_table(400, shift, seed=4)

    def ratio(x):
        """Ratio of the N(shift, 1) density to the N(0, 1) density."""
        return np.exp(shift * x - shift**2 / 2)

    weighted = weighted_conformal(
        cal, ext, (ratio(cal["x"]).to_numpy(), ratio(ext["x"]).to_numpy()), TARGET, MODEL, LEVEL, QUANTILES, (50, 0)
    )
    plain = weighted_conformal(
        cal, ext, (np.ones(len(cal)), np.ones(len(ext))), TARGET, MODEL, LEVEL, QUANTILES, (50, 0)
    )
    w = next(r for r in weighted if r["method"] == "weighted_split")
    p = next(r for r in plain if r["method"] == "weighted_split")
    assert p["coverage"] < LEVEL - 0.03  # the shift moves target points to where errors are larger
    assert abs(w["coverage"] - LEVEL) < 0.03
    assert w["effective_patient_share"] < 1.0


@pytest.mark.parametrize(("shift", "low", "high"), [(0.0, 0.4, 0.6), (2.0, 0.85, 1.0)])
def test_domain_classifier_auc_follows_the_shift(shift, low, high):
    train = make_table(150, 0.0, seed=5)
    cal = make_table(100, 0.0, seed=6)
    ext = make_table(150, shift, seed=7)
    ratio_cal, ratio_ext, auc, gain = domain_weights(train, cal, ext, ["x"], seed=0)
    assert low < auc < high
    assert ratio_cal.shape == (len(cal),)
    assert ratio_ext.shape == (len(ext),)
    assert np.all((ratio_ext >= 1e-3) & (ratio_ext <= 1e3))
    assert gain.sum() == pytest.approx(1.0)
