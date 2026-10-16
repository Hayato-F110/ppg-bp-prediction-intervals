"""Conformal prediction and coverage: on synthetic data, the properties follow the theory."""

import numpy as np
import pytest

from ppg_bp.conformal.core import (
    calibrate,
    conformal_quantile,
    cqr_interval,
    cqr_scores,
    split_interval,
    split_scores,
    subject_weights,
)
from ppg_bp.evaluation.coverage import evaluate, per_subject, summarise


def make_data(n_subjects, seed, sizes=(20, 200)):
    """Data in which the size of the error and the number of segments differ between subjects."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(*sizes, size=n_subjects)
    subjects = np.repeat(np.arange(n_subjects), counts)
    noise = np.repeat(rng.uniform(5.0, 20.0, size=n_subjects), counts)
    y = 120.0 + rng.normal(size=len(subjects)) * noise
    return y, np.full(len(y), 120.0), subjects, noise


def test_quantile_matches_the_textbook_rule_with_equal_weights():
    scores = np.arange(1.0, 100.0)  # 99 scores
    # with equal weights: the ceil((n + 1)(1 − alpha))-th smallest score; the 90th for n = 99, alpha = 0.1
    assert conformal_quantile(scores, np.ones(99), 0.1) == 90.0


def test_quantile_is_infinite_when_calibration_is_too_small():
    assert conformal_quantile([1.0, 2.0, 3.0], np.ones(3), 0.1) == float("inf")


def test_subject_weights_sum_to_one_per_subject():
    weights = subject_weights(["a", "a", "a", "b"])
    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_split_conformal_reaches_nominal_subject_level_coverage():
    """If calibration and test subjects come from the same distribution, subject-level coverage is near nominal."""
    coverage = []
    for seed in range(20):
        y_cal, pred_cal, subj_cal, _ = make_data(300, seed)
        y_test, pred_test, subj_test, _ = make_data(300, 1000 + seed)
        corrections = calibrate(split_scores(y_cal, pred_cal), subj_cal, alpha=0.1)
        lower, upper = split_interval(pred_test, corrections)
        coverage.append(summarise(per_subject(y_test, lower, upper, subj_test), n_boot=10)["coverage"])
    assert np.mean(coverage) == pytest.approx(0.90, abs=0.01)


def test_subject_weighting_matters_when_long_records_are_noisier():
    """If subjects with more segments have larger errors, calibration that weights segments equally gives intervals too
    wide per subject."""
    rng = np.random.default_rng(0)
    counts = np.where(np.arange(400) % 2 == 0, 20, 400)
    subjects = np.repeat(np.arange(400), counts)
    scale = np.repeat(np.where(counts == 20, 5.0, 20.0), counts)
    scores = np.abs(rng.normal(size=len(subjects)) * scale)
    by_subject = calibrate(scores, subjects, 0.1)[None]
    by_segment = calibrate(scores, subjects, 0.1, equal_subjects=False)[None]
    assert by_segment > by_subject * 1.1


def test_cqr_intervals_adapt_and_cover():
    y_cal, _, subj_cal, noise_cal = make_data(300, 1)
    y_test, _, subj_test, noise_test = make_data(300, 2)
    # quantile estimates whose width varies with the input but is slightly too narrow
    lo_cal, hi_cal = 120 - 1.2 * noise_cal, 120 + 1.2 * noise_cal
    lo_test, hi_test = 120 - 1.2 * noise_test, 120 + 1.2 * noise_test
    corrections = calibrate(cqr_scores(y_cal, lo_cal, hi_cal), subj_cal, alpha=0.1)
    lower, upper = cqr_interval(lo_test, hi_test, corrections)
    result = summarise(per_subject(y_test, lower, upper, subj_test), n_boot=200, nominal=0.9)
    assert result["coverage"] == pytest.approx(0.90, abs=0.02)
    assert corrections[None] > 0  # too narrow, so the correction widens them
    assert np.corrcoef(upper - lower, noise_test)[0, 1] > 0.99
    assert result["subjects_below_minus20"] < 0.02


def test_mondrian_repairs_group_coverage_and_falls_back_for_small_groups():
    y_cal, pred_cal, subj_cal, noise_cal = make_data(400, 3)
    y_test, pred_test, subj_test, noise_test = make_data(400, 4)
    group_cal, group_test = np.where(noise_cal > 12.5, "noisy", "quiet"), np.where(noise_test > 12.5, "noisy", "quiet")
    scores = split_scores(y_cal, pred_cal)
    plain = split_interval(pred_test, calibrate(scores, subj_cal, 0.1))
    mondrian = split_interval(pred_test, calibrate(scores, subj_cal, 0.1, groups=group_cal), group_test)
    strata = {"group": group_test}
    before = evaluate(y_test, *plain, subj_test, strata, n_boot=50).set_index("stratum")["coverage"]
    after = evaluate(y_test, *mondrian, subj_test, strata, n_boot=50).set_index("stratum")["coverage"]
    assert before["noisy"] < 0.86 and before["quiet"] > 0.94  # a fixed width is biased between strata
    assert after["noisy"] == pytest.approx(0.90, abs=0.03) and after["quiet"] == pytest.approx(0.90, abs=0.03)
    tiny = calibrate(scores, subj_cal, 0.1, groups=np.where(subj_cal < 5, "rare", "common"))
    assert tiny["rare"] == tiny[None]  # strata with fewer than 30 subjects use the overall correction


def test_miss_rates_separate_upper_and_lower_misses():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    table = per_subject(y, lower=np.full(4, 15.0), upper=np.full(4, 35.0), subjects=["a", "a", "b", "b"])
    assert table.loc["a", ["covered", "above", "below"]].tolist() == [0.5, 0.0, 0.5]
    assert table.loc["b", ["covered", "above", "below"]].tolist() == [0.5, 0.5, 0.0]
    result = summarise(table, n_boot=100)
    assert result["coverage"] + result["miss_above"] + result["miss_below"] == pytest.approx(1.0)
    assert result["coverage_lo"] <= result["coverage"] <= result["coverage_hi"]


def test_bootstrap_interval_contains_the_truth_for_exchangeable_data():
    y_cal, pred_cal, subj_cal, _ = make_data(300, 5)
    y_test, pred_test, subj_test, _ = make_data(600, 6)
    lower, upper = split_interval(pred_test, calibrate(split_scores(y_cal, pred_cal), subj_cal, 0.1))
    result = summarise(per_subject(y_test, lower, upper, subj_test), n_boot=2000)
    assert result["coverage_lo"] < 0.90 < result["coverage_hi"]
    assert result["coverage_hi"] - result["coverage_lo"] < 0.05
