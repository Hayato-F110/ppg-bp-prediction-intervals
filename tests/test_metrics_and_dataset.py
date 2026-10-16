"""Accuracy measures and the selection of training data."""

import numpy as np
import pandas as pd
import pytest

from ppg_bp.evaluation.metrics import bhs_grade, point_metrics, subject_level_mae
from ppg_bp.models.dataset import inner_validation_subjects, sample_per_subject


def test_point_metrics_on_known_errors():
    y = np.zeros(4)
    out = point_metrics(y, y + np.array([-4.0, 4.0, 12.0, 20.0]))
    assert out["mae"] == pytest.approx(10.0)
    assert out["me"] == pytest.approx(8.0)
    assert out["sd"] == pytest.approx(np.std([-4, 4, 12, 20], ddof=1))
    assert (out["within5"], out["within10"], out["within15"]) == (50.0, 50.0, 75.0)
    assert out["bhs"] == "D"
    assert out["aami_error_ok"] is False


def test_bhs_grades_at_the_boundaries():
    assert bhs_grade(60, 85, 95) == "A"
    assert bhs_grade(59.9, 85, 95) == "B"
    assert bhs_grade(50, 75, 89.9) == "C"
    assert bhs_grade(39.9, 65, 85) == "D"


def test_aami_error_criterion():
    rng = np.random.default_rng(0)
    y = rng.normal(120, 20, 5000)
    assert point_metrics(y, y + rng.normal(1.0, 6.0, 5000))["aami_error_ok"] is True
    assert point_metrics(y, y + rng.normal(1.0, 9.0, 5000))["aami_error_ok"] is False


def test_subject_level_mae_weights_subjects_equally():
    y = np.zeros(11)
    pred = np.array([10.0] * 10 + [0.0])  # subject 1: 10 segments with error 10; subject 2: one segment with error 0
    out = subject_level_mae(y, pred, ["a"] * 10 + ["b"])
    assert out["subjects"] == 2
    assert out["subject_mae_mean"] == pytest.approx(5.0)


def test_sample_per_subject_caps_each_subject():
    table = pd.DataFrame({"subjectid": [1] * 50 + [2] * 5, "x": range(55)})
    out = sample_per_subject(table, 10, seed=0)
    assert out["subjectid"].value_counts().to_dict() == {1: 10, 2: 5}
    assert out.equals(sample_per_subject(table, 10, seed=0))
    assert not out.equals(sample_per_subject(table, 10, seed=1))


def test_inner_validation_is_by_subject_and_reproducible():
    subjects = np.repeat(np.arange(100), 3)
    chosen = inner_validation_subjects(subjects, 0.2, seed=0)
    assert len(chosen) == 20
    assert np.array_equal(chosen, inner_validation_subjects(subjects, 0.2, seed=0))
