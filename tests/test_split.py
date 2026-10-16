"""Patient-level split: the most important constraint of the study, so leakage is checked directly."""

import numpy as np
import pandas as pd
import pytest

from ppg_bp.data.split import SPLIT_NAMES, assert_no_subject_overlap, split_subjects

RATIOS = {"train": 0.6, "calibration": 0.2, "test": 0.2}


def make_cases(n_subjects=500, seed=0):
    """A case table in which some subjects have several cases."""
    rng = np.random.default_rng(seed)
    n_cases_each = rng.choice([1, 1, 1, 1, 2, 5, 12], size=n_subjects)
    subject = np.repeat(np.arange(1000, 1000 + n_subjects), n_cases_each)
    return pd.DataFrame({"caseid": np.arange(len(subject)), "subjectid": subject})


def test_every_subject_gets_exactly_one_split():
    cases = make_cases()
    split = split_subjects(cases["subjectid"], RATIOS, seed=1)
    assert split.index.is_unique
    assert set(split.index) == set(cases["subjectid"])
    assert set(split.unique()) == set(SPLIT_NAMES)


def test_cases_of_same_subject_stay_together():
    cases = make_cases()
    # pyright takes cases["subjectid"] for a DataFrame here; it is a Series, whose map accepts a Series
    cases["split"] = cases["subjectid"].map(split_subjects(cases["subjectid"], RATIOS, seed=1))  # pyright: ignore[reportArgumentType]
    assert cases.groupby("subjectid")["split"].nunique().max() == 1
    assert_no_subject_overlap(cases)
    groups = {name: set(g["subjectid"]) for name, g in cases.groupby("split")}
    assert not groups["train"] & groups["calibration"]
    assert not groups["train"] & groups["test"]
    assert not groups["calibration"] & groups["test"]


def test_ratios_are_respected_at_subject_level():
    split = split_subjects(make_cases(1000)["subjectid"], RATIOS, seed=1)
    share = split.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.6, abs=0.001)
    assert share["calibration"] == pytest.approx(0.2, abs=0.001)
    assert share["test"] == pytest.approx(0.2, abs=0.001)


def test_same_seed_gives_same_split_regardless_of_row_order():
    cases = make_cases()
    a = split_subjects(cases["subjectid"], RATIOS, seed=7)
    b = split_subjects(cases["subjectid"].sample(frac=1, random_state=3), RATIOS, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_different_seed_gives_different_split():
    cases = make_cases()
    a = split_subjects(cases["subjectid"], RATIOS, seed=7)
    b = split_subjects(cases["subjectid"], RATIOS, seed=8)
    assert (a != b).any()


def test_overlap_check_detects_leak():
    leaked = pd.DataFrame({"subjectid": [1, 1, 2], "split": ["train", "test", "train"]})
    with pytest.raises(AssertionError):
        assert_no_subject_overlap(leaked)
