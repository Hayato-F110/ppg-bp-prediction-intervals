"""Split by subject, so that no subject appears in more than one of training, calibration and test.

In VitalDB one subject can have several operations (cases), so we split by subject ID, not by case ID.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

SPLIT_NAMES = ("train", "calibration", "test")


def split_subjects(subject_ids: ArrayLike, ratios: dict, seed: int) -> pd.Series:
    """Assign a split name to each subject ID.

    Args:
        subject_ids: subject IDs (duplicates allowed, so a per-case column can be passed as is)
        ratios: ratios such as {"train": 0.6, "calibration": 0.2, "test": 0.2}
        seed: random seed

    Returns:
        Series indexed by subject ID with the split name as value
    """
    unique = np.sort(np.asarray(pd.unique(pd.Series(subject_ids))))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n = len(shuffled)
    n_train = round(n * ratios["train"])
    n_cal = round(n * ratios["calibration"])
    labels = np.empty(n, dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_cal] = "calibration"
    labels[n_train + n_cal :] = "test"
    return pd.Series(labels, index=pd.Index(shuffled, name="subjectid"), name="split").sort_index()


def assert_no_subject_overlap(table: pd.DataFrame, subject_col: str = "subjectid", split_col: str = "split") -> None:
    """Raise if any subject appears in more than one split."""
    per_subject = table.groupby(subject_col)[split_col].nunique()
    leaked = per_subject[per_subject > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} subjects appear in more than one split")
