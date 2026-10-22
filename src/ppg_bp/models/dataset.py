"""Loading of features and reference values, sampling of training segments, and the validation split within training
subjects.

Test and calibration subjects are not loaded when training the models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ppg_bp.config import PROJECT_ROOT

KEYS = ["caseid", "index"]


def load_labels(config: dict, split: str) -> pd.DataFrame:
    """Reference values and attributes (no features) of a split. Rows are in the same order as load_split."""
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    labels = pd.read_parquet(root / "segment_labels.parquet")
    return labels.loc[labels["split"] == split].reset_index(drop=True)


def load_split(config: dict, split: str) -> pd.DataFrame:
    """Reference values, attributes and features of a split, joined."""
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    labels = load_labels(config, split)
    merged_path = root / "features.parquet"
    if merged_path.exists():
        features = pd.read_parquet(merged_path)
    else:  # for testing before features of all cases are computed
        files = [root / "features" / f"{case_id}.parquet" for case_id in labels["caseid"].unique()]
        features = pd.concat([pd.read_parquet(f) for f in files if Path(f).exists()], ignore_index=True)
    table = labels.merge(features, on=KEYS, how="inner", validate="one_to_one").reset_index(drop=True)
    return table.replace([np.inf, -np.inf], np.nan)  # ratios with a near-zero denominator are treated as missing


def load_waveforms(config: dict, table: pd.DataFrame) -> np.ndarray:
    """PPG segments for the rows (caseid, index) of table, in row order (float32)."""
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb" / "segments"
    length = config["segment"]["length_sec"] * config["segment"]["fs_hz"]
    out = np.empty((len(table), length), dtype=np.float32)
    positions = np.arange(len(table))
    for case_id, rows in table.groupby("caseid", sort=False).indices.items():
        with np.load(root / f"{case_id}.npz") as data:
            ppg, index = data["ppg"], data["index"]
        wanted = table["index"].to_numpy()[rows]
        where = np.searchsorted(index, wanted)
        if (where >= len(index)).any() or not np.array_equal(index[where], wanted):
            raise AssertionError("the segments read are not those requested")
        out[positions[rows]] = ppg[where]
    return out


def sample_per_subject(table: pd.DataFrame, max_per_subject: int, seed: int) -> pd.DataFrame:
    """Randomly select at most max_per_subject segments per subject."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table))
    shuffled = table.iloc[order]
    rank = shuffled.groupby("subjectid").cumcount()
    return shuffled[rank < max_per_subject].sort_index()


def inner_validation_subjects(subject_ids: ArrayLike, fraction: float, seed: int) -> np.ndarray:
    """IDs of the training subjects held out for validation."""
    unique = np.sort(np.asarray(pd.unique(pd.Series(subject_ids))))
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(unique)[: round(len(unique) * fraction)])
