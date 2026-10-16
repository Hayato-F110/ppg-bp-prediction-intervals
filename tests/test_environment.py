"""Environment check: the libraries used by the code import, and the settings load."""

import importlib

import pytest

from ppg_bp.config import load_config

LIBRARIES = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "sklearn",
    "lightgbm",
    "shap",
    "matplotlib",
    "vitaldb",
    "wfdb",
    "yaml",
    "torch",
    "optuna",
]


@pytest.mark.parametrize("name", LIBRARIES)
def test_library_imports(name):
    importlib.import_module(name)


def test_base_config_loads():
    config = load_config()
    assert isinstance(config["seed"], int)
    assert config["segment"]["length_sec"] == 10
    assert config["coverage_levels"] == [0.80, 0.90, 0.95]


def test_split_ratios_must_sum_to_one(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("split: {train: 0.6, calibration: 0.2, test: 0.3}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
