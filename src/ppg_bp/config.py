"""Loading of the experiment settings (YAML)."""

from pathlib import Path

import yaml

TOLERANCE = 1e-9  # rounding allowed in the sum of the split ratios

# The package is installed in editable mode from src/ppg_bp, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    """Load a settings file. A relative path is resolved from the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ratios = config["split"]
    total = ratios["train"] + ratios["calibration"] + ratios["test"]
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    return config


def load_settings(name: str) -> dict:
    """Load configs/<name>.yaml (for example "conformal" or "sqi_tertiles")."""
    with (PROJECT_ROOT / "configs" / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)
