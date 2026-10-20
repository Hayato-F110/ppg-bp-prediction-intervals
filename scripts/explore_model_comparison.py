"""Waveform model (CNN) minus feature model (LightGBM) on the same patients. Exploratory (not in the analysis plan).

Added on 2026-10-15 to answer which model is better; the feature model stays the main model.
All differences are waveform model minus feature model.

1. SBP MAE in the test patients (mmHg), over segments as in the paper. In a bootstrap over patients, the MAE of
   the drawn patients' segments is computed for both models and differenced (the same patients for both).
2. Coverage of nominal 90% SBP intervals (percentage points): per-patient differences with paired_difference
   (ppg_bp.evaluation.coverage), as in the supplementary data (sheet differences_methods). VitalDB test patients
   at true SBP >= 160 and < 90, and all of MIMIC-III; split conformal and CQR.

output: results/review2/model_comparison.csv
usage: uv run python -X utf8 scripts/explore_model_comparison.py
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import intervals, read_predictions
from ppg_bp.constants import SBP_HIGH, SBP_LOW
from ppg_bp.evaluation.coverage import paired_difference

TARGET = "sbp"
MODELS = ("lightgbm", "cnn")


def mae_difference(table, n_boot, seed):
    """Difference in segment-level MAE (CNN minus LightGBM) with a 95% CI from a bootstrap over patients."""
    errors = pd.DataFrame({m: (table[f"{TARGET}_{m}_mean"] - table[TARGET]).abs() for m in MODELS}).assign(
        s=table["subjectid"].to_numpy()
    )
    sums = errors.groupby("s")[list(MODELS)].sum()
    counts = errors.groupby("s").size().to_numpy()
    a, b = sums["cnn"].to_numpy(), sums["lightgbm"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(counts), len(counts))
        boots.append((a[i].sum() - b[i].sum()) / counts[i].sum())
    return {
        "patients": len(counts),
        "feature": b.sum() / counts.sum(),
        "waveform": a.sum() / counts.sum(),
        "difference": (a.sum() - b.sum()) / counts.sum(),
        "ci_low": np.percentile(boots, 2.5),
        "ci_high": np.percentile(boots, 97.5),
    }


def main():
    config = load_config()
    settings = load_settings("conformal")
    cuts = load_settings("sqi_tertiles")["sqi_template"]
    seed, n_boot = config["seed"], settings["bootstrap"]
    results = PROJECT_ROOT / config["paths"]["results"]

    cal, test, ext = (read_predictions(results, name, cuts) for name in ("calibration", "test", "mimic3wdb"))

    rows = [
        {"dataset": "VitalDB test", "stratum": "all", "measure": "MAE (mmHg)", **mae_difference(test, n_boot, seed)}
    ]

    def covered(table, model, method):
        lower, upper, _ = intervals(cal, table, TARGET, model, method, 0.9, settings)
        return ((table[TARGET] >= lower) & (table[TARGET] <= upper)).to_numpy().astype(float)

    for dataset, table, strata in (
        ("VitalDB test", test, {"≥160": test[TARGET] >= SBP_HIGH, "<90": test[TARGET] < SBP_LOW}),
        ("MIMIC-III", ext, {"all": pd.Series(True, index=ext.index)}),
    ):
        for method in ("split", "cqr"):
            cov = {m: covered(table, m, method) for m in MODELS}
            for stratum, selected in strata.items():
                mask = selected.to_numpy()
                subjects = table["subjectid"].to_numpy()[mask]
                per = pd.DataFrame({m: cov[m][mask] for m in MODELS}).groupby(subjects).mean()
                rows.append(
                    {
                        "dataset": dataset,
                        "stratum": stratum,
                        "measure": f"coverage {method}",
                        "feature": per["lightgbm"].mean(),
                        "waveform": per["cnn"].mean(),
                        **paired_difference(cov["cnn"][mask], cov["lightgbm"][mask], subjects, n_boot, seed),
                    }
                )

    out = results / "review2" / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
