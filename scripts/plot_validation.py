"""Bland-Altman plots and errors by BP band for the validation patients (within the training patients).

usage: uv run python -X utf8 scripts/plot_validation.py
input: results/feature_models/validation_predictions.parquet, results/waveform_model/validation_predictions.parquet (if
present)
output: results/figures/phase3_bland_altman_<model>, phase3_error_by_band_<model> (.png, .svg, .pdf and .eps),
phase3_error_by_band.csv
"""

import matplotlib.pyplot as plt
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.evaluation.plots import (
    PAGE_WIDTH,
    TARGET_LABELS,
    apply_style,
    bland_altman,
    error_by_band,
    save_figure,
)

# cut points of the strata (analysis plan, Section 4.3; configs/conformal.yaml)
BAND_EDGES = {target: load_settings("conformal")["strata"][target] for target in ("sbp", "dbp")}
MODELS = {
    "lightgbm": ("feature_models", "lightgbm_ppg_only", "LightGBM on PPG features"),
    "cnn": ("waveform_model", "cnn_waveform", "1-D CNN on raw PPG"),
}


def main():
    config = load_config()
    results = PROJECT_ROOT / config["paths"]["results"]
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    band_rows = []
    for key, (folder, suffix, _name) in MODELS.items():
        path = results / folder / "validation_predictions.parquet"
        if not path.exists():
            print(f"{key}: no estimates, skipped")
            continue
        table = pd.read_parquet(path)
        fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.5), constrained_layout=True)
        mesh = None
        for ax, target in zip(axes, ("sbp", "dbp", "mbp"), strict=True):
            mesh, _ = bland_altman(ax, table[target], table[f"{target}_{suffix}"], TARGET_LABELS[target])
        if mesh is None:
            raise AssertionError("no panel was drawn")
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.01, label="Segments per bin (log)")
        colorbar.outline.set_visible(False)
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)  # draw the colour bar as vectors in the SVG too
        save_figure(fig, out_dir / f"phase3_bland_altman_{key}.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8), constrained_layout=True)
        for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
            rows = error_by_band(
                ax, table[target], table[f"{target}_{suffix}"], BAND_EDGES[target], TARGET_LABELS[target]
            )
            band_rows += [
                {"model": key, "target": target, "band": b, "median": m, "q25": lo, "q75": hi, "share_pct": share}
                for b, m, lo, hi, share in rows
            ]
        save_figure(fig, out_dir / f"phase3_error_by_band_{key}.png")
        plt.close(fig)
        print(f"{key}: saved the figures")
    if band_rows:
        pd.DataFrame(band_rows).to_csv(out_dir / "phase3_error_by_band.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
