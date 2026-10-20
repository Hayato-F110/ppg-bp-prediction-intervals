"""SHAP: which features drive the point estimate and the interval width.

The models are the LightGBM SBP models of the main analysis (results/conformal/models/). The width is that of the
uncorrected nominal 90% CQR interval: the 95% quantile model minus the 5% quantile model. The segments are drawn at
random from the validation patients within the training patients, who were not used to fit the models.
The detection of the e wave depends on the pulse rate, so the importance is also given within tertiles of the pulse
interval.

usage: uv run python -X utf8 scripts/explain_models.py
output: results/shap/importance.csv, importance_by_pulse_interval.csv, direction.csv, results/figures/phase6_shap
"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.evaluation.plots import MIN_TEXT, PAGE_WIDTH, apply_style, half_up, save_figure
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import inner_validation_subjects, load_split

MIN_VALUES, MIN_DISTINCT = 100, 5  # a rank correlation needs enough values and distinct values

N_SEGMENTS = 30000
TARGET = "sbp"
COLUMNS = list(FEATURE_NAMES)


# Names of the features in the figure (definitions as in Table S2)
READABLE = {
    "f_a": "f/a",
    "e_a": "e/a",
    "d_a": "d/a",
    "c_a": "c/a",
    "b_a": "b/a",
    "b_a_iqr": "IQR of b/a",
    "aging_index": "ageing index",
    "apg_index": "APG index",
    "t_ab": "time a to b",
    "t_ac": "time a to c",
    "t_ad": "time a to d",
    "t_ae": "time a to e",
    "t_af": "time a to f",
    "t_be": "time b to e",
    "t_de": "time d to e",
    "t_ae_rel": "time a to e / beat",
    "crest_time": "crest time",
    "crest_time_rel": "crest time / beat",
    "upslope_time": "upslope time",
    "max_slope": "steepest upslope",
    "width50_rel": "relative pulse width",
    "dw_sw50": "width after / before peak",
    "sw25": "width before peak, 25%",
    "dw25": "width after peak, 25%",
    "sw50": "width before peak, 50%",
    "dw50": "width after peak, 50%",
    "sw75": "width before peak, 75%",
    "dw75": "width after peak, 75%",
    "area_ratio": "area ratio",
    "t_peak_e": "time peak to e",
    "height_e": "height at e",
    "ibi_median": "pulse interval",
    "ibi_sd": "SD of pulse interval",
    "ibi_rmssd": "RMSSD of pulse interval",
    "ibi_cv": "CV of pulse interval",
    "sqi_template": "template-matching quality",
    "sqi_skewness": "skewness",
    "sqi_kurtosis": "kurtosis",
    "n_beats": "number of beats",
    "frac_e_found": "share of beats with e",
    "frac_cd_found": "share of beats with c, d",
    "frac_cd_distinct": "share with distinct c, d",
}


def shap_values(results, x) -> dict:
    """SHAP values of the mean model and of the 5% and 95% quantile models. SHAP values add up, so the contribution
    to the width of the uncorrected nominal 90% CQR interval is the difference of the two quantile models."""
    values = {}
    for label in ("mean", "q050", "q950"):
        booster = lgb.Booster(model_file=str(results / "conformal" / "models" / f"{TARGET}_{label}.txt"))
        values[label] = shap.TreeExplainer(booster).shap_values(x)
        print(f"{label} done", flush=True)
    values["width"] = values["q950"] - values["q050"]
    return values


def importance_table(values) -> pd.DataFrame:
    importance = pd.DataFrame({name: np.abs(v).mean(axis=0) for name, v in values.items()}, index=COLUMNS)
    importance["share_mean"] = importance["mean"] / importance["mean"].sum()
    importance["share_width"] = importance["width"] / importance["width"].sum()
    return importance


def direction_table(x, values) -> pd.DataFrame:
    """Rank correlation between the value of a feature and its SHAP value (direction of the effect), without the
    missing values."""
    rows = []
    for k, name in enumerate(COLUMNS):
        ok = x[name].notna().to_numpy()
        if ok.sum() < MIN_VALUES or x[name][ok].nunique() < MIN_DISTINCT:
            continue
        rows.append(
            {
                "feature": name,
                "missing_share": 1 - ok.mean(),
                "rho_mean": spearmanr(x[name][ok], values["mean"][ok, k])[0],
                "rho_width": spearmanr(x[name][ok], values["width"][ok, k])[0],
                "shap_width_when_missing": values["width"][~ok, k].mean() if (~ok).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def importance_by_pulse_interval(sample, values) -> pd.DataFrame:
    """Importance within tertiles of the pulse interval (the e-wave detection depends on the pulse rate)."""
    tertile = pd.qcut(sample["ibi_median"], 3, labels=["short", "middle", "long"])
    by_ibi = {}
    for group in ("short", "middle", "long"):
        mask = (tertile == group).to_numpy()
        for name in ("mean", "width"):
            by_ibi[(name, group)] = np.abs(values[name][mask]).mean(axis=0)
    return pd.DataFrame(by_ibi, index=COLUMNS)


def plot_importance(importance, out_path) -> None:
    """Supplementary figure: the 12 most important features for the estimate and for the width."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2), constrained_layout=True)
    for ax, column, title in (
        (axes[0], "mean", "Point estimate of SBP"),
        (axes[1], "width", "Width of the 90% interval"),
    ):
        top = importance[column].sort_values(ascending=False).head(12)[::-1]
        ax.barh(range(len(top)), top.to_numpy(), color="#2a78d6", height=0.6)
        ax.set_yticks(range(len(top)), [READABLE.get(name, name) for name in top.index])
        for k, value in enumerate(top.to_numpy()):
            ax.annotate(
                half_up(value, 2), (value, k), xytext=(2, 0), textcoords="offset points", va="center", fontsize=MIN_TEXT
            )
        ax.set_xlabel("Mean absolute SHAP value (mmHg)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, top.max() * 1.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, panels=axes)


def main():
    config = load_config()
    settings = load_settings("feature_models")
    seed = config["seed"]
    results = PROJECT_ROOT / config["paths"]["results"]
    out_dir = results / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(config, "train")
    held_out = inner_validation_subjects(train["subjectid"], settings["validation_fraction"], seed)
    sample = train[train["subjectid"].isin(held_out)].sample(n=N_SEGMENTS, random_state=seed)
    x = sample[COLUMNS]
    print(f"{len(sample)} segments from {sample['subjectid'].nunique()} validation patients", flush=True)

    values = shap_values(results, x)
    importance = importance_table(values)
    importance.sort_values("mean", ascending=False).to_csv(out_dir / "importance.csv", float_format="%.6f")
    direction_table(x, values).to_csv(out_dir / "direction.csv", index=False, float_format="%.6f")
    importance_by_pulse_interval(sample, values).to_csv(
        out_dir / "importance_by_pulse_interval.csv", float_format="%.6f"
    )
    plot_importance(importance, results / "figures" / "phase6_shap.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
