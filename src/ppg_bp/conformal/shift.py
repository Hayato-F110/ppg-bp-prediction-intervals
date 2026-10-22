"""Intervals on a new dataset: weighted conformal prediction, recalibration and re-weighting to another SBP.

Weighted conformal prediction under covariate shift (Tibshirani et al. 2019): a classifier that tells the source
(VitalDB) from the target dataset estimates the density ratio of each segment. Calibration scores are weighted by
(equal weight per patient) x (density ratio), and each target segment gets its own weighted quantile.
"""

from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score

from ppg_bp.conformal.core import calibrate, cqr_interval, cqr_scores, split_interval, split_scores, subject_weights
from ppg_bp.conformal.pipeline import quantile_column
from ppg_bp.constants import SBP_HIGH, SBP_LOW
from ppg_bp.evaluation.coverage import per_subject, summarise

SBP_SHARE_EDGES = np.arange(40, 261, 5)  # mmHg: bins for re-weighting one SBP distribution to another


def domain_training_sample(train: pd.DataFrame, per_patient: int, seed: int) -> pd.DataFrame:
    """Source segments for fitting the domain classifier: per_patient segments drawn per training patient (with
    replacement), each kept once."""
    sample = train.groupby("subjectid").sample(n=per_patient, replace=True, random_state=seed)
    # cast: the type information of pandas does not tell that sample() of a grouped DataFrame is a DataFrame
    return cast("pd.DataFrame", sample).drop_duplicates(subset=["caseid", "index"])


def domain_weights(
    train: pd.DataFrame,
    cal_features: pd.DataFrame,
    ext_features: pd.DataFrame,
    columns: list[str],
    seed: int,
    params: dict,
    clip: tuple[float, float] = (1e-3, 1e3),
) -> tuple[np.ndarray, np.ndarray, float, pd.Series]:
    """Density ratios from a classifier of source against target.

    The target patients are split in two; a classifier fitted on one half is applied to the other. On the source
    side the classifier is fitted on training patients and applied to the calibration patients, so no weighted
    point is used for fitting. params: settings of the LightGBM classifier (domain_classifier in
    configs/conformal.yaml). Returns (ratio of each calibration segment, ratio of each target segment,
    AUC on unseen data, share of the gain of each feature).
    """
    rng = np.random.default_rng(seed)
    subjects = ext_features["subjectid"].unique()
    fold = pd.Series(rng.permutation(len(subjects)) % 2, index=subjects).loc[ext_features["subjectid"]].to_numpy()
    ratio_cal, ratio_ext = np.zeros(len(cal_features)), np.zeros(len(ext_features))
    aucs, gains = [], []
    for k in (0, 1):
        fit_ext = ext_features[fold == k]
        # cast: pandas does not tell the type checker that a list of columns gives a DataFrame
        x = cast("pd.DataFrame", pd.concat([train[columns], fit_ext[columns]], ignore_index=True))
        label = np.r_[np.zeros(len(train)), np.ones(len(fit_ext))]
        model = lgb.LGBMClassifier(**params, random_state=seed)
        model.fit(x, label)
        held = ext_features[fold != k]
        p_cal = np.asarray(model.predict_proba(cast("pd.DataFrame", cal_features[columns])))[:, 1]
        p_held = np.asarray(model.predict_proba(cast("pd.DataFrame", held[columns])))[:, 1]
        aucs.append(roc_auc_score(np.r_[np.zeros(len(p_cal)), np.ones(len(p_held))], np.r_[p_cal, p_held]))
        ratio_cal += np.clip(p_cal / (1 - p_cal), *clip) / 2  # calibration segments: mean of the two classifiers
        ratio_ext[fold != k] = np.clip(p_held / (1 - p_held), *clip)
        gains.append(pd.Series(model.booster_.feature_importance("gain"), index=columns))
    gain = pd.concat(gains, axis=1).sum(axis=1)
    return ratio_cal, ratio_ext, float(np.mean(aucs)), (gain / gain.sum()).sort_values(ascending=False)


def sbp_shares(values: ArrayLike, subjects: ArrayLike, edges: ArrayLike = SBP_SHARE_EDGES) -> np.ndarray:
    """Share of each SBP bin, with equal weight per patient: the distribution that sbp_weights() matches."""
    values, edges = np.asarray(values, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    counts = np.histogram(values, edges, weights=subject_weights(subjects))[0]
    return counts / counts.sum()


def sbp_weights(
    values: ArrayLike, subjects: ArrayLike, shares: np.ndarray, edges: ArrayLike = SBP_SHARE_EDGES
) -> np.ndarray:
    """Weight of each segment (equal weight per patient x ratio of the bin shares), so that the SBP distribution of
    values matches shares. Bins without any segment of values get no weight."""
    values, edges = np.asarray(values, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    own = sbp_shares(values, subjects, edges)
    ratio = np.divide(shares, own, out=np.zeros_like(shares), where=own > 0)
    which = np.clip(np.digitize(values, edges) - 1, 0, len(ratio) - 1)
    return subject_weights(subjects) * ratio[which]


def recalibration(
    ext: pd.DataFrame,
    settings: dict,
    seed: int,
    repeats: int,
    sizes: tuple[int, ...],
    models: tuple[str, ...],
    offset_correction: bool = False,
) -> pd.DataFrame:
    """Recalibrate on a few target patients drawn at random and evaluate on the other target patients.

    For each model and number of patients, the same patients serve split conformal and CQR. With offset_correction,
    each draw is evaluated twice: as it is (shifted False) and with the mean error true − estimate of the drawn
    patients (equal weight per patient) added to the estimates and quantiles first (shifted True).
    """
    target, level = settings["primary_target"], settings["primary_level"]
    q_lo, q_hi = settings["levels"][level]
    rng = np.random.default_rng(seed)
    subjects = ext["subjectid"].unique()
    code = pd.Series(np.arange(len(subjects)), index=subjects)[ext["subjectid"]].to_numpy()
    y = ext[target].to_numpy()
    high, low = y >= SBP_HIGH, y < SBP_LOW
    rows = []
    for model in models:
        mean = ext[f"{target}_{model}_mean"].to_numpy()
        lo, hi = (ext[quantile_column(target, model, q)].to_numpy() for q in (q_lo, q_hi))
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        for size in sizes:
            if size > len(subjects) - 30:
                continue
            for repeat in range(repeats):
                chosen = np.zeros(len(subjects), dtype=bool)
                chosen[rng.choice(len(subjects), size=size, replace=False)] = True
                cal, out = chosen[code], ~chosen[code]
                offset = float(np.average(y[cal] - mean[cal], weights=subject_weights(code[cal])))
                for shifted in (False, True) if offset_correction else (False,):
                    d = offset if shifted else 0.0
                    for method in ("split", "cqr"):
                        if method == "split":
                            q = calibrate(split_scores(y[cal], mean[cal] + d), code[cal], 1 - level)
                            lower, upper = split_interval(mean[out] + d, q)
                        else:
                            q = calibrate(cqr_scores(y[cal], lo[cal] + d, hi[cal] + d), code[cal], 1 - level)
                            lower, upper = cqr_interval(lo[out] + d, hi[out] + d, q)
                        covered = (y[out] >= lower) & (y[out] <= upper)
                        is_high = high[out]
                        per_high = pd.Series(covered[is_high]).groupby(code[out][is_high]).mean()
                        rows.append(
                            {
                                "model": model,
                                "method": method,
                                "shifted": shifted,
                                "subjects": size,
                                "repeat": repeat,
                                "offset": offset,
                                "correction": q[None],
                                "coverage": pd.Series(covered).groupby(code[out]).mean().mean(),
                                "width_median": float(np.median(upper - lower)),
                                "coverage_sbp_ge160": covered[is_high].mean() if is_high.any() else np.nan,
                                "coverage_sbp_lt90": covered[low[out]].mean() if low[out].any() else np.nan,
                                "coverage_sbp_ge160_patient": per_high.mean() if len(per_high) else np.nan,
                                "patients_sbp_ge160": len(per_high),
                            }
                        )
    return pd.DataFrame(rows)


def patient_effective_share(subjects: ArrayLike, weights: ArrayLike) -> float:
    """Kish effective sample size of the calibration patients (weights summed per patient), as a share of them."""
    per_patient = pd.Series(weights).groupby(np.asarray(subjects)).sum().to_numpy()
    per_patient = per_patient / per_patient.sum()
    return float(1.0 / (per_patient**2).sum() / len(per_patient))


def weighted_conformal(
    cal: pd.DataFrame,
    ext: pd.DataFrame,
    ratios: tuple[np.ndarray, np.ndarray],
    target: str,
    model: str,
    level: float,
    quantiles: tuple[float, float],
    bootstrap: tuple[int, int],
) -> list[dict]:
    """Weighted split and CQR intervals for the target segments, summarised per patient.

    ratios is (density ratio of each calibration segment, of each target segment); bootstrap is (resamples, seed).
    """
    ratio_cal, ratio_ext = ratios
    n_boot, seed = bootstrap
    q_lo, q_hi = quantiles
    lo_c, hi_c = cal[quantile_column(target, model, q_lo)], cal[quantile_column(target, model, q_hi)]
    lo_e = ext[quantile_column(target, model, q_lo)].to_numpy()
    hi_e = ext[quantile_column(target, model, q_hi)].to_numpy()
    setups = {
        "weighted_split": (split_scores(cal[target], cal[f"{target}_{model}_mean"]), None),
        "weighted_cqr": (
            cqr_scores(cal[target], np.minimum(lo_c, hi_c), np.maximum(lo_c, hi_c)),
            (np.minimum(lo_e, hi_e), np.maximum(lo_e, hi_e)),
        ),
    }
    base = subject_weights(cal["subjectid"])
    base = base / base.sum() * cal["subjectid"].nunique()
    rows = []
    for method, (score, bounds) in setups.items():
        order = np.argsort(score)
        sorted_scores = score[order]
        cumulative = np.cumsum((base * ratio_cal)[order])
        # The weight of a target point x is its ratio w(x); the quantile is the first score whose cumulative weight
        # reaches level x (total calibration weight + w(x))
        position = np.searchsorted(cumulative, level * (cumulative[-1] + ratio_ext), side="left")
        q = np.where(position < len(sorted_scores), sorted_scores[np.minimum(position, len(sorted_scores) - 1)], np.inf)
        if bounds is None:
            mean = ext[f"{target}_{model}_mean"].to_numpy()
            lower, upper = mean - q, mean + q
        else:
            lower, upper = bounds[0] - q, bounds[1] + q
        result = summarise(
            per_subject(ext[target], lower, upper, ext["subjectid"]), n_boot=n_boot, seed=seed, nominal=level
        )
        result["infinite_share"] = float(np.mean(~np.isfinite(q)))
        weights = (base * ratio_cal) / (base * ratio_cal).sum()
        result["effective_sample_share"] = float(1.0 / (weights**2).sum() / len(weights))
        result["effective_patient_share"] = patient_effective_share(cal["subjectid"], base * ratio_cal)
        rows.append({"model": model, "level": level, "method": method, **result})
    return rows
