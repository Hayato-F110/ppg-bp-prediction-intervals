"""Weighted conformal prediction under covariate shift (Tibshirani et al. 2019).

A classifier that tells the source (VitalDB) from the target dataset estimates the density ratio of each segment.
Calibration scores are weighted by (equal weight per patient) x (density ratio), and each target segment gets its
own weighted quantile.
"""

from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score

from ppg_bp.conformal.core import cqr_scores, split_scores, subject_weights
from ppg_bp.conformal.pipeline import quantile_column
from ppg_bp.evaluation.coverage import per_subject, summarise


def domain_weights(
    train: pd.DataFrame,
    cal_features: pd.DataFrame,
    ext_features: pd.DataFrame,
    columns: list[str],
    seed: int,
    clip: tuple[float, float] = (1e-3, 1e3),
) -> tuple[np.ndarray, np.ndarray, float, pd.Series]:
    """Density ratios from a classifier of source against target.

    The target patients are split in two; a classifier fitted on one half is applied to the other. On the source
    side the classifier is fitted on training patients and applied to the calibration patients, so no weighted
    point is used for fitting. Returns (ratio of each calibration segment, ratio of each target segment,
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
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=4,
            verbose=-1,
            random_state=seed,
        )
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
