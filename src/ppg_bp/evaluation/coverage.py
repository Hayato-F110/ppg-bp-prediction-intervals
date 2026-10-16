"""Evaluation of prediction intervals. The statistical unit is the subject (analysis plan, Sections 4.2 and 8).

The main measure is subject-level coverage: coverage is computed for each subject and averaged over subjects.
Confidence intervals come from a bootstrap over subjects. Coverage is always reported together with width.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def per_subject(y: ArrayLike, lower: ArrayLike, upper: ArrayLike, subjects: ArrayLike) -> pd.DataFrame:
    """Per subject: coverage, rate of misses above and below, median width and number of segments."""
    y, lower, upper = (np.asarray(v, dtype=np.float64) for v in (y, lower, upper))
    table = pd.DataFrame(
        {
            "subject": np.asarray(subjects),
            "covered": (y >= lower) & (y <= upper),
            "above": y > upper,  # true value above the interval
            "below": y < lower,  # true value below the interval
            "width": upper - lower,
        }
    )
    grouped = table.groupby("subject")
    out = pd.DataFrame(grouped[["covered", "above", "below"]].mean())
    out["width"] = grouped["width"].median()
    out["n"] = grouped.size()
    return out


def summarise(subject_table: pd.DataFrame, n_boot: int = 2000, seed: int = 0, nominal: float | None = None) -> dict:
    """Subject-level measures with bootstrap 95% confidence intervals."""
    values = subject_table[["covered", "above", "below"]].to_numpy()
    n = len(values)
    out: dict[str, float] = {"subjects": n, "segments": int(subject_table["n"].sum())}
    means = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) if n > 1 else np.tile(means, (n_boot, 1))
    for k, name in enumerate(("coverage", "miss_above", "miss_below")):
        out[name] = float(means[k])
        out[f"{name}_lo"], out[f"{name}_hi"] = (float(v) for v in np.percentile(boot[:, k], [2.5, 97.5]))
    weights = subject_table["n"].to_numpy()
    out["coverage_segment_level"] = float(np.average(subject_table["covered"], weights=weights))
    width = subject_table["width"]
    out["width_median"], out["width_q25"], out["width_q75"] = (float(width.quantile(q)) for q in (0.5, 0.25, 0.75))
    if nominal is not None:
        covered = subject_table["covered"]
        out["subjects_below_minus10"] = float((covered < nominal - 0.10).mean())
        out["subjects_below_minus20"] = float((covered < nominal - 0.20).mean())
    return out


def evaluate(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    subjects: ArrayLike,
    strata: Mapping[str, ArrayLike] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Measures overall and by stratum. strata is {variable name: array of each segment's stratum label}."""
    y, lower, upper, subjects = (np.asarray(v) for v in (y, lower, upper, subjects))
    rows = [{"variable": "all", "stratum": "all", **summarise(per_subject(y, lower, upper, subjects), **kwargs)}]
    for variable, values in (strata or {}).items():
        labels = np.asarray(values)
        for stratum in pd.unique(labels[pd.notna(labels)]):
            mask = labels == stratum
            rows.append(
                {
                    "variable": variable,
                    "stratum": stratum,
                    **summarise(per_subject(y[mask], lower[mask], upper[mask], subjects[mask]), **kwargs),
                }
            )
    return pd.DataFrame(rows)


def paired_difference(a: ArrayLike, b: ArrayLike, subjects: ArrayLike, n_boot: int, seed: int) -> dict:
    """Mean of the per-subject difference in coverage (a − b), with a 95% CI from a bootstrap that keeps each
    subject's pair of values together."""
    table = pd.DataFrame({"a": a, "b": b, "s": subjects}).groupby("s")[["a", "b"]].mean()
    diff = np.asarray(table["a"] - table["b"])
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {
        "patients": len(diff),
        "difference": diff.mean(),
        "ci_low": np.percentile(boots, 2.5),
        "ci_high": np.percentile(boots, 97.5),
    }
