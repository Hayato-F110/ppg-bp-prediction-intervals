"""LightGBM models of the mean and quantiles: training, saving and loading.

Training segments and the number of trees are chosen as in model development (80% of training subjects for fitting, 20%
for early stopping).
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import lightgbm as lgb
import pandas as pd

from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import inner_validation_subjects, load_split, sample_per_subject

COLUMNS: list[str] = list(FEATURE_NAMES)


def quantile_name(q: float) -> str:
    return f"q{round(q * 1000):03d}"


def output_labels(quantiles: Sequence[float]) -> list[str]:
    return ["mean"] + [quantile_name(q) for q in quantiles]


def fit_lightgbm(  # noqa: PLR0913 (the data, settings and seed of one fit)
    params: dict,
    objective: str,
    alpha: float | None,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    *,
    eval_metric: str | None = None,
) -> lgb.LGBMRegressor:
    """Fit with early stopping on (x_val, y_val). eval_metric adds a metric that early stopping also watches."""
    params = dict(params)
    rounds = params.pop("early_stopping_rounds")
    params["objective"] = objective
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params, random_state=seed)
    model.fit(
        x_fit,
        y_fit,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(rounds, verbose=False)],
    )
    return model


def train_models(
    config: dict, feature_settings: dict, quantiles: Sequence[float], log: Callable[[str], None] = print
) -> dict:
    """{(target, output name): Booster}."""
    seed = config["seed"]
    train = load_split(config, "train")
    is_val = train["subjectid"].isin(
        list(inner_validation_subjects(train["subjectid"], feature_settings["validation_fraction"], seed))
    )
    fit = sample_per_subject(train.loc[~is_val], feature_settings["max_segments_per_subject"], seed)
    val = train.loc[is_val]
    models = {}
    for target in feature_settings["targets"]:
        jobs = [("mean", "regression", None)] + [(quantile_name(q), "quantile", q) for q in quantiles]
        for label, objective, alpha in jobs:
            model = fit_lightgbm(
                feature_settings["lightgbm"],
                objective,
                alpha,
                # cast: pandas does not tell the type checker that a list of columns gives a DataFrame
                cast("pd.DataFrame", fit[COLUMNS]),
                cast("pd.Series", fit[target]),
                cast("pd.DataFrame", val[COLUMNS]),
                cast("pd.Series", val[target]),
                seed,
            )
            models[(target, label)] = model.booster_
            log(f"LightGBM {target} {label}: {model.best_iteration_} trees")
    return models


def save_models(models: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for (target, label), booster in models.items():
        booster.save_model(str(folder / f"{target}_{label}.txt"), num_iteration=booster.best_iteration)


def load_models(folder: Path, targets: Sequence[str], quantiles: Sequence[float]) -> dict | None:
    """Load the saved models if all exist; otherwise None."""
    paths = {(t, label): folder / f"{t}_{label}.txt" for t in targets for label in output_labels(quantiles)}
    if not all(p.exists() for p in paths.values()):
        return None
    return {key: lgb.Booster(model_file=str(path)) for key, path in paths.items()}


def predict_all(models: dict, features: pd.DataFrame, prefix: str = "lightgbm") -> pd.DataFrame:
    """Columns are named <target>_<prefix>_<output name>."""
    x = features[COLUMNS]
    return pd.DataFrame(
        {f"{t}_{prefix}_{label}": booster.predict(x).astype("float32") for (t, label), booster in models.items()},
        index=features.index,
    )
