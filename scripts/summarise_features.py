"""Missing rates and distributions of the features in the training patients only, and the signal-quality tertiles.

Features of test and calibration patients are used only to check the number of rows.

usage: uv run python -X utf8 scripts/summarise_features.py
output: results/features/summary_train.csv, configs/sqi_tertiles.yaml
"""

import numpy as np
import pandas as pd
import yaml

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.features.extract import FEATURE_NAMES
from ppg_bp.models.dataset import KEYS, load_split

ORIGINAL_THRESHOLD = 0.86  # usable-segment threshold of the template-matching index (Orphanidou et al. 2015)


def main():
    config = load_config()
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    labels = pd.read_parquet(root / "segment_labels.parquet", columns=KEYS)
    features = pd.read_parquet(root / "features.parquet", columns=KEYS)
    matched = len(labels.merge(features, on=KEYS, how="inner", validate="one_to_one"))
    print(f"label rows {len(labels)}, feature rows {len(features)}, matched {matched}")
    if not (matched == len(labels) == len(features)):
        raise AssertionError("features and reference values do not match row for row")

    train = load_split(config, "train")
    values = train[list(FEATURE_NAMES)]
    summary = values.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T[["1%", "25%", "50%", "75%", "99%"]]
    summary.insert(0, "missing_pct", 100 * values.isna().mean())
    # spread of the missing rate between subjects (finds features that depend on the subject)
    per_subject = values.isna().groupby(train["subjectid"]).mean() * 100
    summary.insert(1, "missing_pct_subject_median", per_subject.median())
    summary.insert(2, "missing_pct_subject_p95", per_subject.quantile(0.95))

    sqi = train["sqi_template"]
    cuts = [float(x) for x in sqi.quantile([1 / 3, 2 / 3])]
    out_dir = PROJECT_ROOT / config["paths"]["results"] / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary_train.csv", float_format="%.4f")
    with (PROJECT_ROOT / "configs" / "sqi_tertiles.yaml").open("w", encoding="utf-8") as f:
        f.write(
            "# Tertile cut points of signal quality (sqi_template), from the training patients only (analysis plan, "
            "Section 4.4).\n"
        )
        f.write("# Written by scripts/summarise_features.py; do not edit by hand.\n")
        yaml.safe_dump(
            {
                "sqi_template": {
                    "lower": cuts[0],
                    "upper": cuts[1],
                    "n_segments": len(sqi),
                    "n_subjects": int(train["subjectid"].nunique()),
                }
            },
            f,
        )

    pd.set_option("display.width", 220, "display.max_rows", 60, "display.float_format", "{:.3f}".format)
    print(f"\ntraining: {train['subjectid'].nunique()} patients, {len(train)} segments")
    print(summary.to_string())
    print(f"\nsignal-quality tertile cut points: {cuts[0]:.4f}, {cuts[1]:.4f}")
    print(
        f"sqi_template 0 (fewer than 3 beats): {100 * np.mean(sqi == 0):.2f}%, below {ORIGINAL_THRESHOLD} (original "
        f"threshold): {100 * np.mean(sqi < ORIGINAL_THRESHOLD):.1f}%"
    )


if __name__ == "__main__":
    main()
