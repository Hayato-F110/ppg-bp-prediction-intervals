"""Check, in some training patients, whether the position of the e wave depends on the pulse rate.

Segments with a time from a to e (t_ae) below 0.28 s count as an early e; their share is given by pulse interval.
Used to check a change of the detector before recomputing all data.

usage: uv run python -X utf8 scripts/check_e_wave_timing.py [--cases 80] [--per-case 150]
"""

import argparse

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.features.extract import segment_features

EARLY_SEC = 0.28
BINS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 2.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--per-case", type=int, default=150)
    args = parser.parse_args()

    config = load_config()
    fs = config["segment"]["fs_hz"]
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    split = pd.read_csv(root / "split.csv")
    train = split[(split["split"] == "train") & (split["ok"] > 0)]
    rng = np.random.default_rng(config["seed"])
    rows = []
    for case_id in rng.choice(train["caseid"].to_numpy(), size=min(args.cases, len(train)), replace=False):
        with np.load(root / "segments" / f"{case_id}.npz") as data:
            ppg = data["ppg"]
        for i in rng.choice(len(ppg), size=min(args.per_case, len(ppg)), replace=False):
            out = segment_features(ppg[i], fs)
            rows.append((out["t_ae"], out["ibi_median"], out["frac_cd_found"], out["frac_e_found"]))
    table = pd.DataFrame(rows, columns=["t_ae", "ibi", "cd_found", "e_found"])
    print(
        f"segments {len(table)}, share of beats with e (mean over segments) {table['e_found'].mean():.3f}, c and d "
        f"{table['cd_found'].mean():.3f}"
    )
    valid = table.dropna(subset=["t_ae", "ibi"])
    valid = valid.assign(early=valid["t_ae"] < EARLY_SEC)
    print(f"segments with an early e: {100 * valid['early'].mean():.1f}%")
    grouped = valid.groupby(pd.cut(valid["ibi"], BINS), observed=True)
    print(
        grouped.agg(
            segments=("early", "size"),
            early_e_share=("early", "mean"),
            t_ae_median=("t_ae", "median"),
            cd_found=("cd_found", "mean"),
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
