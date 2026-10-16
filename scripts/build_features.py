"""Features of all segments.

One file per case (data/processed/vitaldb/features/<caseid>.parquet), merged at the end.
If stopped, a rerun skips the cases already done.

usage: uv run python -X utf8 scripts/build_features.py [--workers 5] [--limit N] [--overwrite]
output: data/processed/vitaldb/features.parquet (caseid, index, features; rows match segment_labels.parquet)
"""

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.features.extract import FEATURE_NAMES, segment_features

CONFIG = load_config()
FS = CONFIG["segment"]["fs_hz"]
ROOT = PROJECT_ROOT / CONFIG["paths"]["processed"] / "vitaldb"
OUT_DIR = ROOT / "features"
OVERWRITE = "--overwrite" in sys.argv  # recompute cases already done (after a change of definitions)


def process_case(case_id: int) -> int:
    out_path = OUT_DIR / f"{case_id}.parquet"
    if out_path.exists() and not OVERWRITE:
        return 0
    with np.load(ROOT / "segments" / f"{case_id}.npz") as data:
        ppg, index = data["ppg"], data["index"]
    values = np.array([list(segment_features(row, FS).values()) for row in ppg], dtype=np.float32)
    table = pd.DataFrame(values, columns=FEATURE_NAMES)
    table.insert(0, "index", index)
    table.insert(0, "caseid", np.int64(case_id))
    tmp = out_path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return len(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = sorted(int(p.stem) for p in (ROOT / "segments").glob("*.npz"))[: args.limit]
    start, done = time.time(), 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(process_case, case_ids), 1):
            done += n
            if i % 100 == 0 or i == len(case_ids):
                print(f"{i}/{len(case_ids)} cases, {done} segments, {(time.time() - start) / 60:.1f} min", flush=True)

    if args.limit is None:
        tables = [pd.read_parquet(OUT_DIR / f"{case_id}.parquet") for case_id in case_ids]
        merged = pd.concat(tables, ignore_index=True)
        merged.to_parquet(ROOT / "features.parquet", index=False)
        print(f"features.parquet: {len(merged)} rows, {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
