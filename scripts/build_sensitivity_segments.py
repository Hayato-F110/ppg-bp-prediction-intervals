"""Data for the sensitivity analysis (analysis plan, Section 2.1): re-cut the segments under other exclusion rules.

Only cases of calibration and test patients are downloaded (the models are unchanged; only the evaluated segments
change).
Only kept segments were saved, so other rules need a new download. In one pass, the quantities the rules use
(pulse rate, SD of beat-to-beat SBP, pulse pressure, reference values) are kept for every segment, and
the PPG is saved only for segments that the main rules exclude but wider rules admit.

rules compared (scripts/evaluate_sensitivity.py):
    base        pulse rate 40-180, SD of SBP at most 15 (main analysis)
    rate_wide   pulse rate 20-200
    sd10 / sd20 SD of SBP at most 10 / 20

usage: uv run python -X utf8 scripts/build_sensitivity_segments.py [--workers 4] [--limit N]
output: data/processed/vitaldb_sensitivity/<caseid>.npz    log: results/logs/sensitivity_build.log
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.data.segmentation import (
    ABP_MAX,
    ABP_MIN,
    MAX_BEAT_SBP_SD,
    MIN_BEATS,
    MIN_PULSE_PRESSURE,
    PULSE_RATE_RANGE,
    beat_pressures,
    has_flatline,
)
from ppg_bp.data.vitaldb_source import NATIVE_FS, load_case_signals
from ppg_bp.runlog import RunLog

CONFIG = load_config()
OUT_DIR = PROJECT_ROOT / CONFIG["paths"]["processed"] / "vitaldb_sensitivity"
LOG = PROJECT_ROOT / CONFIG["paths"]["results"] / "logs" / "sensitivity_build.log"
WIDE_RATE = (20.0, 200.0)
WIDE_SD = 20.0


log = RunLog(LOG)


def process_case(case_id: int) -> tuple:
    out_path = OUT_DIR / f"{case_id}.npz"
    if out_path.exists():
        return case_id, "skip"
    try:
        ppg, abp = load_case_signals(case_id)
    except Exception as error:
        return case_id, f"error:{type(error).__name__}"
    seg_sec, out_fs = CONFIG["segment"]["length_sec"], CONFIG["segment"]["fs_hz"]
    seg = seg_sec * NATIVE_FS
    n_seg = len(ppg) // seg
    rows = []
    for i in range(n_seg):
        p, a = ppg[i * seg : (i + 1) * seg], abp[i * seg : (i + 1) * seg]
        # the first four checks of the main analysis (missing, pressure range, flat, few beats) are shared by all rules
        if np.isnan(p).any() or np.isnan(a).any() or a.min() < ABP_MIN or a.max() > ABP_MAX:
            continue
        if has_flatline(p, NATIVE_FS) or has_flatline(a, NATIVE_FS):
            continue
        peaks, sbp_beats, dbp_beats = beat_pressures(a, NATIVE_FS)
        if len(peaks) < MIN_BEATS:
            continue
        rate = 60.0 / (np.median(np.diff(peaks)) / NATIVE_FS)
        rows.append((i, rate, sbp_beats.std(), sbp_beats.mean(), dbp_beats.mean(), a.mean()))
    if not rows:
        np.savez(out_path, index=np.empty(0, np.int32))
        return case_id, "ok"
    index, rate, sd, sbp, dbp, mbp = (np.array(v) for v in zip(*rows, strict=False))
    pressure_ok = sbp - dbp >= MIN_PULSE_PRESSURE
    base = (rate >= PULSE_RATE_RANGE[0]) & (rate <= PULSE_RATE_RANGE[1]) & (sd <= MAX_BEAT_SBP_SD) & pressure_ok
    wide = (rate >= WIDE_RATE[0]) & (rate <= WIDE_RATE[1]) & (sd <= WIDE_SD) & pressure_ok
    new = wide & ~base  # segments admitted only by wider rules; their PPG is saved
    ppg_new = np.empty((0, seg_sec * out_fs), dtype=np.float32)
    if new.any():
        filled = ppg[: n_seg * seg].astype(np.float64)
        missing = np.isnan(filled)
        if missing.any():
            where = np.arange(len(filled))
            filled[missing] = np.interp(where[missing], where[~missing], filled[~missing])
        down = resample_poly(filled, out_fs, NATIVE_FS).astype(np.float32)
        ppg_new = down[: n_seg * seg_sec * out_fs].reshape(n_seg, -1)[index[new].astype(int)]
    np.savez(
        out_path,
        index=index.astype(np.int32),
        pulse_rate=rate.astype(np.float32),
        sbp_sd=sd.astype(np.float32),
        sbp=sbp.astype(np.float32),
        dbp=dbp.astype(np.float32),
        mbp=mbp.astype(np.float32),
        is_base=base,
        is_new=new,
        ppg_new=ppg_new,
    )
    return case_id, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "vitaldb" / "split.csv")
    case_ids = split.loc[split["split"].isin(["calibration", "test"]) & (split["ok"] > 0), "caseid"].tolist()[
        : args.limit
    ]
    log(f"=== start: {len(case_ids)} cases of calibration and test patients")
    start, errors = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (_case_id, status) in enumerate(pool.map(process_case, case_ids), 1):
            errors += status.startswith("error")
            if k % 50 == 0 or k == len(case_ids):
                log(f"{k}/{len(case_ids)} cases, errors {errors}, {(time.time() - start) / 60:.1f} min")
    log("=== done")


if __name__ == "__main__":
    main()
