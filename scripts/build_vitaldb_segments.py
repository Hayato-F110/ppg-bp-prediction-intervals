"""Download all VitalDB cases and save 10-s segments, reference pressures and exclusion reasons.

If stopped, a rerun skips the cases already done.

usage: uv run python scripts/build_vitaldb_segments.py [--limit N]
output (not in the repository):
    data/processed/vitaldb/segments/<caseid>.npz   PPG and reference values of the kept segments
    data/processed/vitaldb/case_summary.csv        per case: number of segments and exclusion reasons
    data/processed/vitaldb/cases.csv               subject ID and attributes
"""

import argparse
import csv
import time

import numpy as np

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.data.segmentation import REASONS, segment_case
from ppg_bp.data.vitaldb_source import NATIVE_FS, list_cases, load_case_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only the first N cases (for testing)")
    args = parser.parse_args()

    config = load_config()
    seg_sec = config["segment"]["length_sec"]
    out_fs = config["segment"]["fs_hz"]
    out_dir = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    cases = list_cases()
    cases.to_csv(out_dir / "cases.csv", index=False)
    case_ids = cases["caseid"].tolist()[: args.limit]

    summary_path = out_dir / "case_summary.csv"
    done = set()
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            done = {int(row["caseid"]) for row in csv.DictReader(f)}
    fields = ["caseid", "n_segments", *REASONS, "error"]
    write_header = not summary_path.exists()

    start = time.perf_counter()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        todo = [c for c in case_ids if c not in done]
        print(f"{len(case_ids)} cases, {len(case_ids) - len(todo)} done, {len(todo)} to go", flush=True)
        for n, case_id in enumerate(todo, 1):
            row = {"caseid": case_id, "n_segments": 0, "error": "", **dict.fromkeys(REASONS, 0)}
            try:
                ppg, abp = load_case_signals(case_id)
                ppg_out, rows, counts = segment_case(ppg, abp, NATIVE_FS, seg_sec, out_fs)
                row.update(counts)
                row["n_segments"] = sum(counts.values())
                if rows:
                    table = np.array(rows, dtype=np.float64)
                    np.savez(
                        seg_dir / f"{case_id}.npz",
                        ppg=ppg_out,
                        index=table[:, 0].astype(np.int32),
                        sbp=table[:, 1].astype(np.float32),
                        dbp=table[:, 2].astype(np.float32),
                        mbp=table[:, 3].astype(np.float32),
                        pulse_rate=table[:, 4].astype(np.float32),
                    )
            except Exception as exc:  # one failing case does not stop the run; the reason is recorded
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            writer.writerow(row)
            f.flush()
            if n % 50 == 0 or n == len(todo):
                elapsed = time.perf_counter() - start
                remaining = elapsed / n * (len(todo) - n)
                print(f"{n}/{len(todo)} cases, {elapsed / 60:.0f} min, about {remaining / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
