"""Detection rate, amplitude ratios and speed of the fiducial points in some VitalDB training patients.

A check before computing the features of all data. Uses training cases only and prints summaries only.

usage: uv run python -X utf8 scripts/check_fiducials_vitaldb.py [--cases 60] [--per-case 200]
output: results/fiducial_validation/vitaldb_train_sample.csv (summary per case)
"""

import argparse
import time

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.features.fiducials import POINTS, analyse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--per-case", type=int, default=200)
    args = parser.parse_args()

    config = load_config()
    fs = config["segment"]["fs_hz"]
    root = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    split = pd.read_csv(root / "split.csv")
    train = split[(split["split"] == "train") & (split["ok"] > 0)]
    rng = np.random.default_rng(config["seed"])
    case_ids = rng.choice(train["caseid"].to_numpy(), size=min(args.cases, len(train)), replace=False)

    rows, elapsed, n_segments = [], 0.0, 0
    for case_id in sorted(case_ids.tolist()):
        with np.load(root / "segments" / f"{case_id}.npz") as data:
            ppg = data["ppg"]
        chosen = rng.choice(len(ppg), size=min(args.per_case, len(ppg)), replace=False)
        beats = {key: [] for key in ("cd_distinct", "duration", *POINTS, *(f"{p}_over_a" for p in "bcde"))}
        no_beat = 0
        for i in chosen:
            start = time.perf_counter()
            out = analyse(ppg[i], fs)
            elapsed += time.perf_counter() - start
            n_segments += 1
            if len(out["t_onset"]) == 0:
                no_beat += 1
                continue
            beats["cd_distinct"].append(out["cd_distinct"])
            beats["duration"].append(out["t_end"] - out["t_onset"])
            for p in POINTS:
                beats[p].append(~np.isnan(out[f"t_{p}"]))
            for p in "bcde":
                beats[f"{p}_over_a"].append(out[f"v_{p}"] / out["v_a"])
        if not beats["duration"]:
            rows.append({"caseid": case_id, "segments": len(chosen), "no_beat_segments": no_beat, "beats": 0})
            continue
        beats = {key: np.concatenate(value) for key, value in beats.items()}
        row = {
            "caseid": case_id,
            "segments": len(chosen),
            "no_beat_segments": no_beat,
            "beats": len(beats["duration"]),
            "beats_per_segment": len(beats["duration"]) / len(chosen),
            "cd_distinct": beats["cd_distinct"].mean(),
        }
        row |= {f"found_{p}": beats[p].mean() for p in POINTS}
        row |= {f"median_{p}_over_a": np.nanmedian(beats[f"{p}_over_a"]) for p in "bcde"}
        rows.append(row)

    table = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / config["paths"]["results"] / "fiducial_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "vitaldb_train_sample.csv", index=False, float_format="%.4f")

    total_beats = table["beats"].sum()
    print(f"cases {len(table)}, segments {n_segments}, beats {int(total_beats)}")
    print(
        f"time: {1000 * elapsed / n_segments:.2f} ms per segment (about {3.654e6 * elapsed / n_segments / 3600:.1f} h "
        "for 3.65 million segments, one process)"
    )
    print(f"segments without any beat: {table['no_beat_segments'].sum() / n_segments:.1%}")
    print(f"beats per segment (median over cases): {table['beats_per_segment'].median():.1f}")
    cols = ["found_a", "found_b", "found_c", "found_e", "found_f", "cd_distinct"]
    cols += [f"median_{p}_over_a" for p in "bcde"]
    print("\ndistribution over cases")
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(
        table[cols]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T[["5%", "25%", "50%", "75%", "95%", "min", "max"]]
    )


if __name__ == "__main__":
    main()
