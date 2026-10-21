"""Compare the detected fiducial points with public manual annotations.

Annotations: Goda, Charlton, Behar, "pyPPG benchmarking", Zenodo, DOI 10.5281/zenodo.10523285 (CC BY 4.0).
Two annotators marked the points on the first complete beat of each of the 219 records of the PPG-BP dataset
(1 kHz, about 2 s). As in the paper, the reference is the mean of the two.

Errors are computed at the original 1 kHz and after downsampling to the 125 Hz of this study.

usage: uv run python -I -X utf8 scripts/validate_fiducials.py
output: results/fiducial_validation/ (error table, errors per record)
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from ppg_bp.config import PROJECT_ROOT
from ppg_bp.features.fiducials import POINTS, analyse

TIGHT_MS, LOOSE_MS = 10, 20  # error limits reported (ms)

ANNOT_DIR = PROJECT_ROOT / "data" / "raw" / "pyppg_annot" / "extracted"
OUT_DIR = PROJECT_ROOT / "results" / "fiducial_validation"
ANNOTATORS = ("MG", "PC")
NATIVE_FS = 1000
RATES = (1000, 125)
MIN_OVERLAP = 0.5  # overlap needed to match an annotated beat with a detected beat

# Table 9 (errors of pyPPG) and Table A1 (between annotators) of the paper; mean absolute error, ms.
PAPER_MAE = {"onset": 7, "peak": 5, "a": 1, "b": 2, "c": 4, "d": 4, "e": 2, "f": 3}
PAPER_INTER = {"onset": 3, "peak": 3, "a": 2, "b": 3, "c": 3, "d": 2, "e": 2, "f": 2}
ANNOT_KEYS = {"onset": "os", "peak": "pk"} | {p: p for p in POINTS}


def load_signals():
    if not (ANNOT_DIR / "PPG-BP_ref1.mat").exists():
        raise SystemExit(f"annotations not found in {ANNOT_DIR}; download them as described in README.md (Data)")
    data = loadmat(ANNOT_DIR / "PPG-BP_ref1.mat")["ppg_data"][0]
    return {str(r["name"][0]): r["sig"].ravel().astype(np.float64) for r in data}


def load_annotations(annotator):
    """Record name -> {point name: array of times (s)}."""
    path = ANNOT_DIR / f"{annotator}final_PPG-BP_annot" / f"{annotator}final_PPG-BP_annot.mat"
    out = {}
    for record in loadmat(path)["fiducial_annot"][0]:
        annot = record["annot"][0, 0]
        out[str(record["name"][0])] = {
            name: annot[key][0, 0]["t"].ravel().astype(np.float64) for name, key in ANNOT_KEYS.items()
        }
    return out


def first_time(times):
    times = times[~np.isnan(times)]
    return float(times[0]) if len(times) else np.nan


def reference(annots, name):
    """Mean and difference of the two annotators. Beat onset and end are the first and second annotations."""
    per_annotator = [a[name] for a in annots]
    ref = {p: np.mean([first_time(a[p]) for a in per_annotator]) for p in ANNOT_KEYS}
    diff = {p: first_time(per_annotator[0][p]) - first_time(per_annotator[1][p]) for p in ANNOT_KEYS}
    ends = [a["onset"][1] if len(a["onset"]) > 1 else np.nan for a in per_annotator]
    ref["end"] = float(np.mean(ends))
    return ref, diff


def match_beat(result, ref):
    """Index of the detected beat that best overlaps the annotated beat, or None."""
    if len(result["t_onset"]) == 0 or np.isnan(ref["onset"]) or np.isnan(ref["end"]):
        return None
    overlap = np.minimum(result["t_end"], ref["end"]) - np.maximum(result["t_onset"], ref["onset"])
    k = int(np.argmax(overlap))
    return k if overlap[k] >= MIN_OVERLAP * (ref["end"] - ref["onset"]) else None


def summarise(errors: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fs in RATES:
        sub = errors[errors["fs"] == fs]
        for point in ANNOT_KEYS:
            has_ref = sub[f"ref_{point}"].notna()
            err = sub.loc[has_ref, f"err_{point}"] * 1000
            found = err.notna()
            e = err[found]
            rows.append(
                {
                    "fs": fs,
                    "point": point,
                    "annotated": int(has_ref.sum()),
                    "found_pct": 100 * found.mean(),
                    "MAE_ms": e.abs().mean(),
                    "median_abs_error_ms": e.abs().median(),
                    "bias_ms": e.mean(),
                    "SD_ms": e.std(),
                    "within_10ms_pct": 100 * (e.abs() <= TIGHT_MS).mean(),
                    "within_20ms_pct": 100 * (e.abs() <= LOOSE_MS).mean(),
                    "inter_annotator_mae_ms": (inter[point] * 1000).abs().mean(),
                    "paper_pyppg_mae_ms": PAPER_MAE[point],
                    "paper_inter_annotator_mae_ms": PAPER_INTER[point],
                }
            )
    return pd.DataFrame(rows)


def main():
    signals = load_signals()
    annots = [load_annotations(a) for a in ANNOTATORS]
    names = sorted(set(signals) & set(annots[0]) & set(annots[1]))
    print(f"records {len(signals)}, with both annotators {len(names)}")

    rows, inter_rows = [], []
    for name in names:
        ref, diff = reference(annots, name)
        inter_rows.append({"name": name, **diff})
        for fs in RATES:
            sig = signals[name] if fs == NATIVE_FS else resample_poly(signals[name], fs, NATIVE_FS)
            result = analyse(sig, fs)
            k = match_beat(result, ref)
            row = {"name": name, "fs": fs, "matched": k is not None}
            for point in ANNOT_KEYS:
                row[f"ref_{point}"] = ref[point]
                row[f"err_{point}"] = np.nan if k is None else result[f"t_{point}"][k] - ref[point]
            if k is not None:
                row["cd_distinct"] = bool(result["cd_distinct"][k])
                for p in "bcde":
                    row[f"{p}_over_a"] = result[f"v_{p}"][k] / result["v_a"][k]
            rows.append(row)

    errors, inter = pd.DataFrame(rows), pd.DataFrame(inter_rows)
    summary = summarise(errors, inter)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUT_DIR / "errors_per_record.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.2f")

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    for fs in RATES:
        sub = errors[errors["fs"] == fs]
        print(f"\n=== {fs} Hz: records with a matched beat {sub['matched'].mean():.1%},")
        print(f"    c and d found as separate extrema {sub['cd_distinct'].mean():.1%}")
        print(summary[summary["fs"] == fs].drop(columns="fs").to_string(index=False))

    # how much downsampling changes the amplitude ratios (same record, same beat)
    wide = errors.pivot(index="name", columns="fs")  # noqa: PD010 (one value per cell)
    print("\n=== amplitude ratios: agreement between 1000 Hz and 125 Hz")
    for p in "bcde":
        hi, lo = wide[(f"{p}_over_a", 1000)], wide[(f"{p}_over_a", 125)]
        ok = hi.notna() & lo.notna()
        print(
            f"  {p}/a: n={ok.sum()}, median {hi[ok].median():.3f} / {lo[ok].median():.3f}, "
            f"correlation {np.corrcoef(hi[ok], lo[ok])[0, 1]:.3f}, median absolute difference "
            f"{(hi[ok] - lo[ok]).abs().median():.3f}"
        )


if __name__ == "__main__":
    main()
