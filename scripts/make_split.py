"""Make the patient-level split and summarise the exclusions and the pressures of each part.

usage: uv run python scripts/make_split.py (the split is fixed; do not rerun)
output (not in the repository):
    data/processed/vitaldb/split.csv        per case: subject ID, part and number of kept segments
    data/processed/vitaldb/segment_labels.parquet  per kept segment: case ID and reference values (no waveforms)
Only summary statistics are printed.
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.data.segmentation import REASONS
from ppg_bp.data.split import SPLIT_NAMES, assert_no_subject_overlap, split_subjects

PRINT_LOW, PRINT_HIGH = 100, 140  # SBP limits (mmHg) of the summary printed for each part
# The split of the paper (case ID, subject ID, part; scripts/export_id_lists.py). A download made later may differ
# slightly (failed or updated cases), so the published split is used and the differences are reported.
PUBLISHED_SPLIT = PROJECT_ROOT / "data_lists" / "vitaldb_split.csv"
PUBLISHED_CASES = PROJECT_ROOT / "data_lists" / "vitaldb_cases.csv"


def compare_with_published_cases(summary: pd.DataFrame) -> None:
    """Report cases whose numbers of segments and of kept segments differ from those of the paper."""
    published = pd.read_csv(PUBLISHED_CASES)
    here = summary[["caseid", "n_segments", "ok"]]
    merged = published.merge(here, on="caseid", how="left", suffixes=("", "_here"))
    missing = merged["ok_here"].isna()
    differ = ~missing & ((merged["n_segments"] != merged["n_segments_here"]) | (merged["ok"] != merged["ok_here"]))
    print(
        f"published cases: {len(published)}; {int(missing.sum())} not downloaded here, "
        f"{int(differ.sum())} with different numbers of segments, {int((~missing & ~differ).sum())} identical"
    )


def use_published_split(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the cases of the published split, with its parts; report how the seeded split compares."""
    published = pd.read_csv(PUBLISHED_SPLIT)
    keys = ["caseid", "subjectid"]
    merged = table.merge(published, on=keys, how="outer", suffixes=("", "_published"), indicator=True)
    only_here = int((merged["_merge"] == "left_only").sum())
    only_published = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    differ = int((both["split"] != both["split_published"]).sum())
    print(
        f"published split: {len(published)} cases; {only_here} cases here not in it (dropped), "
        f"{only_published} of its cases missing here, {differ} cases where the seeded split differs (published used)"
    )
    kept = table.merge(published, on=["caseid", "subjectid"], suffixes=("_seeded", ""))
    return kept.drop(columns="split_seeded")


def main():
    config = load_config()
    out_dir = PROJECT_ROOT / config["paths"]["processed"] / "vitaldb"
    cases = pd.read_csv(out_dir / "cases.csv")
    summary = pd.read_csv(out_dir / "case_summary.csv")

    errors = summary[summary["error"].fillna("") != ""]
    print(f"cases processed {len(summary)}, errors {len(errors)}")
    for _, row in errors.iterrows():
        print(f"  case {row['caseid']}: {row['error']}")

    if PUBLISHED_CASES.exists():
        compare_with_published_cases(summary)

    totals = summary[list(REASONS)].sum()
    print(f"\n10-s segments in total {int(totals.sum()):,}")
    for reason in REASONS:
        print(f"  {reason:15s} {int(totals[reason]):>10,}  {totals[reason] / totals.sum():6.1%}")

    table = cases.merge(summary[["caseid", "ok"]], on="caseid", how="left").fillna({"ok": 0})
    table = table[table["ok"] > 0].copy()
    print(
        f"\ncases with at least one kept segment {len(table)} / {len(cases)}, subjects {table['subjectid'].nunique()}"
    )
    per_subject = table.groupby("subjectid")["ok"].sum()
    print(
        f"kept segments per subject: median {per_subject.median():.0f}, "
        f"5th-95th percentile {per_subject.quantile(0.05):.0f}-{per_subject.quantile(0.95):.0f}, max "
        f"{per_subject.max():.0f}"
    )

    split = split_subjects(table["subjectid"], config["split"], config["seed"])
    table["split"] = table["subjectid"].map(split)
    if PUBLISHED_SPLIT.exists():
        table = use_published_split(table)
    assert_no_subject_overlap(table)
    table.to_csv(out_dir / "split.csv", index=False)

    rows = []
    for case_id in table["caseid"]:
        data = np.load(out_dir / "segments" / f"{case_id}.npz")
        rows.append(
            pd.DataFrame(
                {
                    "caseid": case_id,
                    "index": data["index"],
                    "sbp": data["sbp"],
                    "dbp": data["dbp"],
                    "mbp": data["mbp"],
                    "pulse_rate": data["pulse_rate"],
                }
            )
        )
    labels = pd.concat(rows, ignore_index=True)
    labels = labels.merge(table[["caseid", "subjectid", "split", "age", "sex"]], on="caseid")
    labels.to_parquet(out_dir / "segment_labels.parquet", index=False)

    print(f"\nkept segments {len(labels):,} (split with seed {config['seed']}; no subject in two parts)")
    print(
        f"{'part':12s} {'subj.':>6s} {'cases':>6s} {'segments':>10s} {'age':>10s} {'female':>6s} "
        f"{'SBP mean±SD':>14s} {'SBP 5-95%':>12s} {'<100':>6s} {'>=140':>6s} {'DBP mean±SD':>13s}"
    )
    for name in (*SPLIT_NAMES, "all"):
        part = labels if name == "all" else labels[labels["split"] == name]
        subjects = part.drop_duplicates("subjectid")
        print(
            f"{name:12s} {part['subjectid'].nunique():>8d} {part['caseid'].nunique():>8d} {len(part):>12,} "
            f"{subjects['age'].mean():>6.1f}±{subjects['age'].std():<4.1f} {np.mean(subjects['sex'] == 'F'):>7.1%} "
            f"{part['sbp'].mean():>9.1f}±{part['sbp'].std():<5.1f} "
            f"{part['sbp'].quantile(0.05):>6.0f}-{part['sbp'].quantile(0.95):<5.0f} "
            f"{np.mean(part['sbp'] < PRINT_LOW):>6.1%} {np.mean(part['sbp'] >= PRINT_HIGH):>6.1%} "
            f"{part['dbp'].mean():>8.1f}±{part['dbp'].std():<5.1f}"
        )


if __name__ == "__main__":
    main()
