"""Write the lists of identifiers that fix which patients and records the paper used (no waveforms, no pressures).

- data_lists/vitaldb_split.csv: VitalDB case ID, subject ID and part (train, calibration, test) of every case with at
  least one kept segment. make_split.py uses this split.
- data_lists/vitaldb_cases.csv: every VitalDB case downloaded (3,459), with its subject ID, the number of 10-s segments
  (n_segments) and of kept segments (ok). make_split.py compares a new download with it case by case.
- data_lists/mimic3wdb_subjects.csv: every MIMIC-III subject visited, in the order visited, with the record read, the
  start of the first segment read in s (start_sec), the outcome (status) and the number of kept 10-s segments (ok).
  With --subject-list, build_mimic_segments.py reads only the subjects with status "ok", from the same record, and
  compares start_sec and ok with this list.

The identifiers are those of the public databases (VitalDB, CC BY-NC-SA 4.0 under its data use agreement; MIMIC-III
Waveform Database Matched Subset, ODbL 1.0); the lists are released under the same terms (data_lists/README.md).

usage: uv run python -X utf8 scripts/export_id_lists.py
input: data/processed/vitaldb/split.csv, cases.csv, case_summary.csv; data/processed/mimic3wdb/subject_summary.csv
output: data_lists/vitaldb_split.csv, data_lists/vitaldb_cases.csv, data_lists/mimic3wdb_subjects.csv
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

OUT_DIR = PROJECT_ROOT / "data_lists"


def main() -> None:
    processed = PROJECT_ROOT / load_config()["paths"]["processed"]
    OUT_DIR.mkdir(exist_ok=True)
    split = pd.read_csv(processed / "vitaldb" / "split.csv")[["caseid", "subjectid", "split"]]
    split.to_csv(OUT_DIR / "vitaldb_split.csv", index=False, lineterminator="\n")
    cases = pd.read_csv(processed / "vitaldb" / "cases.csv")[["caseid", "subjectid"]].merge(
        pd.read_csv(processed / "vitaldb" / "case_summary.csv")[["caseid", "n_segments", "ok"]], on="caseid"
    )
    cases.to_csv(OUT_DIR / "vitaldb_cases.csv", index=False, lineterminator="\n")
    mimic = pd.read_csv(processed / "mimic3wdb" / "subject_summary.csv", dtype={"record": str})
    mimic = mimic[["subject", "record", "start_sec", "status", "ok"]].fillna({"record": ""})
    mimic.to_csv(OUT_DIR / "mimic3wdb_subjects.csv", index=False, lineterminator="\n")
    print(
        f"VitalDB: {len(split)} cases, {split.subjectid.nunique()} subjects; "
        f"MIMIC-III: {len(mimic)} subjects visited, {int((mimic.ok > 0).sum())} with kept segments"
    )


if __name__ == "__main__":
    main()
