"""Write the lists of identifiers that fix which patients and records the paper used (no waveforms, no pressures).

- data_lists/vitaldb_split.csv: VitalDB case ID, subject ID and part (train, calibration, test) of every case with at
  least one kept segment. make_split.py checks its split against this list.
- data_lists/mimic3wdb_subjects.csv: every MIMIC-III subject visited, in the order visited, with the record read, the
  start of the first segment read (s), the outcome and the number of kept 10-s segments. With --subject-list,
  build_mimic_segments.py reads only the subjects of this list with outcome "ok", from the same record.

The identifiers are those of the public databases (VitalDB, CC BY-NC-SA 4.0 under its data use agreement; MIMIC-III
Waveform Database Matched Subset, ODbL 1.0); the lists are released under the same terms (data_lists/README.md).

usage: uv run python -X utf8 scripts/export_id_lists.py
input: data/processed/vitaldb/split.csv, data/processed/mimic3wdb/subject_summary.csv
output: data_lists/vitaldb_split.csv, data_lists/mimic3wdb_subjects.csv
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

OUT_DIR = PROJECT_ROOT / "data_lists"


def main() -> None:
    processed = PROJECT_ROOT / load_config()["paths"]["processed"]
    OUT_DIR.mkdir(exist_ok=True)
    split = pd.read_csv(processed / "vitaldb" / "split.csv")[["caseid", "subjectid", "split"]]
    split.to_csv(OUT_DIR / "vitaldb_split.csv", index=False, lineterminator="\n")
    mimic = pd.read_csv(processed / "mimic3wdb" / "subject_summary.csv", dtype={"record": str})
    mimic = mimic[["subject", "record", "start_sec", "status", "ok"]].fillna({"record": ""})
    mimic.to_csv(OUT_DIR / "mimic3wdb_subjects.csv", index=False, lineterminator="\n")
    print(
        f"VitalDB: {len(split)} cases, {split.subjectid.nunique()} subjects; "
        f"MIMIC-III: {len(mimic)} subjects visited, {int((mimic.ok > 0).sum())} with kept segments"
    )


if __name__ == "__main__":
    main()
