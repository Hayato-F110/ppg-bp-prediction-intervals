# Lists of the patients and records used in the paper

These lists fix which patients and records the analysis used. They contain identifiers only: no waveforms, no
pressures and no clinical data. They are written by `scripts/export_id_lists.py`.

| File | Contents | Used by |
|---|---|---|
| `vitaldb_split.csv` | Columns `caseid`, `subjectid`, `split`: the part (`train`, `calibration`, `test`) of the 3,241 cases (3,134 subjects) with at least one kept 10-s segment | `scripts/make_split.py` uses this split and reports any difference from the seeded split |
| `vitaldb_cases.csv` | Columns `caseid`, `subjectid`, `n_segments`, `ok`: every VitalDB case downloaded (3,459), with the number of 10-s segments cut and the number kept | `scripts/make_split.py` compares a new download with this list case by case |
| `mimic3wdb_subjects.csv` | Columns `subject`, `record`, `start_sec`, `status`, `ok`: every MIMIC-III subject visited (1,613), in the order visited, with the record read, the start of the first segment read (s), the outcome (`ok`, `no_signals`, or `error:` and the network error) and the number of kept 10-s segments | `scripts/build_mimic_segments.py --subject-list data_lists/mimic3wdb_subjects.csv` reads the 454 subjects with status `ok` from the same records (416 of them have kept segments) and compares `start_sec` and `ok` with this list |

The MIMIC-III subjects were those reached before a time limit for downloading, so the selection depended on download
speed. Subjects with a network error (36) were not retried in the paper; 1,123 subjects had no record with both
signals.

## Sources and licences

- VitalDB (https://vitaldb.net): case and subject IDs of the open dataset, used under its data use agreement and the
  CC BY-NC-SA 4.0 licence. This list is released under the same licence.
- MIMIC-III Waveform Database Matched Subset v1.0 (https://doi.org/10.13026/c2294b): subject and record names of the
  open-access database, under the Open Data Commons Open Database License (ODbL) v1.0. This list is released under the
  ODbL v1.0.
