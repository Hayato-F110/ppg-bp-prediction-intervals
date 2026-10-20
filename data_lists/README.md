# Lists of the patients and records used in the paper

These lists fix which patients and records the analysis used. They contain identifiers only: no waveforms, no
pressures and no clinical data. They are written by `scripts/export_id_lists.py`.

| File | Contents | Used by |
|---|---|---|
| `vitaldb_split.csv` | VitalDB case ID, subject ID and part (`train`, `calibration`, `test`) of the 3,241 cases (3,134 subjects) with at least one kept 10-s segment | `scripts/make_split.py` uses this split and reports any difference from the seeded split |
| `mimic3wdb_subjects.csv` | Every MIMIC-III subject visited (1,613), in the order visited: the record read, the start of the first segment read (s), the outcome (`ok`, `no_signals`, or a network error) and the number of kept 10-s segments | `scripts/build_mimic_segments.py --subject-list data_lists/mimic3wdb_subjects.csv` reads the 454 subjects with outcome `ok` from the same records; 416 of them have kept segments |

The MIMIC-III subjects were those reached before a time limit for downloading, so the selection depended on download
speed. Subjects with a network error were not retried in the paper.

## Sources and licences

- VitalDB (https://vitaldb.net): case and subject IDs of the open dataset, used under its data use agreement and the
  CC BY-NC-SA 4.0 licence. This list is released under the same licence.
- MIMIC-III Waveform Database Matched Subset v1.0 (https://doi.org/10.13026/c2294b): subject and record names of the
  open-access database, under the Open Data Commons Open Database License (ODbL) v1.0. This list is released under the
  ODbL v1.0.
