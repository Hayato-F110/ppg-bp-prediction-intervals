# Results of the paper, for comparison

The values printed in the paper, as CSV files (aggregate results only; no patient data).

| File | Contents |
|---|---|
| `table1_characteristics.csv` | Table 1: patients and segments of each part |
| `table2_methods.csv` | Table 2: patient-level coverage by method (with 95% confidence intervals) and median width |
| `supplement_tables.md` | Tables S1–S8 of the supplementary material, as written by `scripts/make_tables.py` |
| `supplementary_data/*.csv` | The supplementary data of the paper, one file per sheet of the Excel workbook; `README.csv` gives the title, the description and the contents of each sheet |

After a reproduction (`scripts/run_all.py`), `scripts/compare_reference.py` compares these files with the ones the
pipeline writes to `results/tables/`. The waveform model is not exactly reproducible on several CPU threads, so its
rows may differ slightly.
