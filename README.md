# Prediction intervals for calibration-free cuffless blood pressure


Code for the paper *Where Do Prediction Intervals Fail for Calibration-Free Cuffless Blood Pressure? A Patient-Level Analysis Within and Across Datasets* (Hayato Fukushima; submitted to Physiological Measurement).

Two models estimate blood pressure from the photoplethysmogram (PPG) alone: a feature model (LightGBM on 42 PPG features) and a waveform model (a one-dimensional CNN). Conformal prediction gives each estimate a prediction interval. The code evaluates how often these intervals contain the true pressure, per patient and by range of the true pressure, within VitalDB and on MIMIC-III.

The repository contains code, settings, the lists of patients used (identifiers only) and the aggregate results of the paper for comparison. No waveform, model or patient-level result is included.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the exact versions recorded in `uv.lock`.
- About 16 GB of memory. No GPU is needed.
- Disk space: about 17 GB for the VitalDB download and tens of GB for the processed data (the exact total was not measured).

```bash
uv sync
uv run pytest
```

## Data

The data are not redistributed. Obtain them from the original sources and accept their terms.

| Dataset | Source | Licence | How it is obtained |
|---|---|---|---|
| VitalDB | <https://vitaldb.net/> | Data use agreement; CC BY-NC-SA 4.0 | Downloaded by `scripts/build_vitaldb_segments.py` with the `vitaldb` package (tracks SNUADC/PLETH and SNUADC/ART), October 2026 |
| MIMIC-III Waveform Database Matched Subset v1.0 | <https://doi.org/10.13026/c2294b> | ODbL 1.0 | Downloaded by `scripts/build_mimic_segments.py` (open access; no clinical data are used) |
| PPG-BP annotations of the pyPPG benchmark | <https://doi.org/10.5281/zenodo.10523285> | CC BY 4.0 | Download `PPG-BP_annot.zip` by hand (see below); used only to validate the fiducial-point detector |

The annotations are needed by `scripts/validate_fiducials.py` (step 3). Run these commands in bash (on Windows, Git Bash):

```bash
mkdir -p data/raw/pyppg_annot/extracted
curl -L -o data/raw/pyppg_annot/PPG-BP_annot.zip https://zenodo.org/records/10523285/files/PPG-BP_annot.zip
unzip data/raw/pyppg_annot/PPG-BP_annot.zip -d data/raw/pyppg_annot/extracted
```

### The patients of the paper

`data_lists/` fixes which patients and records were used (see `data_lists/README.md`):

- `vitaldb_split.csv`: the patient-level split (training, calibration, test), fixed before any model was fitted. `scripts/make_split.py` uses it and reports any difference from a new download.
- `mimic3wdb_subjects.csv`: the MIMIC-III subjects visited, with the record read. In the paper, subjects were visited in random order until a time limit, so the selection depended on download speed. `run_all.py` therefore reads the listed subjects with `--subject-list`, retries subjects that fail on a network error, and at the end compares the start and the number of kept segments of each subject with the list (all 454 subjects matched in our rerun).

## Running the analysis

`scripts/run_all.py` runs the steps in order and records the time of each step in `results/logs/run_all_times.csv`.

```bash
uv run python scripts/run_all.py --list        # the steps
uv run python scripts/run_all.py --download    # everything, including the downloads
uv run python scripts/run_all.py               # everything after the downloads
uv run python scripts/compare_reference.py     # compare tables 1 and 2 and the supplementary data with the paper
```

Times measured on one laptop PC (AMD Ryzen 5 7430U, 6 cores / 12 threads, 16 GB memory, no GPU), October 2026:

| Step | Time |
|---|---|
| VitalDB download and segmentation (3,459 cases, 17 GB) | about 2 h 20 min |
| MIMIC-III download and segmentation | 7.5 h in the paper, for 1,613 subjects visited; with the published list only the 454 subjects with signals are read (not timed) |
| Segments under other exclusion rules (sensitivity analysis; downloads VitalDB again) | not timed separately |
| Features (8 processes) | about 67 min |
| Waveform model training | about 1 h per run; the main run and five further seeds (table S8) take about 6 h; about 2 h per run with `--deterministic` (one thread) |
| Feature models, estimates, evaluation, tables and figures | about 40 min in total; estimates for the calibration and test patients take about 22 min of this |
| Exploratory analyses (`explore_*` steps) | not timed separately |

Downloads depend on the network.

## Where each result of the paper comes from

| In the paper | Script | Output |
|---|---|---|
| Table 1 | `make_characteristics.py` | `results/tables/characteristics.csv` |
| Table 2 | `make_table2.py` | `results/tables/table2_methods.csv` |
| Figure 1 | `plot_study_design.py` (the numbers are those of section 2) | `results/figures/study_design` |
| Figures 2, 4, 7; figures S1, S2 | `plot_main_figures.py` | `main_accuracy`, `main_subgroups`, `main_withholding`, `supp_bland_altman_test`, `supp_slope_curve` |
| Figure 3; figure S6 | `plot_conformal.py` | `phase4_coverage_by_sbp`, `phase4_calibration_size` |
| Figures 5, 6 | `plot_external.py` | `phase5_coverage_source_vs_target`, `phase5_recalibration` |
| Figures S3, S4, S5, S7 | `plot_supplement.py` | `supp_true_vs_predicted`, `supp_coverage_by_band`, `supp_differences`, `supp_reference_reading` |
| Figure S8 | `explain_models.py` | `phase6_shap` |
| Tables S1–S8 | `make_tables.py` | `results/tables/supplement_tables.md`: tables S1–S4 under the same numbers; S5–S8 are parts (b), (a), (d) and (e) of its last table |
| Supplementary data (Excel) | `make_supplementary_data.py` | `results/tables/supplementary_data.xlsx` and the same tables as CSV files |
| Numbers in the text | the evaluation and exploratory steps | `results/conformal/`, `results/external/`, `results/review*/`, `results/shap/` |

Figures are written as PNG, SVG, PDF and EPS under `results/figures/`. The files submitted with the paper carry the numbers of the figures instead: `figure1` (study_design), `figure2a_2b` (main_accuracy), `figure3a_3b` (phase4_coverage_by_sbp), `figure4` (main_subgroups), `figure5a_5b` (phase5_coverage_source_vs_target), `figure6a_6c` (phase5_recalibration) and `figure7a_7b` (main_withholding); the supplementary figures keep the names above.

## Reproducibility

- One seed (20261013, `configs/base.yaml`) fixes the split, the sampling of training segments, the bootstrap, the resplits and the order in which MIMIC-III subjects were visited. All settings are in `configs/`.
- Results that depend on the feature model are reproduced exactly: the features and all later steps were recomputed with the final code and were identical byte for byte, except for the column of computing times.
- The waveform model is not exactly reproducible on several CPU threads, because parallel sums are added in a varying order. `scripts/train_waveform_model.py --deterministic` trains on one thread and reproduces exactly, at about twice the time. The models of the paper were trained without this option. A deterministic run gave an SBP MAE of 13.37 mmHg on the validation patients, against 13.51 mmHg for the model of the paper; five runs with different seeds ranged from 13.34 to 13.50 mmHg.
- `reference_results/` holds tables 1 and 2 and the supplementary data of the paper as CSV files; `scripts/compare_reference.py` compares a reproduction with them.
- The analysis plan, fixed before any model was fitted, is in `analysis_plan.md` (an English translation of the Japanese original; the section numbers cited in the code refer to it). Analyses marked *exploratory* were added after the analysis plan, many in response to review.

## Layout

```
configs/            settings (YAML)
src/ppg_bp/         package: data, features, models, conformal prediction, evaluation
scripts/            analysis steps (run_all.py gives the order)
tests/              unit tests (uv run pytest)
data_lists/         the patients and records of the paper (identifiers only)
reference_results/  tables of the paper as CSV files, for comparison
assets/fonts/       TeX Gyre Heros, the Helvetica-compatible font of the figures (GUST Font License)
analysis_plan.md    the analysis plan, fixed before any model was fitted (English translation)
```

## Names in the code and in the paper

| Code | Paper |
|---|---|
| `subject`, `subjectid` | patient (VitalDB `subjectid`; one patient can have several operations, `caseid`) |
| `cal` | calibration set (patients used to fit the conformal correction) |
| `ext` | external dataset (MIMIC-III) |
| `split` | split conformal prediction |
| `cqr` | conformalized quantile regression (CQR) |
| `mondrian_pred`, `mondrian_sqi`, `mondrian_pred_sqi` | group-wise conformal prediction by predicted SBP, by signal quality, or by both |
| `cqr_mondrian_pred_sqi` | CQR made group-wise by predicted SBP and signal quality |
| `weighted_split`, `weighted_cqr` | weighted conformal prediction (density-ratio weights) |
| `q050`, `q950` | estimated 5% and 95% quantiles |
| `level` | nominal coverage (for example 0.9) |

## Licence and citation

Code: MIT licence (see `LICENSE`). `data_lists/` follows the licences of the source databases (see `data_lists/README.md`). Please cite the paper and the archived version of this code (Zenodo; all versions: https://doi.org/10.5281/zenodo.23513090; each release also has its own DOI on Zenodo).

The author used a generative AI assistant (Claude Opus 5.5, Anthropic) to write and test this code under the author's direction, as disclosed in the paper.
