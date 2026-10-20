# Prediction intervals for calibration-free cuffless blood pressure


Code for the paper *Where Do Prediction Intervals Fail for Calibration-Free Cuffless Blood Pressure? A Patient-Level Analysis Within and Across Datasets* (Hayato Fukushima; submitted to Advanced Biomedical Engineering).

Two models estimate blood pressure from the photoplethysmogram (PPG) alone: a feature model (LightGBM on 42 PPG features) and a waveform model (a one-dimensional CNN). Conformal prediction gives each estimate a prediction interval. The code evaluates how often these intervals contain the true pressure, per patient and by range of the true pressure, within VitalDB and on MIMIC-III.

The repository contains code and settings only. No data, model or result is included.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the exact versions recorded in `uv.lock`.
- About 16 GB of memory. No GPU is needed.
- Disk space for the processed data (VitalDB segments: tens of GB). Exact figure: not measured.

```bash
uv sync
uv run pytest
```

## Data

The data are not redistributed. Obtain them from the original sources and accept their terms.

| Dataset | Source | Licence | How it is read |
|---|---|---|---|
| VitalDB | <https://vitaldb.net/> | Data use agreement; CC BY-NC-SA 4.0 | Downloaded by `scripts/build_vitaldb_segments.py` with the `vitaldb` package (tracks SNUADC/PLETH and SNUADC/ART) |
| MIMIC-III Waveform Database Matched Subset | <https://physionet.org/content/mimic3wdb-matched/1.0/> | ODbL 1.0 | Downloaded by `scripts/build_mimic_segments.py` (open access; no clinical data are used) |
| PPG-BP annotations (pyPPG) | see `scripts/validate_fiducials.py` | see source | Used only to validate the fiducial-point detector |

The patient split is fixed in `data/processed/vitaldb/split.csv`, written by `scripts/make_split.py`. It was fixed before any model was fitted.

## Running the analysis

`scripts/run_all.py` runs the steps in order and records the time of each step in `results/logs/run_all_times.csv`.

```bash
uv run python scripts/run_all.py --list        # the steps
uv run python scripts/run_all.py --download    # everything, including the downloads
uv run python scripts/run_all.py               # everything after the downloads
```

Times measured on one laptop PC (AMD Ryzen 5 7430U, 6 cores / 12 threads, 16 GB memory, no GPU), October 2026:

| Step | Time |
|---|---|
| VitalDB download and segmentation (3,459 cases, 17 GB) | about 2 h 20 min |
| MIMIC-III download and segmentation | about 7.5 h (limited by the transfer rate) |
| Features (8 processes) | about 67 min |
| Waveform model training | about 1 h; about 2 h with `--deterministic` (one thread) |
| Feature models, estimates, evaluation, tables and figures | about 40 min in total; estimates for the calibration and test patients take about 22 min of this, every other step 6 min or less |

Downloads depend on the network. Each run of `run_all.py` appends its step times to `results/logs/run_all_times.csv`.

| Stage | Main outputs | In the paper |
|---|---|---|
| models | `results/feature_models/`, `results/waveform_model/` | Table S4, Fig. 2 |
| intervals | `results/conformal/coverage.csv`, `selective.csv` | Figs. 3, 4, 7; Tables S5–S7, S11 |
| external | `results/external/mimic3wdb/` | Figs. 5, 6; Table S10 |
| exploratory | `results/review*/` | Tables S8, S12, S13 |
| outputs | `results/figures/`, `results/tables/` | all figures and supplementary tables |

## Reproducibility

- Seeds are fixed in `configs/base.yaml`; all settings are in `configs/`.
- Results that depend on the feature model are reproduced exactly (checked byte for byte).
- The waveform model is not exactly reproducible on several CPU threads, because parallel sums are added in a varying order. `scripts/train_waveform_model.py --deterministic` trains on one thread and reproduces exactly, at about twice the time. The models of the paper were trained without this option. A deterministic run gave an SBP MAE of 13.37 mmHg on the validation patients, against 13.51 mmHg for the model of the paper; five runs with different seeds ranged from 13.34 to 13.50 mmHg.
- Analyses marked *exploratory* were added after the analysis plan, many in response to peer review.

## Layout

```
configs/          settings (YAML)
src/ppg_bp/       package: data, features, models, conformal prediction, evaluation
scripts/          analysis steps (run_all.py gives the order)
tests/            unit tests (uv run pytest)
```

## Names in the code and in the paper

| Code | Paper |
|---|---|
| `subject`, `subjectid` | patient (VitalDB `subjectid`; one patient can have several operations, `caseid`) |
| `cal` | calibration set (patients used to fit the conformal correction) |
| `ext` | external dataset (MIMIC-III) |
| `split` | split conformal prediction |
| `cqr` | conformalized quantile regression (CQR) |
| `mondrian_pred`, `mondrian_sqi`, `mondrian_pred_sqi` | group-wise conformal prediction by estimated SBP, by signal quality, or by both |
| `cqr_mondrian_pred_sqi` | CQR made group-wise by estimated SBP and signal quality |
| `weighted_split`, `weighted_cqr` | weighted conformal prediction (density-ratio weights) |
| `q050`, `q950` | estimated 5% and 95% quantiles |
| `level` | nominal coverage (for example 0.9) |

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the paper and the archived version of this code (Zenodo, version v1.0.0: https://doi.org/10.5281/zenodo.23513091; all versions: https://doi.org/10.5281/zenodo.23513090; see `CITATION.cff`).

The author used a generative AI assistant (Claude, Anthropic) to write and test this code under the author's direction, as disclosed in the paper.
