"""Run the analysis scripts in order.

Each step runs `python -X utf8 scripts/<name>.py` from the project root and stops the run if a step fails.
The wall-clock time of every step is appended to results/logs/run_all_times.csv.

Steps that download data (VitalDB, MIMIC-III) take hours and need a network connection; they are skipped
unless --download is given. The patient split (make_split) is fixed in data/processed/vitaldb/split.csv
and is skipped when that file exists.

usage:
    uv run python scripts/run_all.py --list
    uv run python scripts/run_all.py                    # everything after the downloads
    uv run python scripts/run_all.py --download         # including the downloads
    uv run python scripts/run_all.py --from evaluate_conformal --to make_tables
    uv run python scripts/run_all.py --optional         # also the optional checks
"""

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ppg_bp.config import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class Step:
    name: str
    stage: str
    description: str
    args: tuple[str, ...] = field(default_factory=tuple)
    download: bool = False
    optional: bool = False
    isolated: bool = False  # run with -I because the step reads downloaded third-party files


STEPS = (
    Step("build_vitaldb_segments", "data", "Download VitalDB, cut 10-s segments, reference values", download=True),
    Step("make_split", "data", "Patient-level split (fixed; skipped when split.csv exists)"),
    Step("validate_fiducials", "features", "Check the fiducial points against public annotations", isolated=True),
    Step("build_features", "features", "Features of all segments", ("--workers", "8")),
    Step("summarise_features", "features", "Missing rates, distributions, signal-quality tertiles"),
    Step("train_feature_models", "models", "Median, linear regression and LightGBM: training and validation"),
    Step("train_waveform_model", "models", "One-dimensional CNN: training and validation"),
    Step("plot_validation", "models", "Bland-Altman plots and errors by band"),
    Step("predict_calibration_test", "models", "Estimates for the calibration and test patients"),
    Step("evaluate_conformal", "intervals", "Intervals within VitalDB: overall, by subgroup, calibration size"),
    Step("evaluate_skewness_strata", "intervals", "Coverage by skewness tertile (planned sensitivity analysis)"),
    Step(
        "build_mimic_segments",
        "external",
        "Download MIMIC-III waveforms and cut segments",
        ("--subjects", "700", "--hours-limit", "7.5", "--workers", "6"),
        download=True,
    ),
    Step("predict_external", "external", "Features and estimates for MIMIC-III (models are not retrained)"),
    Step("evaluate_external", "external", "Across datasets: as is, weighted, recalibrated, shift"),
    Step("evaluate_selective", "intervals", "Withholding estimates with wide intervals"),
    Step("explain_models", "intervals", "SHAP for the estimate and the interval width"),
    Step("build_sensitivity_segments", "sensitivity", "Segments under other exclusion rules", download=True),
    Step("evaluate_sensitivity", "sensitivity", "Coverage under other exclusion rules"),
    Step("explore_review_analyses", "exploratory", "Test accuracy, shrinkage model, offset correction, differences"),
    Step("explore_review2_analyses", "exploratory", "Extremes, MBP bands, slope, pulse-rate groups, shrinkage fix"),
    Step("explore_model_comparison", "exploratory", "Waveform model minus feature model on the same patients"),
    Step("make_characteristics", "outputs", "Table 1 (characteristics of the datasets)"),
    Step("make_tables", "outputs", "Supplementary tables"),
    Step("plot_study_design", "outputs", "Fig. 1"),
    Step("plot_main_figures", "outputs", "Figs. 2, 4 and 7"),
    Step("plot_conformal", "outputs", "Fig. 3 and supplementary figures"),
    Step("plot_external", "outputs", "Figs. 5 and 6"),
    Step("tune_lightgbm", "optional", "Check that the LightGBM settings are not far from tuned ones", optional=True),
    Step("check_fiducials_vitaldb", "optional", "Fiducial points on a VitalDB sample", optional=True),
    Step("check_e_wave_timing", "optional", "Timing of the e wave by pulse rate", optional=True),
    Step("explore_required_width", "optional", "Interval width needed for nominal coverage by band", optional=True),
    Step("explore_clinical_threshold", "optional", "Intervals against the 140 mmHg threshold", optional=True),
)


def select(args) -> list[Step]:
    names = [s.name for s in STEPS]
    for name in (args.start, args.stop):
        if name and name not in names:
            raise SystemExit(f"unknown step: {name}")
    first = names.index(args.start) if args.start else 0
    last = names.index(args.stop) if args.stop else len(STEPS) - 1
    chosen = []
    for step in STEPS[first : last + 1]:
        if step.download and not args.download:
            continue
        if step.optional and not args.optional:
            continue
        chosen.append(step)
    return chosen


def run(step: Step, log_path: Path) -> None:
    split_file = PROJECT_ROOT / load_config()["paths"]["processed"] / "vitaldb" / "split.csv"
    if step.name == "make_split" and split_file.exists():
        print(f"[skip] make_split: {split_file.relative_to(PROJECT_ROOT)} exists (the split is fixed)")
        return
    command = [sys.executable, *(["-I"] if step.isolated else []), "-X", "utf8", f"scripts/{step.name}.py", *step.args]
    print(f"[run ] {step.name}: {step.description}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)  # noqa: S603 (fixed list of this project's scripts)
    seconds = time.perf_counter() - started
    with log_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.now().isoformat(timespec="seconds"), step.name, f"{seconds:.1f}", result.returncode]
        )
    if result.returncode != 0:
        raise SystemExit(f"step {step.name} failed with exit code {result.returncode}")
    print(f"[done] {step.name} in {seconds / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--from", dest="start", help="first step to run")
    parser.add_argument("--to", dest="stop", help="last step to run")
    parser.add_argument("--download", action="store_true", help="include the steps that download data")
    parser.add_argument("--optional", action="store_true", help="include the optional checks")
    args = parser.parse_args()

    if args.list:
        for number, step in enumerate(STEPS, 1):
            flags = " [download]" if step.download else " [optional]" if step.optional else ""
            print(f"{number:2d}. {step.name:28s} {step.stage:12s} {step.description}{flags}")
        return

    log_path = PROJECT_ROOT / load_config()["paths"]["results"] / "logs" / "run_all_times.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("time,step,seconds,returncode\n", encoding="utf-8")
    for step in select(args):
        run(step, log_path)


if __name__ == "__main__":
    main()
