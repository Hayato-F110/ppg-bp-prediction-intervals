"""Compare a reproduction with the results of the paper (reference_results/): tables 1 and 2, all supplementary tables
the supplementary data and the exploratory analyses of results/review4/.

CSV files are compared cell by cell, as text; the supplementary tables (Markdown) line by line. The values are those
printed in the paper, already rounded, so any difference shows. The waveform model is not exactly reproducible on
several CPU threads (README), so its rows may differ slightly.

usage: uv run python -X utf8 scripts/compare_reference.py
input: reference_results/, results/tables/
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

REFERENCE = PROJECT_ROOT / "reference_results"
RESULTS = PROJECT_ROOT / load_config()["paths"]["results"]
TABLES = RESULTS / "tables"
# reference file -> file written by the pipeline
PAIRS = {
    "table1_characteristics.csv": TABLES / "characteristics.csv",
    "table2_methods.csv": TABLES / "table2_methods.csv",
    "supplement_tables.md": TABLES / "supplement_tables.md",
    **{
        f"supplementary_data/{path.name}": TABLES / "supplementary_data" / path.name
        for path in sorted((REFERENCE / "supplementary_data").glob("*.csv"))
    },
    # exploratory analyses after the fourth (mock) review, cited in the text and the supplement
    **{
        f"review4/{path.name}": RESULTS / "review4" / path.name
        for path in sorted((REFERENCE / "review4").glob("*.csv"))
    },
}
SHOWN = 5  # differing lines printed per Markdown file


def compare_csv(reference_path, produced) -> int:
    reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    result = pd.read_csv(produced, dtype=str, keep_default_na=False)
    if reference.shape != result.shape or list(reference.columns) != list(result.columns):
        print(f"  shape or columns differ ({reference.shape} against {result.shape})")
        return 1
    return int((reference != result).to_numpy().sum())


def compare_lines(reference_path, produced) -> int:
    reference = reference_path.read_text(encoding="utf-8").splitlines()
    result = produced.read_text(encoding="utf-8").splitlines()
    differing = [(k + 1, a, b) for k, (a, b) in enumerate(zip(reference, result, strict=False)) if a != b]
    for number, a, b in differing[:SHOWN]:
        print(f"  line {number}: paper  {a[:100]}\n           here   {b[:100]}")
    return len(differing) + abs(len(reference) - len(result))


def main() -> None:
    differing = 0
    for name, produced in PAIRS.items():
        if not produced.exists():
            print(f"{name}: missing {produced.relative_to(PROJECT_ROOT)}")
            differing += 1
            continue
        compare = compare_lines if name.endswith(".md") else compare_csv
        count = compare(REFERENCE / name, produced)
        differing += count > 0
        unit = "lines" if name.endswith(".md") else "cells"
        print(f"{name}: {'identical' if count == 0 else f'{count} {unit} differ'}")
    print(f"\n{len(PAIRS)} files compared, {differing} with differences")
    if differing:
        raise SystemExit(1)  # so that run_all.py and other callers see a failed reproduction


if __name__ == "__main__":
    main()
