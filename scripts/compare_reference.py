"""Compare a reproduction with the results of the paper (reference_results/): tables 1 and 2 and the supplementary data.

Each reference CSV file is compared cell by cell, as text, with the file of the same name written by the pipeline; the
values are those printed in the paper, already rounded. The waveform model is not exactly reproducible on several CPU
threads (README), so its rows may differ slightly.

usage: uv run python -X utf8 scripts/compare_reference.py
input: reference_results/, results/tables/
"""

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

REFERENCE = PROJECT_ROOT / "reference_results"
TABLES = PROJECT_ROOT / load_config()["paths"]["results"] / "tables"
# reference file -> file written by the pipeline
PAIRS = {
    "table1_characteristics.csv": TABLES / "characteristics.csv",
    "table2_methods.csv": TABLES / "table2_methods.csv",
    **{
        f"supplementary_data/{path.name}": TABLES / "supplementary_data" / path.name
        for path in sorted((REFERENCE / "supplementary_data").glob("*.csv"))
    },
}


def main() -> None:
    differing = 0
    for name, produced in PAIRS.items():
        reference = pd.read_csv(REFERENCE / name, dtype=str, keep_default_na=False)
        if not produced.exists():
            print(f"missing: {produced.relative_to(PROJECT_ROOT)}")
            differing += 1
            continue
        result = pd.read_csv(produced, dtype=str, keep_default_na=False)
        if reference.shape != result.shape or list(reference.columns) != list(result.columns):
            print(f"{name}: shape or columns differ ({reference.shape} against {result.shape})")
            differing += 1
            continue
        cells = int((reference != result).to_numpy().sum())
        differing += cells > 0
        print(f"{name}: {'identical' if cells == 0 else f'{cells} cells differ'}")
    print(f"\n{len(PAIRS)} files compared, {differing} with differences")


if __name__ == "__main__":
    main()
