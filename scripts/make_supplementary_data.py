"""Supplementary data of the paper (CSV files): the exact values behind the figures, one file per analysis.

The values are copied by the program from the output of scripts/make_tables.py (results/tables/supplement_tables.md),
not by hand. Files are named by their content, not by table numbers. The title (at most 30 characters, as IOP asks)
and a description (at most 30 words) are in README.csv, with the contents of each file. The paper cites these CSV
files in the code repository (reference_results/supplementary_data/); there is no Excel version (one format only).

usage: uv run python -X utf8 scripts/make_supplementary_data.py
input: results/tables/supplement_tables.md
output: results/tables/supplementary_data/*.csv (one file per analysis, with README.csv); the files of the paper
        are in reference_results/supplementary_data/
"""

import re

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

TABLES = PROJECT_ROOT / load_config()["paths"]["results"] / "tables"
SOURCE = TABLES / "supplement_tables.md"
CSV_DIR = TABLES / "supplementary_data"  # open format, as IOP's data policy prefers
TITLE = "Supplementary data"
DESCRIPTION = (
    "Exact values behind the figures: patient-level coverage, interval width and errors of the prediction intervals, "
    "by method, dataset and subgroup."
)
# file name -> (start of the heading of the table in make_tables.py, part (a), (b), ... of that table; None: the first)
FILES = {
    "coverage_overall": ("## Table S5.", None),
    "coverage_by_band": ("## Table S6.", None),
    "coverage_by_band_dbp": ("## Table S6.", 1),
    "subgroups": ("## Table S7.", None),
    "true_vs_predicted": ("## Table S8.", None),
    "recalibration": ("## Table S10.", None),
    "withholding": ("## Table S11.", None),
    "differences_methods": ("## Table S12.", None),
    "alternatives": ("## Table S12.", 1),
    "differences_models": ("## Table S12.", 2),
    "reference_reading": ("## Table S13.", 2),
    "accuracy": ("## Table S14.", None),
    "threshold_sorting": ("## Table S15.", None),
    "distribution_sbp": ("## Table S16.", 0),
    "distribution_coverage": ("## Table S16.", 1),
}


def blocks(text: str) -> dict[str, str]:
    """Heading -> the text from that heading to the next."""
    parts = re.split(r"^(## Table S\d+\..*)$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def frame(table: str) -> pd.DataFrame:
    header, *rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table.strip().splitlines()
        if not line.startswith("|---")
    ]
    return pd.DataFrame(rows, columns=header)


def main() -> None:
    found = blocks(SOURCE.read_text(encoding="utf-8"))
    readme = [("Title", TITLE), ("Description", DESCRIPTION), ("", "")]
    files = {}
    for name, (heading, part) in FILES.items():
        title, body = next((h, b) for h, b in found.items() if h.startswith(heading))
        tables = re.findall(r"(?:^\|.*\n)+", body, flags=re.MULTILINE)
        notes = [line for line in body.splitlines() if line and not line.startswith("|")]
        files[name] = frame(tables[part or 0])
        caption = re.sub(r"^## Table S\d+\. ", "", title)
        if part is not None:
            caption += " " + re.sub(r"\*\*", "", next(n for n in notes if n.startswith(f"**({'abc'[part]})")))
        explanations = [n for n in notes if not n.startswith("**(")]  # notes of the table (meaning of symbols etc.)
        readme.append((name, " ".join([caption, *explanations])))
    CSV_DIR.mkdir(exist_ok=True)
    pd.DataFrame(readme, columns=["File", "Contents"]).to_csv(CSV_DIR / "README.csv", index=False, lineterminator="\n")
    for name, table in files.items():
        table.to_csv(CSV_DIR / f"{name}.csv", index=False, lineterminator="\n")
    print(f"{CSV_DIR.relative_to(PROJECT_ROOT)}: {len(files)} CSV files and README.csv")


if __name__ == "__main__":
    main()
