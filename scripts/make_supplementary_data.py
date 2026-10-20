"""Supplementary data of the paper (Excel workbook): the exact values behind the figures, one sheet per analysis.

The values are copied by the program from the output of scripts/make_tables.py (results/tables/supplement_tables.md),
not by hand. Sheets are named by their content, not by table numbers. As IOP asks, the title (at most 30 characters)
and a description (at most 30 words) are inside the file, on the first sheet "README".

usage: uv run python -X utf8 scripts/make_supplementary_data.py
input: results/tables/supplement_tables.md
output: results/tables/supplementary_data.xlsx, and the same tables as CSV files (one per sheet, with README.csv)
        in results/tables/supplementary_data/; the CSV files of the paper are also in reference_results/
"""

import re

import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config

TABLES = PROJECT_ROOT / load_config()["paths"]["results"] / "tables"
SOURCE = TABLES / "supplement_tables.md"
OUT = TABLES / "supplementary_data.xlsx"
CSV_DIR = TABLES / "supplementary_data"  # open format, as IOP's data policy prefers
TITLE = "Supplementary data"
DESCRIPTION = (
    "Exact values behind the figures: patient-level coverage, interval width and errors of the prediction intervals, "
    "by method, dataset and subgroup."
)
# sheet name -> (start of the heading of the table in make_tables.py, part (a), (b), ... of that table; None: the first)
SHEETS = {
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
    sheets = {}
    for name, (heading, part) in SHEETS.items():
        title, body = next((h, b) for h, b in found.items() if h.startswith(heading))
        tables = re.findall(r"(?:^\|.*\n)+", body, flags=re.MULTILINE)
        notes = [line for line in body.splitlines() if line and not line.startswith("|")]
        sheets[name] = frame(tables[part or 0])
        caption = re.sub(r"^## Table S\d+\. ", "", title)
        if part is not None:
            caption += " " + re.sub(r"\*\*", "", next(n for n in notes if n.startswith(f"**({'abc'[part]})")))
        explanations = [n for n in notes if not n.startswith("**(")]  # notes of the table (meaning of symbols etc.)
        readme.append((name, " ".join([caption, *explanations])))
    with pd.ExcelWriter(OUT) as writer:
        pd.DataFrame(readme, columns=["Sheet", "Contents"]).to_excel(writer, sheet_name="README", index=False)
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    CSV_DIR.mkdir(exist_ok=True)
    pd.DataFrame(readme, columns=["Sheet", "Contents"]).to_csv(CSV_DIR / "README.csv", index=False, lineterminator="\n")
    for name, table in sheets.items():
        table.to_csv(CSV_DIR / f"{name}.csv", index=False, lineterminator="\n")
    print(f"{OUT.relative_to(PROJECT_ROOT)}: {len(sheets)} sheets; CSV files in {CSV_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
