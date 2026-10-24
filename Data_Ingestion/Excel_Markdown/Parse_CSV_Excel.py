import csv
import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _detect_csv_delimiter(file_path: str, encoding: str = "utf-8") -> str:
    """
    Detects whether a CSV/TSV file is comma-separated (',') or tab-separated ('\\t').
    Falls back gracefully on single-column, header-only, or single-cell files where
    csv.Sniffer may raise an error.
    """
    if Path(file_path).suffix.lower() == ".tsv":
        return "\t"

    try:
        with open(file_path, "r", encoding=encoding, errors="replace", newline="") as f:
            sample = f.read(8192)
    except OSError as e:
        logger.warning("Could not sample file %s for delimiter detection: %s", file_path, e)
        return ","

    if not sample.strip():
        return ","

    # Try csv.Sniffer restricted to comma and tab delimiters
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        if dialect.delimiter in (",", "\t"):
            return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: compare occurrences of tab vs comma on the first non-empty line
    for line in sample.splitlines():
        if line.strip():
            tab_count = line.count("\t")
            comma_count = line.count(",")
            if tab_count > comma_count:
                return "\t"
            return ","

    return ","


def _clean_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a raw DataFrame read with header=None:
    - Converts whitespace-only strings to NA
    - Drops completely empty rows and completely empty columns
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    # Replace whitespace-only strings with NA so blank-looking cells are treated as empty
    cleaned = raw_df.apply(
        lambda col: col.map(
            lambda val: pd.NA
            if (val is None or (isinstance(val, str) and val.strip() == ""))
            else val
        )
    )
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def _dataframe_to_markdown(cleaned_df: pd.DataFrame) -> str:
    """
    Converts a non-empty cleaned DataFrame (read with header=None) into a Markdown table.
    Handles:
    1. Standard multi-row tables (first non-empty row used as header).
    2. Header-only sheets (1 row with multiple columns -> valid empty Markdown table with header).
    3. Single cell in the middle of the sheet (1x1 grid -> valid Markdown table).
    4. Mid-sheet data where some header cells are blank (fills blank headers cleanly).
    """
    num_rows, _ = cleaned_df.shape

    # Extract the first available row as column headers
    raw_headers = cleaned_df.iloc[0].tolist()
    headers: List[str] = []
    for idx, val in enumerate(raw_headers, start=1):
        if pd.isna(val) or str(val).strip() == "":
            headers.append(f"Column_{idx}")
        else:
            headers.append(str(val).strip().replace("\n", " "))

    # Deduplicate identical column names if any exist
    seen = {}
    unique_headers: List[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)

    if num_rows == 1:
        # Either header-only row OR a single cell in the middle of the sheet
        out_df = pd.DataFrame(columns=unique_headers)
        return out_df.to_markdown(index=False)

    # Multi-row table: remaining rows are data rows
    data_df = cleaned_df.iloc[1:].copy()
    data_df.columns = unique_headers
    data_df = data_df.apply(
        lambda col: col.map(
            lambda v: "" if pd.isna(v) else str(v).strip().replace("\n", " ")
        )
    )
    return data_df.to_markdown(index=False)


def _parse_csv_to_markdown(file_path: str) -> str:
    """
    Reads a CSV or TSV file (comma- or tab-separated) and returns its Markdown representation.
    """
    sep = _detect_csv_delimiter(file_path)

    try:
        raw_df = pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            dtype=object,
            skip_blank_lines=True,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError:
        raw_df = pd.DataFrame()

    cleaned_df = _clean_raw_dataframe(raw_df)
    if cleaned_df.empty:
        file_stem = Path(file_path).stem
        return f"### {file_stem}\n\n{file_stem} : blank"

    return _dataframe_to_markdown(cleaned_df)


def _parse_excel_to_markdown(file_path: str) -> str:
    """
    Reads an Excel workbook (.xlsx / .xls) and converts sheets to Markdown:
    - Places the sheet name at the top of each sheet's Markdown block.
    - Treats header-only sheets and single-cell-in-the-middle sheets as valid.
    - Skips Sheet 2, Sheet 3, etc. if they are completely blank.
    - If Sheet 1 is completely blank, outputs '<Sheet 1 Name> : blank' in Markdown.
    - Appends 'Page <X> of <total_valid_sheets>' at the bottom of each rendered sheet.
    """
    with pd.ExcelFile(file_path) as xls:
        sheet_names = xls.sheet_names
        if not sheet_names:
            return ""

        rendered_sheets: List[Tuple[str, str]] = []

        for idx, sheet_name in enumerate(sheet_names):
            raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=object)
            cleaned_df = _clean_raw_dataframe(raw_df)

            if cleaned_df.empty:
                # If Sheet 1 (index 0) is completely blank, keep it and mark as blank.
                # If Sheet 2, 3, ... (index >= 1) is completely blank, skip it entirely.
                if idx == 0:
                    sheet_md = f"### {sheet_name}\n\n{sheet_name} : blank"
                    rendered_sheets.append((sheet_name, sheet_md))
                else:
                    logger.info("Skipping completely blank sheet '%s' (sheet #%d).", sheet_name, idx + 1)
                    continue
            else:
                table_md = _dataframe_to_markdown(cleaned_df)
                sheet_md = f"### {sheet_name}\n\n{table_md}"
                rendered_sheets.append((sheet_name, sheet_md))

    total_sheets = len(rendered_sheets)
    page_blocks: List[str] = []
    for page_num, (_, sheet_md) in enumerate(rendered_sheets, start=1):
        page_blocks.append(f"{sheet_md}\n\nPage {page_num} of {total_sheets}\n\n")

    return "".join(page_blocks).strip()


def parse_tabular_file(file_path: str) -> str:
    """
    Parses an Excel (.xlsx, .xls) or CSV/TSV (.csv, .tsv) file into Markdown using pandas.

    Rules:
    - CSV (comma- or tab-separated): converts directly to Markdown and returns the Markdown string.
    - Excel (.xlsx, .xls):
      * Includes the sheet name at the top of each sheet's Markdown section.
      * Treats sheets with only a header row OR only a single cell anywhere in the sheet as valid.
      * Omits subsequent sheets (Sheet 2, Sheet 3, ...) if they are completely blank.
      * If Sheet 1 is completely blank, outputs '<Sheet 1 Name> : blank' in Markdown.
      * Appends 'Page <n> of <total_sheets>' at the end of each sheet block.

    Args:
        file_path (str): Path to the .xlsx, .xls, .csv, or .tsv file.

    Returns:
        str: Parsed Markdown representation.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext in (".csv", ".tsv"):
        return _parse_csv_to_markdown(file_path)
    elif ext in (".xlsx", ".xls"):
        return _parse_excel_to_markdown(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected '.xlsx', '.xls', '.csv', or '.tsv'."
        )


if __name__ == "__main__":
    sample_path = "sample.xlsx"
    if os.path.exists(sample_path):
        markdown_output = parse_tabular_file(sample_path)
        print("Final Markdown Content Preview")
        print(markdown_output)