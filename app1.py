# app.py
# Permissive XLSX uploader + schema normalizer
# - Accepts files with missing/empty columns/cells
# - Auto-creates any missing required columns
# - Keeps blank rows (including blank Symbol) so you can edit/export if desired
# - Raises "Max rows per sheet" UI cap to ≤ 3000

import io
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Permissive XLSX Uploader", layout="wide")

REQUIRED_COLS: List[str] = [
    "Symbol",
    "Name",
    "Sector",
    "Industry",
    "Theme",
    "Country",
    "Notes",
    "Asset_Type",
]

MAX_SHEET_ROWS_HARD_CAP = 3000  # was 500

# ---------- Helpers ----------
def safe_read_excel(file_bytes: bytes) -> Tuple[pd.ExcelFile, List[str]]:
    """Read the uploaded Excel bytes into an ExcelFile and return sheet names."""
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    return xls, xls.sheet_names


def normalize_dataframe(df: pd.DataFrame, required_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure required columns exist; create missing ones with blank values.
    Keep all rows, including blank Symbol rows (user may want to edit/export).
    """
    created = []
    out = df.copy()

    # Normalize column names: strip whitespace and unify case
    out.columns = [str(c).strip() for c in out.columns]

    # Create missing columns with empty string (not NaN)
    for c in required_cols:
        if c not in out.columns:
            out[c] = ""
            created.append(c)

    # Reorder to put required columns first (extra columns preserved at the end)
    extra_cols = [c for c in out.columns if c not in required_cols]
    out = out[required_cols + extra_cols]

    # Normalize key fields as strings (avoid NaNs becoming 'nan')
    for c in required_cols:
        out[c] = out[c].astype(str).replace("nan", "").replace("None", "").str.strip()

    return out, created


def cap_rows(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Return the head of df limited to cap rows."""
    if cap is None or cap <= 0:
        return df
    return df.head(cap)


def download_xlsx(df: pd.DataFrame, filename: str = "normalized.xlsx") -> bytes:
    """Convert DataFrame to XLSX bytes for download."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data_1")
    bio.seek(0)
    return bio.read()


# ---------- UI ----------
st.title("📄 Permissive XLSX Uploader (Empty Cells & Columns Allowed)")

with st.sidebar:
    st.header("Settings")
    max_rows = st.number_input(
        "Max rows per sheet (≤ 3000)",
        value=500,
        step=50,
        min_value=100,
        max_value=3000,
        help="Cap how many rows are read per sheet to keep things snappy.",
    )
    show_preview_rows = st.number_input(
        "Preview rows",
        value=10,
        step=5,
        min_value=5,
        max_value=100,
        help="How many rows to show in the preview table.",
    )
    st.caption(f"Required columns (auto-created if missing): {', '.join(REQUIRED_COLS)}")

uploaded = st.file_uploader(
    "Upload an Excel file (.xlsx). Empty cells/columns are fine.",
    type=["xlsx"],
    accept_multiple_files=False,
)

if not uploaded:
    st.info("Upload an .xlsx file to begin.")
    st.stop()

# Read the workbook and list sheets
try:
    xls, sheet_names = safe_read_excel(uploaded.getvalue())
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

sheet = st.selectbox("Select sheet to analyze", options=sheet_names, index=0)

# Read the selected sheet (all cells as string to preserve empties)
try:
    df_raw = pd.read_excel(xls, sheet_name=sheet, dtype=str, keep_default_na=False)
except Exception as e:
    st.error(f"Failed to read sheet '{sheet}': {e}")
    st.stop()

# Cap rows to the configured limit (and hard cap)
user_limit = int(max_rows)
per_sheet_limit = min(user_limit, MAX_SHEET_ROWS_HARD_CAP)
df_raw = cap_rows(df_raw, per_sheet_limit)

st.write(f"Loaded rows from **'{sheet}'**: {len(df_raw):,}")

# Normalize / auto-create missing columns
df, created_cols = normalize_dataframe(df_raw, REQUIRED_COLS)

if created_cols:
    st.info("Created missing columns so the app can proceed: **" + ", ".join(created_cols) + "**")
else:
    st.success("All required columns were present.")

# Show a preview
st.subheader("Preview")
st.dataframe(df.head(int(show_preview_rows)), use_container_width=True)

# Optional: quick summary of blanks in required columns
with st.expander("Data quality summary (required columns)"):
    blanks = {c: int((df[c] == "").sum()) for c in REQUIRED_COLS}
    summary = pd.DataFrame.from_dict(blanks, orient="index", columns=["Blank cells"]).sort_values("Blank cells", ascending=False)
    st.dataframe(summary, use_container_width=True)

# Download normalized file
xlsx_bytes = download_xlsx(df, filename="normalized.xlsx")
st.download_button(
    label="💾 Download normalized XLSX",
    data=xlsx_bytes,
    file_name="normalized.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Notes: This uploader keeps blank rows (including blank Symbol). "
    "If you later run fetching/analysis, make sure to skip rows with blank Symbol."
)
