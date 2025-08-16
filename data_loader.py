# data_loader.py
# Tolerant schema utilities for the Streamlit uploader flow.

from __future__ import annotations
import pandas as pd

# Expected schema (order is preserved when displaying)
EXPECTED_COLS = [
    "Symbol",
    "Name",
    "Sector",
    "Industry",
    "Theme",
    "Country",
    "Notes",
    "Asset_Type",
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and standardize column names to our canon."""
    canon = {c.lower().strip(): c for c in EXPECTED_COLS}

    new_cols = []
    for c in df.columns:
        key = str(c).lower().strip()
        # If it exactly matches one of our expected names (case/space-insensitive),
        # map it back to the canonical casing; else keep original.
        new_cols.append(canon.get(key, str(c).strip()))
    out = df.copy()
    out.columns = new_cols
    return out

def validate_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the dataframe schema tolerant:
      - Normalize column names (strip spaces, restore canonical names).
      - Auto-create any missing EXPECTED_COLS as empty columns.
      - Keep any extra columns untouched (e.g., Exchange).
      - Do NOT error on missing/blank values.
    Returns a COPY of df with ensured columns in a readable order.
    """
    out = _normalize_columns(df)

    # Add any missing columns as empty (None) with same index length
    missing = [c for c in EXPECTED_COLS if c not in out.columns]
    for c in missing:
        out[c] = pd.Series([None] * len(out), index=out.index)

    # Reorder so expected columns come first (in EXPECTED_COLS order), extras follow
    extras = [c for c in out.columns if c not in EXPECTED_COLS]
    out = out[EXPECTED_COLS + extras]

    return out

def clean_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the Symbol column and drop unusable rows.
    Keeps everything else (including blanks).
    """
    out = df.copy()
    out["Symbol"] = (
        out["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": ""})  # handle stringified NaNs
    )

    before = len(out)
    # Drop rows where Symbol is empty after cleaning
    out = out[out["Symbol"] != ""]
    dropped = before - len(out)

    return out, dropped
