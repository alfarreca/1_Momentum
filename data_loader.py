import time
import random
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import pytz
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ========== CONFIG ==========
REQUEST_DELAY: Tuple[float, float] = (0.5, 2.0)
MAX_RETRIES = 3
TIMEZONE = "America/New_York"

# Optional cache dir for yfinance timezone files
yf.set_tz_cache_location("cache")

# Exchange → Yahoo suffix map (covers original app + common variants seen in sheets)
SUFFIX_MAP = {
    "ETR": "DE", "EPA": "PA", "LON": "L", "BIT": "MI", "STO": "ST",
    "SWX": "SW", "TSE": "TO", "TSX": "TO", "TSXV": "V", "ASX": "AX",
    "HKG": "HK", "CNY": "SS", "TORONTO": "TO",
    # Common variants (Yahoo/CSV exports/platforms)
    "NYQ": "", "NYS": "", "NYSE": "", "NMS": "", "NASD": "", "NASDAQ": "",
    "LSE": "L", "PAR": "PA", "FRA": "F",
    "CCC": "", "SHH": "SS", "SHZ": "SZ"
}


def validate_expected_columns(df: pd.DataFrame) -> None:
    """
    Ensure the uploaded sheet has the required business columns (order can vary).
    Extra columns (e.g., Exchange) are allowed.
    """
    expected_cols = {"Symbol", "Name", "Sector", "Industry", "Theme", "Country", "Notes", "Asset_Type"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Uploaded sheet must contain these columns: {', '.join(sorted(expected_cols))}. "
            f"Missing: {', '.join(sorted(missing))}"
        )


def load_excel(uploaded_file) -> pd.DataFrame:
    """
    Load and sanitize the uploaded Excel file.
    - Validates required columns
    - Cleans Symbol (uppercase, strip)
    - Drops duplicates and blanks
    - Does NOT set YF_Symbol here (done in analysis.py so we can respect an optional Exchange column)
    Returns the cleaned DataFrame; extras like 'Exchange' are preserved if present.
    """
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    df = pd.read_excel(xls, sheet_name=sheet_names[0])

    validate_expected_columns(df)

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])

    df.attrs["sheet_names"] = sheet_names  # hint for UI to present options
    df.attrs["loaded_rows"] = before
    return df


def infer_yf_symbol(symbol: str, exchange: Optional[str]) -> str:
    """
    Build a Yahoo Finance symbol using an optional Exchange column.
    - If symbol already looks YF-ready (has '.' suffix or '-' like 'ETH-USD'), keep it.
    - If exchange is provided, append the appropriate suffix when needed.
    - Otherwise, return the symbol as-is (assume US).
    """
    if "." in symbol or "-" in symbol:
        return symbol
    if exchange:
        suffix = SUFFIX_MAP.get(str(exchange).upper(), "")
        return f"{symbol}.{suffix}" if suffix else symbol
    return symbol


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def _safe_yfinance_history(yf_symbol: str, period: str = "3mo") -> pd.DataFrame:
    """Fetch price history with retry and a small random delay."""
    time.sleep(random.uniform(*REQUEST_DELAY))
    return yf.Ticker(yf_symbol).history(period=period)


def get_history(yf_symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Wrapper for yfinance fetch that returns None on empty/bad history."""
    try:
        hist = _safe_yfinance_history(yf_symbol, period=period)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None


def now_str() -> str:
    """Timestamp string in configured timezone."""
    return datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M")
