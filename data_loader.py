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


def validate_expected_columns(df: pd.DataFrame) -> None:
    """
    Ensures the uploaded sheet has exactly the expected business columns (order can vary).
    Raises a ValueError with a descriptive message if something is missing.
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
    - Adds YF_Symbol == Symbol (no Exchange column anymore)
    Returns the cleaned DataFrame.
    """
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    # If there are multiple sheets, user can choose in the Streamlit layer; here we default to the first.
    df = pd.read_excel(xls, sheet_name=sheet_names[0])

    validate_expected_columns(df)

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    df["YF_Symbol"] = df["Symbol"]

    df.attrs["sheet_names"] = sheet_names  # hint for UI to present options
    df.attrs["loaded_rows"] = before
    return df


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def _safe_yfinance_history(yf_symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch price history with retry and a small random delay to be polite with yfinance.
    """
    time.sleep(random.uniform(*REQUEST_DELAY))
    return yf.Ticker(yf_symbol).history(period=period)


def get_history(yf_symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    Wrapper for yfinance fetch that returns None on empty/bad history.
    """
    try:
        hist = _safe_yfinance_history(yf_symbol, period=period)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None


def now_str() -> str:
    """
    Timestamp string in configured timezone.
    """
    return datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M")
