# data_loader.py
import time, random
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import pytz
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

REQUEST_DELAY: Tuple[float, float] = (0.5, 2.0)
MAX_RETRIES = 3
TIMEZONE = "America/New_York"

yf.set_tz_cache_location("cache")

# Exchange → Yahoo suffix map
SUFFIX_MAP = {
    "ETR":"DE","EPA":"PA","LON":"L","BIT":"MI","STO":"ST","SWX":"SW",
    "TSE":"TO","TSX":"TO","TSXV":"V","ASX":"AX","HKG":"HK","CNY":"SS","TORONTO":"TO",
    # common variants
    "NYQ":"","NYS":"","NYSE":"","NMS":"","NASD":"","NASDAQ":"",
    "LSE":"L","PAR":"PA","FRA":"F","CCC":"","SHH":"SS","SHZ":"SZ"
}

def validate_expected_columns(df: pd.DataFrame) -> None:
    expected = {"Symbol","Name","Sector","Industry","Theme","Country","Notes","Asset_Type"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(
            f"Uploaded sheet must contain: {', '.join(sorted(expected))}. "
            f"Missing: {', '.join(sorted(missing))}"
        )

def load_excel(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    df = pd.read_excel(xls, sheet_name=sheet_names[0])
    validate_expected_columns(df)
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df.dropna(subset=["Symbol"]).drop_duplicates("Symbol")
    df.attrs["sheet_names"] = sheet_names
    df.attrs["loaded_rows"] = before
    return df

def infer_yf_symbol(symbol: str, exchange: Optional[str]) -> str:
    # if already YF-formatted (e.g., DPM.TO or ETH-USD), keep it
    if "." in symbol or "-" in symbol:
        return symbol
    if exchange:
        suf = SUFFIX_MAP.get(str(exchange).upper(), "")
        return f"{symbol}.{suf}" if suf else symbol
    return symbol

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def _safe_yfinance_history(yf_symbol: str, period: str = "3mo") -> pd.DataFrame:
    time.sleep(random.uniform(*REQUEST_DELAY))
    return yf.Ticker(yf_symbol).history(period=period)

def get_history(yf_symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    try:
        hist = _safe_yfinance_history(yf_symbol, period=period)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None

def now_str() -> str:
    return datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M")
