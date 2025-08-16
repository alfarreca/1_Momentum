# data_loader.py
import time, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytz
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st  # for cache + warnings

from analysis import calculate_momentum  # no circular import (analysis does not import this module)

# ====== CONFIG ======
MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 2.0)
CACHE_TTL = 3600 * 12
MAX_RETRIES = 3
TIMEZONE = 'America/New_York'

# yfinance cache folder (optional)
try:
    yf.set_tz_cache_location("cache")
except Exception:
    pass

# ---------- Symbol mapping ----------
def exchange_suffix(ex: str) -> str:
    suffix_map = {
        "ETR": "DE", "EPA": "PA", "LON": "L", "BIT": "MI", "STO": "ST",
        "SWX": "SW", "TSE": "TO", "TSX": "TO", "TSXV": "V", "ASX": "AX",
        "HKG": "HK", "CNY": "SS", "TORONTO": "TO"
    }
    return suffix_map.get(ex.upper(), "")

def map_to_yfinance_symbol(symbol: str, exchange: str) -> str:
    if exchange.upper() in ["NYSE", "NASDAQ"]:
        return symbol
    suffix = exchange_suffix(exchange)
    return f"{symbol}.{suffix}" if suffix else symbol

# ---------- Robust fetch ----------
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def _safe_yfinance_fetch(ticker, period="3mo"):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(_ticker: str, exchange: str, yf_symbol: str):
    """Fetch price history, compute indicators, and return one row dict or None."""
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = _safe_yfinance_fetch(ticker_obj)
        if hist.empty or len(hist) < 50:
            return None

        momentum_data = calculate_momentum(hist)
        if not momentum_data:
            return None

        current_price = hist['Close'].iloc[-1]
        five_day_change = ((current_price / hist['Close'].iloc[-5] - 1) * 100) if len(hist) >= 5 else None
        twenty_day_change = ((current_price / hist['Close'].iloc[-20] - 1) * 100) if len(hist) >= 20 else None

        return {
            "Symbol": _ticker,
            "Exchange": exchange,
            "Price": round(current_price, 2),
            "5D_Change": round(five_day_change, 1) if five_day_change is not None else None,
            "20D_Change": round(twenty_day_change, 1) if twenty_day_change is not None else None,
            **momentum_data,
            "Last_Updated": datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
            "YF_Symbol": yf_symbol,
        }
    except Exception as e:
        st.warning(f"Error processing {_ticker}: {str(e)}")
        return None

def read_uploaded_sheet(uploaded_file) -> pd.DataFrame:
    """Load and clean the selected sheet; enforce required columns."""
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    return pd.read_excel(xls, sheet_name=sheet_names[0]), sheet_names  # UI will pick later

def clean_symbols(df: pd.DataFrame) -> pd.DataFrame:
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df.dropna(subset=["Symbol", "Exchange"]).drop_duplicates("Symbol")
    dropped = before - len(df)
    return df, dropped

def enrich_with_yf_symbols(df: pd.DataFrame) -> pd.DataFrame:
    df["YF_Symbol"] = df.apply(lambda r: map_to_yfinance_symbol(r["Symbol"], r["Exchange"]), axis=1)
    return df

def fetch_all(df: pd.DataFrame) -> pd.DataFrame:
    """Parallel fetch across tickers."""
    rows = []
    total = len(df)
    progress = st.progress(0, text="Fetching ticker data...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"])
            for _, row in df.iterrows()
        ]
        for i, f in enumerate(as_completed(futures), start=1):
            data = f.result()
            if data:
                rows.append(data)
            progress.progress(i / total, text=f"Processed {i}/{total} tickers")

    progress.empty()
    return pd.DataFrame(rows)
