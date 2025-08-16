# data_loader.py
import time, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytz
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st

from analysis import calculate_momentum

MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 2.0)
CACHE_TTL = 3600 * 12
MAX_RETRIES = 3
TIMEZONE = 'America/New_York'

try:
    yf.set_tz_cache_location("cache")
except Exception:
    pass

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

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def _safe_yfinance_fetch(ticker, period="3mo"):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(_ticker: str, exchange: str, yf_symbol: str):
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

def read_uploaded_sheet(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0]), xls.sheet_names

def clean_symbols(df: pd.DataFrame):
    for col in ["Symbol", "Exchange"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    before = len(df)
    need_cols = [c for c in ["Symbol", "Exchange"] if c in df.columns]
    df = df.dropna(subset=need_cols).drop_duplicates("Symbol")
    dropped = before - len(df)
    return df, dropped

def enrich_with_yf_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if "YF_Symbol" not in df.columns:
        df["YF_Symbol"] = df.apply(lambda r: map_to_yfinance_symbol(str(r["Symbol"]), str(r["Exchange"])), axis=1)
    return df

def fetch_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    if total == 0:
        return pd.DataFrame()
    progress = st.progress(0, text="Fetching ticker data...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(get_ticker_data, row.get("Symbol",""), row.get("Exchange",""), row.get("YF_Symbol",""))
            for _, row in df.iterrows()
        ]
        for i, f in enumerate(as_completed(futures), start=1):
            data = f.result()
            if data:
                # Keep original metadata columns if present in input df (Sector/Industry/Country)
                for meta_col in ["Sector", "Industry", "Country"]:
                    if meta_col in df.columns:
                        # Value from the source row by symbol (safe merge later is another option)
                        pass
                rows.append(data)
            progress.progress(i / total, text=f"Processed {i}/{total} tickers")

    progress.empty()
    out = pd.DataFrame(rows)

    # If the input df had metadata columns, merge them back by Symbol
    meta_cols = [c for c in ["Sector", "Industry", "Country"] if c in df.columns]
    if meta_cols:
        out = out.merge(df[["Symbol"] + meta_cols].drop_duplicates("Symbol"), on="Symbol", how="left")

    return out
