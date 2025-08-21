# data_loader.py
from __future__ import annotations

import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

import pandas as pd
import pytz
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import streamlit as st

from analysis import calculate_momentum

# -----------------------
# Config
# -----------------------
MAX_WORKERS = 8
REQUEST_DELAY = (0.3, 1.0)
CACHE_TTL = 3600 * 6  # 6 hours


# -----------------------
# Helpers
# -----------------------
EX_SUFFIX = {
    # Americas
    "NYSE": "", "NYS": "", "NYQ": "", "NASDAQ": "", "NMS": "", "NCM": "", "ASE": "",
    "TSX": ".TO", "TSXV": ".V",
    # Europe majors
    "LSE": ".L", "LON": ".L",
    "XETRA": ".DE", "XETR": ".DE", "FRA": ".F",
    "PAR": ".PA", "EPA": ".PA",
    "MIL": ".MI", "BIT": ".MI",
    "AMS": ".AS", "AEX": ".AS",
    "BRU": ".BR",
    "LIS": ".LS",
    "MAD": ".MC",
    "VIE": ".VI",
    "SIX": ".SW", "ZRH": ".SW",
    "STO": ".ST", "HEL": ".HE", "CPH": ".CO", "OSL": ".OL",
    # Asia-Pac
    "TSE": ".T", "TYO": ".T",
    "HKEX": ".HK",
    "ASX": ".AX",
    "SGX": ".SI",
    "KOSPI": ".KS", "KSE": ".KS", "KOSDAQ": ".KQ",
}

def _map_to_yf_symbol(symbol: str, exchange: str | None, existing: str | None) -> str:
    """Return the Yahoo Finance symbol to use."""
    if existing and isinstance(existing, str) and existing.strip():
        return existing.strip()
    sx = (exchange or "").upper().strip()
    suffix = EX_SUFFIX.get(sx, "")
    sym = (symbol or "").strip()
    return f"{sym}{suffix}" if suffix and not sym.endswith(suffix) else sym


# -----------------------
# IO / Cleaning
# -----------------------
def read_uploaded_sheet() -> Optional[pd.DataFrame]:
    st.sidebar.subheader("Upload watchlist")
    file = st.sidebar.file_uploader("Upload XLSX/CSV with at least 'Symbol' and 'Exchange'", type=["xlsx", "csv"])
    if not file:
        return None
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return None


def clean_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Symbol", "Exchange"])
    out = df.copy()
    if "Symbol" not in out.columns:
        st.error("Missing required 'Symbol' column.")
        return pd.DataFrame(columns=["Symbol", "Exchange"])
    if "Exchange" not in out.columns:
        out["Exchange"] = ""
    out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
    out["Exchange"] = out["Exchange"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["Symbol"])
    out = out[out["Symbol"] != ""].drop_duplicates(subset=["Symbol"], keep="first")
    return out


def enrich_with_yf_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Symbol", "Exchange", "YF_Symbol"])
    out = df.copy()
    if "YF_Symbol" not in out.columns:
        out["YF_Symbol"] = [
            _map_to_yf_symbol(sym, ex, None) for sym, ex in zip(out["Symbol"], out["Exchange"])
        ]
    else:
        out["YF_Symbol"] = [
            _map_to_yf_symbol(sym, ex, yfs) for sym, ex, yfs in zip(out["Symbol"], out["Exchange"], out["YF_Symbol"])
        ]
    return out


# -----------------------
# Fetching
# -----------------------
class TemporaryFetchError(Exception):
    pass


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _cached_history(yf_symbol: str) -> pd.DataFrame:
    # Cache just the raw history to keep calculate_momentum fast and uncached.
    # Fetch 2 years to have enough bars for EMA200.
    hist = yf.download(yf_symbol, period="2y", interval="1d", auto_adjust=False, progress=False)
    # yf sometimes returns multiindex columns; normalize
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]
    return hist


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
    retry=retry_if_exception_type(TemporaryFetchError)
)
def _fetch_one_history(yf_symbol: str) -> pd.DataFrame:
    try:
        hist = _cached_history(yf_symbol)
        return hist
    except Exception as e:
        # Wrap in retry-able error
        raise TemporaryFetchError(str(e))


def _sleep_jitter():
    time.sleep(random.uniform(*REQUEST_DELAY))


def get_ticker_data(symbol: str, exchange: str, yf_symbol: str) -> Optional[Dict[str, Any]]:
    yfs = _map_to_yf_symbol(symbol, exchange, yf_symbol)
    _sleep_jitter()
    hist = _fetch_one_history(yfs)

    if hist is None or hist.empty or len(hist) < 60:
        return None

    # Require expected columns
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(hist.columns)):
        return None

    # Compute momentum & indicators
    momentum = calculate_momentum(hist)
    if not momentum:
        return None

    current_price = float(hist["Close"].iloc[-1])
    five_day_change = float(((hist["Close"].iloc[-1] / hist["Close"].iloc[-5]) - 1) * 100) if len(hist) >= 5 else None
    twenty_day_change = float(((hist["Close"].iloc[-1] / hist["Close"].iloc[-20]) - 1) * 100) if len(hist) >= 20 else None

    out: Dict[str, Any] = {
        "Symbol": symbol,
        "Exchange": exchange,
        "YF_Symbol": yfs,
        "Price": current_price,
        "5D_Change": round(five_day_change, 2) if five_day_change is not None else None,
        "20D_Change": round(twenty_day_change, 2) if twenty_day_change is not None else None,
    }
    out.update(momentum)
    return out


def fetch_all(df: pd.DataFrame) -> pd.DataFrame:
    """Parallel fetch across the ticker list and safe-merge any metadata back."""
    rows: List[Dict[str, Any]] = []
    total = len(df) if df is not None else 0
    if total == 0:
        return pd.DataFrame()

    progress = st.progress(0, text="Fetching ticker data...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                get_ticker_data,
                row.get("Symbol", ""),
                row.get("Exchange", ""),
                row.get("YF_Symbol", "")
            )
            for _, row in df.iterrows()
        ]
        for i, f in enumerate(as_completed(futures), start=1):
            try:
                data = f.result()
                if data:
                    rows.append(data)
            except Exception as e:
                # Keep going; log to UI
                st.info(f"Skipped one symbol due to error: {e}")
            finally:
                progress.progress(i / total, text=f"Processed {i}/{total} tickers")

    progress.empty()
    out = pd.DataFrame(rows)

    # Nothing fetched → avoid merge on missing 'Symbol'
    if out.empty or "Symbol" not in out.columns:
        return out

    # Merge back optional metadata if present in the uploaded sheet
    exclude = {"Symbol", "Exchange", "YF_Symbol"}
    meta_cols = [c for c in df.columns if c not in exclude]
    if meta_cols:
        right = df.loc[:, ["Symbol"] + meta_cols].drop_duplicates("Symbol")
        out = out.merge(right, on="Symbol", how="left")

    # Sort by score if present
    if "Momentum_Score" in out.columns:
        out = out.sort_values("Momentum_Score", ascending=False, kind="mergesort").reset_index(drop=True)

    return out
