# analysis.py
from __future__ import annotations

import math
from typing import Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------
# Indicator Calculations
# -----------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = low.diff().mul(-1)

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)

    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def calculate_di_crossovers(hist: pd.DataFrame, period: int = 14):
    high = hist['High']; low = hist['Low']; close = hist['Close']
    adx, plus_di, minus_di = _adx(high, low, close, period)
    bullish_crossover = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    bearish_crossover = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
    return plus_di, minus_di, bullish_crossover, bearish_crossover, adx


# -----------------------
# Momentum & Scoring
# -----------------------
def calculate_momentum(hist: pd.DataFrame) -> Dict[str, Any] | None:
    """Compute indicators & momentum score from a daily OHLCV history."""
    if hist is None or hist.empty:
        return None

    # Normalize column names
    cols = {c.lower(): c for c in hist.columns}
    for expected in ["Open", "High", "Low", "Close", "Volume"]:
        if expected not in hist.columns:
            # try case-insensitive
            for lc, orig in cols.items():
                if lc == expected.lower():
                    hist.rename(columns={orig: expected}, inplace=True)
                    break

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(hist.columns)):
        return None

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    vol = hist["Volume"].astype(float)

    if len(close) < 60:
        return None

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi = _rsi(close, 14)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    plus_di, minus_di, bull_x, bear_x, adx = calculate_di_crossovers(hist)

    # Latest values
    price = float(close.iloc[-1])
    ema20_l = float(ema20.iloc[-1])
    ema50_l = float(ema50.iloc[-1])
    ema200_l = float(ema200.iloc[-1])
    rsi_l = float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else np.nan
    macd_hist_l = float(macd_hist.iloc[-1])
    adx_l = float(adx.iloc[-1]) if not math.isnan(adx.iloc[-1]) else np.nan
    plus_di_l = float(plus_di.iloc[-1])
    minus_di_l = float(minus_di.iloc[-1])
    bull_last = bool(bull_x.iloc[-1]) if len(bull_x) else False
    bear_last = bool(bear_x.iloc[-1]) if len(bear_x) else False

    # --- Scoring ---
    score = 0
    # 1) Price vs EMA
    ema_pts = 0
    if price > ema20_l: ema_pts += 10
    if price > ema50_l: ema_pts += 10
    if price > ema200_l: ema_pts += 10
    if (ema20_l > ema50_l > ema200_l): ema_pts += 10
    score += ema_pts

    # 2) RSI zones (per user spec)
    rsi_pts = 0
    if 60 <= rsi_l < 80:
        rsi_pts = 20
    elif (50 <= rsi_l < 60) or (80 <= rsi_l < 90):
        rsi_pts = 10
    elif 40 <= rsi_l < 50:
        rsi_pts = 0
    elif 30 <= rsi_l < 40:
        rsi_pts = -5
    elif rsi_l < 30:
        rsi_pts = -10
    if rsi_l >= 90:
        rsi_pts -= 5  # exhaustion penalty
    score += rsi_pts

    # 3) MACD histogram
    macd_pts = 10 if macd_hist_l > 0 else 0
    score += macd_pts

    # 4) ADX strength
    adx_pts = 0
    if adx_l >= 60:
        adx_pts = 20
    elif 40 <= adx_l < 60:
        adx_pts = 15
    elif 25 <= adx_l < 40:
        adx_pts = 10
    elif 15 <= adx_l < 25:
        adx_pts = 5
    score += adx_pts

    # 5) DI crossover
    di_pts = 0
    if bull_last: di_pts += 5
    if bear_last: di_pts -= 5
    score += di_pts

    # Clamp
    score = int(max(0, min(100, score)))

    # Trend label
    if price > ema50_l > ema200_l and rsi_l >= 60:
        trend = "↑ Strong"
    elif price > ema200_l and rsi_l >= 50:
        trend = "↑ Medium"
    elif price >= ema200_l:
        trend = "↔ Neutral"
    else:
        trend = "↓ Weak"

    # Volume ratio
    vol_ratio = float((vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])) if vol.rolling(20).mean().iloc[-1] > 0 else np.nan

    momentum: Dict[str, Any] = {
        "EMA20": round(ema20_l, 2),
        "EMA50": round(ema50_l, 2),
        "EMA200": round(ema200_l, 2),
        "RSI": round(rsi_l, 1) if not math.isnan(rsi_l) else None,
        "MACD_Hist": round(macd_hist_l, 3),
        "ADX": round(adx_l, 1) if not math.isnan(adx_l) else None,
        "Volume_Ratio": round(vol_ratio, 2) if not math.isnan(vol_ratio) else None,
        "plus_di_last": round(plus_di_l, 1),
        "minus_di_last": round(minus_di_l, 1),
        "Bullish_Crossover": bool(bull_last),
        "Bearish_Crossover": bool(bear_last),
        "Momentum_Score": score,
        "Trend": trend,
        # Points breakdown (flat columns for easy display)
        "EMA_Points": ema_pts,
        "RSI_Points": rsi_pts,
        "MACD_Points": macd_pts,
        "ADX_Points": adx_pts,
        "DI_Points": di_pts,
    }
    return momentum


# -----------------------
# Filtering
# -----------------------
def _apply_if_present(df: pd.DataFrame, col: str, allowed: Optional[Iterable[str]]):
    if allowed and col in df.columns:
        return df[df[col].isin(list(allowed))]
    return df


def filter_results(
    df: pd.DataFrame,
    min_score: int = 60,
    exchange: Optional[str] = None,
    sectors: Optional[Iterable[str]] = None,
    industries: Optional[Iterable[str]] = None,
    countries: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if exchange and exchange != "All" and "Exchange" in out.columns:
        out = out[out["Exchange"] == exchange]
    out = _apply_if_present(out, "Sector", sectors)
    out = _apply_if_present(out, "Industry", industries)
    out = _apply_if_present(out, "Country", countries)
    if "Momentum_Score" in out.columns:
        out = out[out["Momentum_Score"] >= min_score]
    return out
