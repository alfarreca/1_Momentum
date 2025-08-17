# analysis.py
import pandas as pd
import numpy as np

def calculate_di_crossovers(hist: pd.DataFrame, period: int = 14):
    high = hist['High']; low = hist['Low']; close = hist['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum.reduce([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()])
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    bullish_crossover = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    bearish_crossover = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
    return plus_di, minus_di, bullish_crossover, bearish_crossover

def calculate_momentum(hist: pd.DataFrame):
    """
    Returns a dict with indicator values, composite Momentum_Score, and a Score_Breakdown
    that shows how many points each component added.
    """
    if hist.empty or len(hist) < 50:
        return None

    close = hist['Close']; high = hist['High']; low = hist['Low']; volume = hist['Volume']

    # --- Core indicators ---
    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26; macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
    macd_line_above_signal = bool(macd.iloc[-1] > macd_signal.iloc[-1])

    vol_avg_20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = float(volume.iloc[-1] / vol_avg_20) if vol_avg_20 != 0 else 1.0

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = high.diff().where(lambda x: (x > 0) & (x > low.diff().abs()), 0)
    minus_dm = (-low.diff()).where(lambda x: (x > 0) & (x > high.diff().abs()), 0)
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean().iloc[-1] if not dx.isnull().all() else dx.mean()

    plus_di_c, minus_di_c, bullish_cross, bearish_cross = calculate_di_crossovers(hist)
    last_bullish = bool(bullish_cross.iloc[-1]); last_bearish = bool(bearish_cross.iloc[-1])

    # --- Scoring with per-component points ---
    score = 0
    ema_pts = 0; ema_rule = "None"
    if close.iloc[-1] > ema20 > ema50 > ema200:
        ema_pts = 30; ema_rule = "Close > EMA20 > EMA50 > EMA200"
    elif close.iloc[-1] > ema50 > ema200:
        ema_pts = 20; ema_rule = "Close > EMA50 > EMA200"
    elif close.iloc[-1] > ema200:
        ema_pts = 10; ema_rule = "Close > EMA200"
    score += ema_pts

    rsi_pts = 0; rsi_rule = "Else"
    if 60 <= rsi < 80:
        rsi_pts = 20; rsi_rule = "60 ≤ RSI < 80"
    elif (50 <= rsi < 60) or (80 <= rsi <= 90):
        rsi_pts = 10; rsi_rule = "50–60 or 80–90"
    score += rsi_pts

    macd_pts = 0; macd_rule = "Else"
    if (macd_hist > 0) and macd_line_above_signal:
        macd_pts = 15; macd_rule = "MACD hist > 0 & line > signal"
    score += macd_pts

    vol_pts = 0; vol_rule = "≤ 1.2"
    if volume_ratio > 1.5:
        vol_pts = 15; vol_rule = "> 1.5"
    elif volume_ratio > 1.2:
        vol_pts = 10; vol_rule = "1.2–1.5"
    score += vol_pts

    adx_pts = 0; adx_rule = "≤ 20"
    if adx > 30:
        adx_pts = 20; adx_rule = "> 30"
    elif adx > 25:
        adx_pts = 15; adx_rule = "> 25"
    elif adx > 20:
        adx_pts = 10; adx_rule = "> 20"
    score += adx_pts

    di_pts = 0; di_rule = "None"
    if last_bullish:
        di_pts = 10; di_rule = "Bullish crossover"
    elif last_bearish:
        di_pts = -10; di_rule = "Bearish crossover"
    score += di_pts

    # Clamp
    score = max(0, min(100, score))

    # Trend label
    trend = "↑ Strong" if score >= 80 else "↑ Medium" if score >= 60 else "↗ Weak" if score >= 40 else "→ Neutral"

    return {
        # Raw values (existing)
        "EMA20": round(float(ema20), 2), "EMA50": round(float(ema50), 2), "EMA200": round(float(ema200), 2),
        "RSI": round(float(rsi), 1), "MACD_Hist": round(float(macd_hist), 3),
        "ADX": round(float(adx), 1) if not np.isnan(adx) else None,
        "Volume_Ratio": round(float(volume_ratio), 2),
        "Momentum_Score": score,
        "Trend": trend,
        "Bullish_Crossover": last_bullish,
        "Bearish_Crossover": last_bearish,
        "plus_di_last": round(float(plus_di_c.iloc[-1]), 1) if not np.isnan(plus_di_c.iloc[-1]) else None,
        "minus_di_last": round(float(minus_di_c.iloc[-1]), 1) if not np.isnan(minus_di_c.iloc[-1]) else None,

        # NEW: flat breakdown fields (easy to filter/export)
        "Score_EMA": ema_pts,
        "Score_RSI": rsi_pts,
        "Score_MACD": macd_pts,
        "Score_Volume": vol_pts,
        "Score_ADX": adx_pts,
        "Score_DI": di_pts,

        # NEW: human-readable rules hit
        "Rule_EMA": ema_rule,
        "Rule_RSI": rsi_rule,
        "Rule_MACD": macd_rule,
        "Rule_Volume": vol_rule,
        "Rule_ADX": adx_rule,
        "Rule_DI": di_rule,

        # NEW: nested breakdown (nice in st.json)
        "Score_Breakdown": {
            "EMA_Alignment": {"points": ema_pts, "rule": ema_rule},
            "RSI":           {"points": rsi_pts, "rule": rsi_rule},
            "MACD":          {"points": macd_pts, "rule": macd_rule, "line_above_signal": macd_line_above_signal},
            "Volume":        {"points": vol_pts, "rule": vol_rule},
            "ADX":           {"points": adx_pts, "rule": adx_rule},
            "DI_Crossover":  {"points": di_pts, "rule": di_rule,
                               "latest_bullish": last_bullish, "latest_bearish": last_bearish},
            "Total":         score
        },
    }

def _apply_if_present(df: pd.DataFrame, col: str, values: list[str] | None):
    if values and col in df.columns and len(values) > 0:
        return df[df[col].isin(values)]
    return df

def filter_results(
    df: pd.DataFrame,
    min_score: int,
    exchange: str | None = None,
    sectors: list[str] | None = None,
    industries: list[str] | None = None,
    countries: list[str] | None = None,
):
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
