import streamlit as st
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_loader import load_excel, get_history, now_str
from visualization import display_results, display_symbol_details, download_button_for_df

# ========== APP CONFIG ==========
MAX_WORKERS = 8
CACHE_TTL = 3600 * 12  # 12h


def calculate_di_crossovers(hist: pd.DataFrame, period: int = 14):
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = np.maximum.reduce(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ]
    )
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

    bullish_crossover = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    bearish_crossover = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
    return plus_di, minus_di, bullish_crossover, bearish_crossover


def calculate_momentum(hist: pd.DataFrame):
    if hist.empty or len(hist) < 50:
        return None

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / 14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
    macd_line_above_signal = macd.iloc[-1] > macd_signal.iloc[-1]

    vol_avg_20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_avg_20 if vol_avg_20 != 0 else 1

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_dm = high.diff().where(lambda x: (x > 0) & (x > low.diff().abs()), 0)
    minus_dm = (-low.diff()).where(lambda x: (x > 0) & (x > high.diff().abs()), 0)
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean().iloc[-1] if not dx.isnull().all() else float("nan")

    plus_di_c, minus_di_c, bull_x, bear_x = calculate_di_crossovers(hist)
    last_bullish = bool(bull_x.iloc[-1])
    last_bearish = bool(bear_x.iloc[-1])

    score = 0
    if close.iloc[-1] > ema20 > ema50 > ema200:
        score += 30
    elif close.iloc[-1] > ema50 > ema200:
        score += 20
    elif close.iloc[-1] > ema200:
        score += 10

    if 60 <= rsi < 80:
        score += 20
    elif 50 <= rsi < 60 or 80 <= rsi <= 90:
        score += 10

    if macd_hist > 0 and macd_line_above_signal:
        score += 15

    if volume_ratio > 1.5:
        score += 15
    elif volume_ratio > 1.2:
        score += 10

    if adx > 30:
        score += 20
    elif adx > 25:
        score += 15
    elif adx > 20:
        score += 10

    if last_bullish:
        score += 10
    if last_bearish:
        score -= 10

    score = max(0, min(100, score))
    return {
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "EMA200": round(ema200, 2),
        "RSI": round(rsi, 1),
        "MACD_Hist": round(macd_hist, 3),
        "ADX": round(adx, 1) if not np.isnan(adx) else None,
        "Volume_Ratio": round(volume_ratio, 2),
        "Momentum_Score": score,
        "Trend": "↑ Strong"
        if score >= 80
        else "↑ Medium"
        if score >= 60
        else "↗ Weak"
        if score >= 40
        else "→ Neutral",
        "Bullish_Crossover": last_bullish,
        "Bearish_Crossover": last_bearish,
        "plus_di_last": round(plus_di_c.iloc[-1], 1) if not np.isnan(plus_di_c.iloc[-1]) else None,
        "minus_di_last": round(minus_di_c.iloc[-1], 1) if not np.isnan(minus_di_c.iloc[-1]) else None,
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_ticker_row(symbol: str, yf_symbol: str):
    """
    Pulls history for a ticker, computes momentum, and assembles a single result row (dict).
    Returns None if insufficient data.
    """
    hist = get_history(yf_symbol, period="3mo")
    if hist is None or len(hist) < 50:
        return None

    momentum = calculate_momentum(hist)
    if not momentum:
        return None

    current_price = hist["Close"].iloc[-1]
    five_day_change = ((current_price / hist["Close"].iloc[-5] - 1) * 100) if len(hist) >= 5 else None
    twenty_day_change = ((current_price / hist["Close"].iloc[-20] - 1) * 100) if len(hist) >= 20 else None

    return {
        "Symbol": symbol,
        "Price": round(current_price, 2),
        "5D_Change": round(five_day_change, 1) if five_day_change is not None else None,
        "20D_Change": round(twenty_day_change, 1) if twenty_day_change is not None else None,
        **momentum,
        "Last_Updated": now_str(),
        "YF_Symbol": yf_symbol,
    }


def main():
    st.set_page_config(page_title="S&P 500 Momentum Scanner", layout="wide")
    st.title("S&P 500 Momentum Scanner")

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
    if uploaded_file is None:
        st.warning("Please upload a .xlsx file with your tickers.")
        st.stop()

    # Let user pick sheet visually
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Select sheet to analyze", sheet_names, index=0)

    df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
    st.write(f"Loaded rows from '{selected_sheet}':", len(df_raw))
    st.dataframe(df_raw.head())

    # Validate and clean using data_loader logic
    try:
        # reuse loader for consistent checks
        df = df_raw.copy()
        from data_loader import validate_expected_columns  # local import to avoid circulars
        validate_expected_columns(df)

        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        before = len(df)
        df = df.dropna(subset=["Symbol"]).drop_duplicates("Symbol")
        st.write(f"Dropped rows: {before - len(df)} after cleaning.")
        df["YF_Symbol"] = df["Symbol"]
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar filters (keep behavior; Exchange filter removed)
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)

    # Fetch ticker data concurrently
    ticker_data = []
    progress = st.progress(0, text="Fetching ticker data...")
    total = len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(build_ticker_row, row["Symbol"], row["YF_Symbol"])
            for _, row in df.iterrows()
        ]
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result:
                ticker_data.append(result)
            progress.progress((i + 1) / total, text=f"Processed {i + 1}/{total} tickers")
    progress.empty()

    results_df = pd.DataFrame(ticker_data)

    if not results_df.empty:
        filtered = results_df[results_df["Momentum_Score"] >= min_score].copy()
    else:
        filtered = pd.DataFrame()

    st.session_state["raw_results_df"] = results_df.copy()
    st.session_state["filtered_results"] = filtered

    # Visualization
    display_results(filtered)

    if not filtered.empty:
        download_button_for_df(filtered)

        symbol_options = ["— Select a symbol —"] + filtered["Symbol"].tolist()
        last_selected = st.session_state.get("symbol_select", symbol_options[0])
        if last_selected not in symbol_options:
            last_selected = symbol_options[0]

        selected = st.selectbox(
            "Select a symbol for details",
            options=symbol_options,
            index=symbol_options.index(last_selected),
            key="symbol_select",
        )
        if selected != symbol_options[0]:
            display_symbol_details(selected, filtered)


if __name__ == "__main__":
    main()
