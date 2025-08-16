# visualization.py
import streamlit as st
import pandas as pd

from data_loader import (
    read_uploaded_sheet, clean_symbols, enrich_with_yf_symbols, fetch_all
)
from analysis import filter_results

APP_TITLE = "S&P 500 Momentum Scanner (Modularized)"
REQUIRED_COLUMNS = {"Symbol", "Exchange"}   # Sector/Industry/Country are OPTIONAL

def _multiselect_all(label: str, options: list[str]) -> list[str]:
    """Multiselect with 'select all' default; returns selected values."""
    if not options:
        return []
    default = options  # select all by default
    return st.sidebar.multiselect(label, options=options, default=default)

def display_results(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.warning("No stocks match your current filters.")
        return
    left, right = st.columns(2)
    with left:
        st.metric("Stocks Found", len(filtered_df))
    with right:
        st.metric("Avg Momentum Score", round(filtered_df["Momentum_Score"].mean(), 1))
    st.dataframe(
        filtered_df.sort_values("Momentum_Score", ascending=False),
        use_container_width=True,
        height=600
    )

def display_symbol_details(filtered_df: pd.DataFrame, selected_symbol: str | None):
    if not selected_symbol:
        return
    try:
        row = filtered_df[filtered_df["Symbol"] == selected_symbol].iloc[0]
        st.subheader(f"{selected_symbol} — Detailed Snapshot")
        st.json(row.to_dict())
    except Exception as e:
        st.error(f"Error loading {selected_symbol}: {str(e)}")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
    if uploaded_file is None:
        st.info("Please upload a .xlsx file that includes at least: Symbol, Exchange. Optional columns: Sector, Industry, Country.")
        st.stop()

    # --- Load / pick sheet ---
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Select sheet to analyze", xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet)
    st.caption(f"Loaded rows from '{sheet}': {len(df)}")
    st.dataframe(df.head(), use_container_width=True)

    # --- Validate structure ---
    if not REQUIRED_COLUMNS.issubset(set(df.columns)):
        st.error(f"Uploaded sheet must contain these columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
        st.stop()

    # --- Clean & map ---
    df, dropped = clean_symbols(df)
    if dropped:
        st.caption(f"Dropped {dropped} rows after cleaning (NaN/duplicates).")

    # ===== Sidebar: pre-fetch filters to reduce API calls =====
    st.sidebar.header("Pre-Fetch Filters")
    # Build lists safely (ignore NaN)
    sector_opts   = sorted([s for s in df.get("Sector", pd.Series(dtype=str)).dropna().unique().tolist()])
    industry_opts = sorted([s for s in df.get("Industry", pd.Series(dtype=str)).dropna().unique().tolist()])
    country_opts  = sorted([s for s in df.get("Country", pd.Series(dtype=str)).dropna().unique().tolist()])

    # Only show filter if column exists; otherwise skip
    selected_sectors   = _multiselect_all("Sector", sector_opts) if "Sector" in df.columns else []
    selected_industry  = _multiselect_all("Industry", industry_opts) if "Industry" in df.columns else []
    selected_countries = _multiselect_all("Country", country_opts) if "Country" in df.columns else []

    # Apply pre-fetch subset
    prefetch_df = df.copy()
    if selected_sectors and "Sector" in prefetch_df.columns:
        prefetch_df = prefetch_df[prefetch_df["Sector"].isin(selected_sectors)]
    if selected_industry and "Industry" in prefetch_df.columns:
        prefetch_df = prefetch_df[prefetch_df["Industry"].isin(selected_industry)]
    if selected_countries and "Country" in prefetch_df.columns:
        prefetch_df = prefetch_df[prefetch_df["Country"].isin(selected_countries)]

    # Map to YF symbols after narrowing
    prefetch_df = enrich_with_yf_symbols(prefetch_df)

    # ===== Sidebar: post-fetch filters =====
    st.sidebar.header("Post-Fetch Filters")
    exchanges = sorted(prefetch_df["Exchange"].unique().tolist())
    selected_exchange = st.sidebar.selectbox("Exchange", ["All"] + exchanges, index=0)
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50, step=5)

    # --- Fetch & compute ---
    results_df = fetch_all(prefetch_df)
    st.session_state["raw_results_df"] = results_df.copy()

    # --- Post-fetch filtering (includes Sector / Industry / Country again for safety) ---
    filtered = filter_results(
        results_df,
        min_score=min_score,
        exchange=selected_exchange,
        sectors=selected_sectors if selected_sectors else None,
        industries=selected_industry if selected_industry else None,
        countries=selected_countries if selected_countries else None,
    )
    st.session_state["filtered_results"] = filtered

    # --- Display & export ---
    display_results(filtered)

    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download Filtered Results (CSV)",
            data=csv, file_name="momentum_scanner_results.csv", mime="text/csv"
        )

        symbol_options = ["— Select a symbol —"] + filtered["Symbol"].tolist()
        last_selected = st.session_state.get("symbol_select", symbol_options[0])
        if last_selected not in symbol_options:
            last_selected = symbol_options[0]

        selected = st.selectbox(
            "Select a symbol for details",
            options=symbol_options,
            index=symbol_options.index(last_selected),
            key="symbol_select"
        )
        if selected != symbol_options[0]:
            display_symbol_details(filtered, selected)

if __name__ == "__main__":
    main()
