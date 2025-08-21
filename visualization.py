# visualization.py
from __future__ import annotations

import io
from typing import List

import streamlit as st
import pandas as pd

from data_loader import (
    read_uploaded_sheet, clean_symbols, enrich_with_yf_symbols, fetch_all
)
from analysis import filter_results


APP_TITLE = "Momentum Scanner (Modularized)"
REQUIRED_COLUMNS = {"Symbol", "Exchange"}   # Other metadata columns are optional

# -----------------------
# Small UI helpers
# -----------------------
def _multiselect_all(label: str, options: List[str]) -> List[str]:
    if not options:
        return []
    default = options  # select all by default
    return st.multiselect(label, options=options, default=default)


def _render_score_breakdown(row: pd.Series):
    metrics = pd.DataFrame({
        "Indicator": [
            "Price", "EMA20", "EMA50", "EMA200",
            "RSI(14)", "MACD Hist", "ADX(14)", "Vol / 20d avg",
            "+DI", "-DI",
        ],
        "Value": [
            row.get("Price"), row.get("EMA20"), row.get("EMA50"), row.get("EMA200"),
            row.get("RSI"), row.get("MACD_Hist"), row.get("ADX"), row.get("Volume_Ratio"),
            row.get("plus_di_last"), row.get("minus_di_last"),
        ],
        "Points": [
            None, row.get("EMA_Points"), None, None,
            row.get("RSI_Points"), row.get("MACD_Points"), row.get("ADX_Points"), None,
            row.get("DI_Points"), None,
        ]
    })
    st.dataframe(metrics, hide_index=True, use_container_width=True)


def _download_csv_button(df: pd.DataFrame, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name="momentum_results.csv", mime="text/csv")


def _download_xlsx_button(df: pd.DataFrame, label: str = "Download Excel"):
    import pandas as pd
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    st.download_button(label, data=output.getvalue(), file_name="momentum_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _apply_prefetch_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filters applied before fetching (metadata-only)."""
    out = df.copy()
    with st.expander("Prefetch filters (apply before downloading market data)"):
        # Exchange
        ex_options = ["All"] + sorted(x for x in out["Exchange"].dropna().unique() if x != "")
        ex_selected = st.selectbox("Exchange", options=ex_options, index=0)
        if ex_selected != "All":
            out = out[out["Exchange"] == ex_selected]

        # Metadata filters, if present
        for col in ["Sector", "Industry", "Country", "Theme", "Asset_Type"]:
            if col in out.columns:
                opts = sorted(out[col].dropna().astype(str).unique())
                if opts:
                    selected = st.multiselect(col, options=opts, default=opts)
                    out = out[out[col].isin(selected)]
    return out


def display_symbol_details(filtered_df: pd.DataFrame, selected_symbol: str):
    try:
        row = filtered_df[filtered_df["Symbol"] == selected_symbol].iloc[0]
        st.subheader(f"{selected_symbol} — Detailed Snapshot")

        # 1) Quick breakdown table
        _render_score_breakdown(row)

        # 2) Full JSON (raw fields)
        st.json(row.to_dict())
    except Exception as e:
        st.error(f"Error loading {selected_symbol}: {str(e)}")


# -----------------------
# Main
# -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        /* try to enforce dark-ish vibes regardless of theme */
        .stApp { background-color: #0e1117; color: #e5e7eb; }
        .stMarkdown, .stDataFrame, .stSelectbox, .stMultiSelect, .stButton, .stDownloadButton, .stText { color: #e5e7eb; }
        div[data-testid="stTable"], div[data-testid="stDataFrame"] { background-color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.write("Upload a sheet with **Symbol** and **Exchange**. Optional columns like Sector/Industry/Country are supported.")

    # 1) Upload + clean
    raw = read_uploaded_sheet()
    if raw is None:
        st.info("Waiting for upload…")
        return

    base_df = clean_symbols(raw)
    if base_df.empty:
        st.warning("No valid symbols found.")
        return

    # 2) Optional: filter BEFORE fetching (metadata-only)
    prefetch_df = _apply_prefetch_filters(base_df)

    # 3) Map YF symbols
    with st.status("Mapping Yahoo tickers…", state="running"):
        prefetch_df = enrich_with_yf_symbols(prefetch_df)

    st.write(f"Tickers to fetch: **{len(prefetch_df)}**")
    if len(prefetch_df) == 0:
        st.stop()

    # 4) Fetch button to avoid unnecessary API calls
    if st.button("Fetch data", type="primary"):
        results_df = fetch_all(prefetch_df)

        if results_df.empty:
            st.warning("No symbols returned data (insufficient history, invalid suffixes, or API throttling). Try fewer tickers or different exchanges.")
            st.stop()

        # 5) Post-fetch filtering & display
        with st.expander("Post-fetch filters (score & metadata)"):
            min_score = st.slider("Minimum Momentum Score", 0, 100, 60, 5)
            ex_post = ["All"] + sorted(results_df["Exchange"].dropna().unique().tolist())
            ex_selected = st.selectbox("Exchange (post)", options=ex_post, index=0)
            sectors = _multiselect_all("Sector", sorted(results_df["Sector"].dropna().unique())) if "Sector" in results_df.columns else []
            industries = _multiselect_all("Industry", sorted(results_df["Industry"].dropna().unique())) if "Industry" in results_df.columns else []
            countries = _multiselect_all("Country", sorted(results_df["Country"].dropna().unique())) if "Country" in results_df.columns else []

        filtered = filter_results(
            results_df,
            min_score=min_score,
            exchange=ex_selected,
            sectors=sectors,
            industries=industries,
            countries=countries,
        )

        st.subheader("Results")
        st.dataframe(filtered, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1: _download_csv_button(filtered, "Download filtered CSV")
        with col2: _download_xlsx_button(filtered, "Download filtered Excel")

        # 6) Symbol details
        symbol_options = ["(choose)"] + filtered["Symbol"].astype(str).tolist()
        last_selected = st.session_state.get("last_symbol", symbol_options[0])
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
