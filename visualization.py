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

# -----------------------
# Small UI helpers
# -----------------------
def _multiselect_all(label: str, options: List[str], key: str | None = None) -> List[str]:
    if not options:
        return []
    default = options  # select all by default
    return st.multiselect(label, options=options, default=default, key=key)

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
    """
    Robust Excel export:
    - Try XlsxWriter
    - Fallback to openpyxl
    - If neither is present, fall back to CSV
    """
    output = io.BytesIO()

    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None

    if engine:
        with pd.ExcelWriter(output, engine=engine) as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        st.download_button(
            label,
            data=output.getvalue(),
            file_name="momentum_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Excel engine not available. Downloading CSV instead.")
        _download_csv_button(df, "Download CSV (fallback)")

def _apply_prefetch_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filters applied before fetching (metadata-only)."""
    out = df.copy()
    with st.expander("Prefetch filters (apply before downloading market data)"):
        ex_options = ["All"] + sorted(x for x in out["Exchange"].dropna().unique() if x != "")
        ex_selected = st.selectbox("Exchange", options=ex_options, index=0, key="pre_exchange")
        if ex_selected != "All":
            out = out[out["Exchange"] == ex_selected]

        for col in ["Sector", "Industry", "Country", "Theme", "Asset_Type"]:
            if col in out.columns:
                opts = sorted(out[col].dropna().astype(str).unique())
                if opts:
                    selected = st.multiselect(col, options=opts, default=opts, key=f"pre_{col.lower()}")
                    out = out[out[col].isin(selected)]
    return out

def display_symbol_details(filtered_df: pd.DataFrame, selected_symbol: str):
    try:
        row = filtered_df[filtered_df["Symbol"] == selected_symbol].iloc[0]
        st.subheader(f"{selected_symbol} — Detailed Snapshot")
        _render_score_breakdown(row)      # quick table with points breakdown
        st.json(row.to_dict())            # full raw fields
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
        /* dark theme-ish */
        .stApp { background-color: #0e1117; color: #e5e7eb; }
        div[data-testid="stTable"], div[data-testid="stDataFrame"] { background-color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.write("Upload a sheet with **Symbol** and **Exchange**. Optional columns like Sector/Industry/Country are supported.")

    # --- state init ---
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "min_score" not in st.session_state:
        st.session_state["min_score"] = 60
    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = None

    # 1) Upload + clean
    raw = read_uploaded_sheet()
    if raw is None:
        st.info("Waiting for upload…")
        return

    base_df = clean_symbols(raw)
    if base_df.empty:
        st.warning("No valid symbols found.")
        return

    # 2) Prefetch (metadata-only) filters
    prefetch_df = _apply_prefetch_filters(base_df)

    # 3) Map YF symbols
    with st.status("Mapping Yahoo tickers…", state="running"):
        prefetch_df = enrich_with_yf_symbols(prefetch_df)

    st.write(f"Tickers to fetch: **{len(prefetch_df)}**")
    if len(prefetch_df) == 0:
        st.stop()

    # 4) Fetch button (persist results in session_state to survive reruns)
    if st.button("Fetch data", type="primary", key="btn_fetch"):
        results_df = fetch_all(prefetch_df)
        st.session_state["results_df"] = results_df if not results_df.empty else None
        # auto-select top symbol if any
        if st.session_state["results_df"] is not None and "Symbol" in st.session_state["results_df"].columns:
            first_symbol = str(st.session_state["results_df"]["Symbol"].iloc[0])
            st.session_state["selected_symbol"] = first_symbol
        st.rerun()

    results_df = st.session_state["results_df"]
    if results_df is None:
        st.info("Click **Fetch data** to load market data.")
        return

    # 5) Post-fetch filters & display
    with st.expander("Post-fetch filters (score & metadata)"):
        st.session_state["min_score"] = st.slider(
            "Minimum Momentum Score",
            0, 100, st.session_state["min_score"], 5, key="post_min_score"
        )
        ex_post = ["All"] + sorted(results_df["Exchange"].dropna().unique().tolist())
        ex_selected = st.selectbox("Exchange (post)", options=ex_post, index=0, key="post_exchange")

        sectors = []
        industries = []
        countries = []
        if "Sector" in results_df.columns:
            sectors = _multiselect_all("Sector", sorted(results_df["Sector"].dropna().unique()), key="post_sector")
        if "Industry" in results_df.columns:
            industries = _multiselect_all("Industry", sorted(results_df["Industry"].dropna().unique()), key="post_industry")
        if "Country" in results_df.columns:
            countries = _multiselect_all("Country", sorted(results_df["Country"].dropna().unique()), key="post_country")

    filtered = filter_results(
        results_df,
        min_score=st.session_state["min_score"],
        exchange=ex_selected,
        sectors=sectors,
        industries=industries,
        countries=countries,
    )

    st.caption(f"Rows: fetched **{len(results_df)}**, after filters **{len(filtered)}**")
    if filtered.empty:
        st.warning("No rows after filters. Lower the score threshold or clear metadata filters.")
        # Keep the dropdown hidden when empty and stop here
        return

    st.subheader("Results")
    st.dataframe(filtered, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1: _download_csv_button(filtered, "Download filtered CSV")
    with col2: _download_xlsx_button(filtered, "Download filtered Excel")

    # 6) Symbol details — auto-select first row if none selected
    options = filtered["Symbol"].astype(str).tolist()
    if options:
        if st.session_state["selected_symbol"] not in options:
            st.session_state["selected_symbol"] = options[0]  # auto-pick first

        selected = st.selectbox(
            "Select a symbol for details",
            options=["(choose)"] + options,
            index=options.index(st.session_state["selected_symbol"]) + 1 if st.session_state["selected_symbol"] in options else 0,
            key="post_symbol_select",
        )

        if selected != "(choose)":
            st.session_state["selected_symbol"] = selected
            display_symbol_details(filtered, selected)

if __name__ == "__main__":
    main()
