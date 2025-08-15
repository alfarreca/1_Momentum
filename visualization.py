import streamlit as st
import pandas as pd


def display_results(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        st.warning("No stocks match your current filters.")
        return
    st.metric("Stocks Found", len(filtered_df))
    if "Momentum_Score" in filtered_df.columns and not filtered_df["Momentum_Score"].empty:
        st.metric("Avg Momentum Score", round(filtered_df["Momentum_Score"].mean(), 1))
    st.dataframe(
        filtered_df.sort_values("Momentum_Score", ascending=False),
        use_container_width=True,
        height=600,
    )


def display_symbol_details(selected_symbol: str, filtered_df: pd.DataFrame) -> None:
    if not selected_symbol:
        return
    try:
        symbol_data = filtered_df[filtered_df["Symbol"] == selected_symbol].iloc[0]
        st.subheader(f"{selected_symbol} — Detailed Analysis")
        st.json(symbol_data.to_dict())
    except Exception as e:
        st.error(f"Error loading {selected_symbol}: {str(e)}")


def download_button_for_df(df: pd.DataFrame, filename: str = "momentum_scanner_results.csv") -> None:
    if df.empty:
        return
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Filtered Results as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
