# app.py
# Streamlit app: tolerant XLSX uploader (blank cells & missing columns allowed)

from __future__ import annotations
import io
import pandas as pd
import streamlit as st

from data_loader import validate_expected_columns, clean_symbols, EXPECTED_COLS

st.set_page_config(page_title="Momentum Sheet Loader", layout="wide")

# ---- UI: Header / Cache
st.title("📈 Momentum – Sheet Uploader (tolerant)")

left, right = st.columns([1, 5])
with left:
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

st.caption(
    "Upload an **.xlsx** file. Missing columns will be auto-created as empty. "
    "Only **Symbol** is required; all other fields can be blank."
)

# ---- File uploader
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

@st.cache_data(show_spinner=False)
def read_xlsx_return_sheets(file_bytes: bytes):
    x = pd.ExcelFile(io.BytesIO(file_bytes))
    return x.sheet_names

@st.cache_data(show_spinner=False)
def read_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)

if uploaded is None:
    st.info("Waiting for an .xlsx file…")
    st.stop()

# List sheets to choose from
try:
    sheet_names = read_xlsx_return_sheets(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not open Excel file: {e}")
    st.stop()

sheet = st.selectbox("Select sheet to analyze", options=sheet_names, index=0)

# ---- Read selected sheet
try:
    df_raw = read_sheet(uploaded.getvalue(), sheet)
except Exception as e:
    st.error(f"Failed to read sheet: {e}")
    st.stop()

st.markdown("### Loaded preview")
st.dataframe(df_raw.head(20), use_container_width=True)

# ---- Tolerant schema handling
try:
    df = validate_expected_columns(df_raw)
    df, dropped = clean_symbols(df)
except Exception as e:
    st.error(f"Validation failed: {e}")
    st.stop()

# ---- Report what we did
missing_cols = [c for c in EXPECTED_COLS if c not in df_raw.columns]
if missing_cols:
    st.warning(
        "Missing columns were auto-created as empty: "
        + ", ".join(missing_cols)
    )

if dropped > 0:
    st.info(f"Dropped **{dropped}** rows with empty/invalid `Symbol`.")

st.success("Sheet accepted. You can proceed with blanks or empty columns.")

# ---- Show the normalized, ready-to-use table
st.markdown("### Normalized data (ready for downstream steps)")
st.dataframe(df, use_container_width=True)

# ---- Download buttons (XLSX via openpyxl; fallback to CSV)
@st.cache_data(show_spinner=False)
def to_excel_bytes_openpyxl(frame: pd.DataFrame, sheet_name: str) -> bytes:
    bio = io.BytesIO()
    # Prefer openpyxl for Streamlit Cloud
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.read()

@st.cache_data(show_spinner=False)
def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")

st.markdown("#### Download")
xlsx_ok = True
xlsx_bytes = b""
try:
    xlsx_bytes = to_excel_bytes_openpyxl(df, sheet)
except Exception as e:
    xlsx_ok = False
    st.warning(
        "Could not create XLSX (missing `openpyxl` or another writer). "
        "You can still download CSV below."
    )

cols = st.columns(2)
with cols[0]:
    if xlsx_ok:
        st.download_button(
            "⬇️ Download normalized sheet (.xlsx)",
            data=xlsx_bytes,
            file_name=f"normalized_{sheet}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
with cols[1]:
    st.download_button(
        "⬇️ Download normalized data (.csv)",
        data=to_csv_bytes(df),
        file_name=f"normalized_{sheet}.csv",
        mime="text/csv",
    )

st.caption(
    "Tip: Keep extra columns if you like (e.g., `Exchange`). "
    "They will be preserved alongside the expected schema."
)
