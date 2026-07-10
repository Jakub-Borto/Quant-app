"""
Trades-file IO for the Analytics module.

Extracted verbatim from legacy_streamlit/views/analytics.py. Reads the parquet
files the Backtester saves, normalizes datetime columns, and decodes the
filter kv-metadata the Backtester stamps on save.

Change vs the old view: functions take a full Path instead of a filename
resolved against the old hardcoded data/trades (files can now live in any
configured data root's trades/ folder).
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure entry_time / exit_time / date columns are typed correctly.

    Notes on the dtype check:
    - We use pd.api.types.is_datetime64_any_dtype because it correctly handles
      timezone-aware dtypes (e.g. datetime64[ns, America/New_York]). NumPy's
      np.issubdtype raises on tz-aware dtypes, which is how an earlier version
      of this code crashed on real data.
    - For the 'date' column, we only re-parse when it's stored as strings
      (object dtype). If pandas already gave us a python date object column
      (typical for `.dt.date` results), we leave it alone.
    """
    for col in ("entry_time", "exit_time"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        if df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def load_trades(path: Path) -> pd.DataFrame:
    """Read a trades parquet file and normalize its datetime columns."""
    df = pd.read_parquet(path)
    return _coerce_datetime_columns(df)


def read_filter_metadata(path: Path) -> dict | None:
    """
    Read the filter kv-metadata the backtester stamps onto saved trades.
    Returns None for unfiltered or legacy (pre-metadata) files; else
    {"day_types": [...keys], "trade_types": "all" | [...values]}.
    """
    schema_meta = pq.read_schema(path).metadata or {}
    if schema_meta.get(b"filtered", b"false").decode() != "true":
        return None

    day_types = json.loads(schema_meta.get(b"selected_day_types", b"[]").decode())
    tt_raw    = schema_meta.get(b"selected_trade_types", b"all").decode()
    trade_types = "all" if tt_raw == "all" else json.loads(tt_raw)
    return {"day_types": day_types, "trade_types": trade_types}
