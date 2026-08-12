"""
Single-trade chart windowing, shared by the trade-detail widgets.

Extracted verbatim from legacy_streamlit/views/trade_report.py:
resolve_chart_window slices a session's candles around a trade according to
the chart-view settings dict, and _is_timestamp backs the trade-notes
formatting.
"""

import pandas as pd


def resolve_chart_window(session: pd.DataFrame, entry_ts: pd.Timestamp,
                         exit_ts: pd.Timestamp, chart_settings: dict) -> pd.DataFrame:
    exit_loc = session.index.searchsorted(exit_ts, side="right") - 1
    exit_loc = max(0, min(exit_loc, len(session) - 1))
    end_loc  = min(exit_loc + chart_settings["candles_after"], len(session) - 1)

    if chart_settings["view_mode"] == "Candles before entry":
        entry_loc = session.index.searchsorted(entry_ts, side="left")
        entry_loc = max(0, min(entry_loc, len(session) - 1))
        start_loc = max(0, entry_loc - chart_settings["candles_before"])
    else:
        time_mask = session.index.time >= chart_settings["session_start_time"]
        start_loc = int(time_mask.argmax()) if time_mask.any() else 0

    return session.iloc[start_loc: end_loc + 1]


def _is_timestamp(val) -> bool:
    try:
        pd.Timestamp(val)
        return isinstance(val, str) and (":" in val or "-" in val)
    except Exception:
        return False
