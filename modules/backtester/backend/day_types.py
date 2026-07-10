"""
Day-type tagging for the Backtester module.

Extracted verbatim from legacy_streamlit/views/backtester.py. The
classification logic itself lives in modules/optimizer/backend/buckets.py
(shared with the Strategy Optimizer — same priority rules, same
EVENT_KEYWORDS config), so both modules classify a given date identically.
This module keeps only the backtester's historical names: a `day_type`
column, and 'high_impact' instead of the shared 'other_high_impact'.

Change vs the old view: the FF events parquet path is now passed in
explicitly (it comes from the settings' data roots) instead of the old
hardcoded data/news_and_holidays location.
"""

import pandas as pd

from modules.optimizer.backend.buckets import load_bucket_map


def load_day_classifications(ff_events_path) -> dict[str, str]:
    """
    {date_iso: day_type} from the FF events parquet ({} when the path is
    None or the file is missing — every date then resolves to 'normal' in
    tag_trades()).
    """
    bucket_map = load_bucket_map(ff_events_path) if ff_events_path else {}
    return {
        date: ("high_impact" if bucket == "other_high_impact" else bucket)
        for date, bucket in bucket_map.items()
    }


def tag_trades(trades: pd.DataFrame, day_classifications: dict) -> pd.DataFrame:
    """Adds a single 'day_type' column ('normal' for unlisted dates)."""
    trades = trades.copy()
    trades["day_type"] = trades["date"].apply(
        lambda d: day_classifications.get(pd.Timestamp(d).date().isoformat(), "normal")
    )
    return trades
