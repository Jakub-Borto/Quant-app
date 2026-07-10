"""
Trading-date -> day_bucket mapping from the Forex Factory USD calendar.

Single source of truth for news/holiday day classification. The optimizer
tags every trade with `day_bucket`; views/backtester.py delegates its
`day_type` tagging here as well (it only renames `other_high_impact` to its
historical `high_impact`), so both views classify a given date identically.

A calendar date maps to exactly ONE bucket, resolved by priority:

  1. any grey-impact event that date            -> holiday
  2. else any red event matching EVENT_KEYWORDS -> first bucket hit, in
     EVENT_KEYWORDS order (fomc, cpi, nfp, ppi)
  3. else any unmatched red event               -> other_high_impact
  4. else                                       -> normal

Whole-day bucketing (rather than per-trade) keeps day-level filters clean:
excluding a bucket removes entire trading days, so daily-Sharpe stays
well-defined on the remainder.

Limitation: the calendar is USD-only, so bucketing of non-US-centric assets
(e.g. the 6E/6J/6B/6C FX pairs) ignores their foreign-calendar events — same
limitation as the backtester.
"""

from pathlib import Path

import pandas as pd

FF_EVENTS_PATH = Path("data/news_and_holidays/ff_usd_events.parquet")

# Buckets in priority order (first match wins). Labels drive the UI checkboxes.
BUCKET_ORDER = [
    ("holiday",           "Holidays"),
    ("fomc",              "FOMC"),
    ("cpi",               "CPI"),
    ("nfp",               "Non-Farm Employment"),
    ("ppi",               "PPI"),
    ("other_high_impact", "Other High Impact News"),
    ("normal",            "Normal Trading Days"),
]
BUCKET_KEYS = [key for key, _ in BUCKET_ORDER]

# bucket -> case-insensitive substrings matched against RED-impact event names.
# Order matters twice: across buckets (a date with both CPI and FOMC events is
# an fomc day) and within a list (first substring hit classifies the event).
#
# The defaults reproduce the backtester's original RED_EVENT_PATTERNS exactly,
# verified against the real ff_usd_events.parquet strings: "FOMC" covers
# FOMC Statement/Press Conference/Minutes/Member speeches (the Federal Funds
# Rate release co-occurs with FOMC Statement, so day-level it is already fomc);
# "CPI"/"PPI" also catch the Core m/m variants. Caveat: "Non-Farm Employment
# Change" also matches the ADP release, so ADP Wednesdays land in the nfp
# bucket — long-standing backtester behaviour, kept for consistency.
EVENT_KEYWORDS = {
    "fomc": ["FOMC"],
    "cpi":  ["CPI"],
    "nfp":  ["Non-Farm Employment Change"],
    "ppi":  ["PPI"],
}


def classify_red_event(event: str, keywords: dict | None = None) -> str:
    """Bucket for one red-impact event name (first keyword hit wins)."""
    keywords = EVENT_KEYWORDS if keywords is None else keywords
    lowered = str(event).lower()
    for bucket, patterns in keywords.items():
        for pattern in patterns:
            if pattern.lower() in lowered:
                return bucket
    return "other_high_impact"


def build_bucket_map(events: pd.DataFrame, keywords: dict | None = None) -> dict:
    """
    {date_iso: bucket} from an events table with columns date/event/impact.
    Dates absent from the result are 'normal' (see bucket_for_date).
    """
    keywords = EVENT_KEYWORDS if keywords is None else keywords
    if events.empty:
        return {}

    events = events.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date

    priority = {key: i for i, key in enumerate(BUCKET_KEYS)}
    bucket_map: dict[str, str] = {}
    for date, group in events.groupby("date"):
        if (group["impact"] == "grey").any():
            bucket_map[date.isoformat()] = "holiday"
            continue
        red_events = group.loc[group["impact"] == "red", "event"]
        if red_events.empty:
            continue  # normal — not stored
        buckets = {classify_red_event(e, keywords) for e in red_events}
        bucket_map[date.isoformat()] = min(buckets, key=priority.__getitem__)
    return bucket_map


def load_bucket_map(path: Path = FF_EVENTS_PATH,
                    keywords: dict | None = None) -> dict:
    """
    Bucket map from the FF events parquet. Returns {} when the file is missing
    — every date then resolves to 'normal'; callers should surface a warning.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return build_bucket_map(pd.read_parquet(path), keywords)


def bucket_for_date(date, bucket_map: dict) -> str:
    """Bucket of a single date-like value ('normal' when unlisted)."""
    return bucket_map.get(pd.Timestamp(date).date().isoformat(), "normal")


def tag_day_bucket(trades: pd.DataFrame, bucket_map: dict) -> pd.DataFrame:
    """Copy of `trades` with a 'day_bucket' column derived from 'date'."""
    trades = trades.copy()
    keys = pd.to_datetime(trades["date"]).dt.date.astype(str)
    trades["day_bucket"] = keys.map(lambda k: bucket_map.get(k, "normal"))
    return trades
