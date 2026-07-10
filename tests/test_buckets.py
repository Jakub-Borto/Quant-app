"""Day-bucket mapping: priorities, keywords, missing file (spec §6.4 / §14)."""

import pandas as pd
import pytest

from modules.optimizer.backend.buckets import (
    BUCKET_KEYS, build_bucket_map, bucket_for_date, classify_red_event,
    load_bucket_map, tag_day_bucket,
)


def events(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "event", "impact"])


@pytest.mark.parametrize("event,expected", [
    ("FOMC Statement",                "fomc"),
    ("FOMC Press Conference",         "fomc"),
    ("CPI m/m",                       "cpi"),
    ("Core CPI m/m",                  "cpi"),
    ("Non-Farm Employment Change",    "nfp"),
    ("PPI m/m",                       "ppi"),
    ("Core PPI m/m",                  "ppi"),
    ("Retail Sales m/m",              "other_high_impact"),   # unmatched red
    ("Flash Manufacturing PMI",       "other_high_impact"),   # PMI is not PPI
])
def test_classify_red_event(event, expected):
    assert classify_red_event(event) == expected


def test_holiday_beats_red():
    bm = build_bucket_map(events([
        ("2026-01-05", "Bank Holiday", "grey"),
        ("2026-01-05", "CPI m/m",      "red"),
    ]))
    assert bm["2026-01-05"] == "holiday"


def test_matched_red_beats_unmatched():
    bm = build_bucket_map(events([
        ("2026-01-06", "Retail Sales m/m", "red"),
        ("2026-01-06", "CPI m/m",          "red"),
    ]))
    assert bm["2026-01-06"] == "cpi"


def test_cross_bucket_priority_order():
    # fomc outranks cpi (BUCKET_ORDER)
    bm = build_bucket_map(events([
        ("2026-01-07", "CPI m/m",        "red"),
        ("2026-01-07", "FOMC Statement", "red"),
    ]))
    assert bm["2026-01-07"] == "fomc"


def test_unmatched_red_only():
    bm = build_bucket_map(events([("2026-01-08", "Housing Starts", "red")]))
    assert bm["2026-01-08"] == "other_high_impact"


def test_no_events_is_normal():
    bm = build_bucket_map(events([("2026-01-05", "CPI m/m", "red")]))
    assert bucket_for_date("2026-02-02", bm) == "normal"
    assert bucket_for_date(pd.Timestamp("2026-01-05"), bm) == "cpi"


def test_missing_file_empty_map(tmp_path):
    assert load_bucket_map(tmp_path / "nope.parquet") == {}


def test_custom_keywords():
    kw = {"cpi": ["Inflation Print"]}
    assert classify_red_event("Inflation Print y/y", kw) == "cpi"
    assert classify_red_event("CPI m/m", kw) == "other_high_impact"


def test_tag_day_bucket():
    bm = build_bucket_map(events([
        ("2026-01-05", "CPI m/m",      "red"),
        ("2026-01-06", "Bank Holiday", "grey"),
    ]))
    trades = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "pnl_ticks": [1.0, 2.0, 3.0],
    })
    tagged = tag_day_bucket(trades, bm)
    assert list(tagged["day_bucket"]) == ["cpi", "holiday", "normal"]
    assert "day_bucket" not in trades.columns          # input untouched
    assert all(b in BUCKET_KEYS for b in tagged["day_bucket"])
