"""
modules.engine — the strategy engine: Trade + output frame, the day loop,
declared-column reads, additional data slots, lookback (previous days), the
prepared-day cache and the RAM cache budget. Qt-free; every strategy here is a
tiny module-like object running on synthetic day files in tmp_path.
"""

import datetime
import json
import math
import types

import numpy as np
import pandas as pd
import pytest

from modules.common.backend.data_roots import DatasetRef, default_dataset_for_slot
from modules.engine import (OUTPUT_COLUMNS, DayCache, EngineError, Trade,
                            additional_slots, run_strategy, strategy_spec,
                            trades_to_frame)

TZ = "America/New_York"
DAYS = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def write_dataset(folder, days=DAYS, columns=("open", "high", "low", "close"),
                  base=100.0):
    """One parquet per day: 3 one-minute bars from 09:30, prices base + day#."""
    folder.mkdir(parents=True, exist_ok=True)
    for n, day in enumerate(days):
        idx = pd.date_range(f"{day} 09:30", periods=3, freq="1min", tz=TZ)
        data = {c: np.array([base + n, base + n + 1, base + n + 2], dtype=float)
                for c in columns}
        pd.DataFrame(data, index=idx).to_parquet(folder / f"{day}.parquet")
    return folder


def strategy(**attrs):
    """A module-like strategy object."""
    mod = types.SimpleNamespace(__name__="toy", PARAMS={"x": 1},
                                DATA={"main": ["close"]})
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def run(mod, folder, start="2026-01-01", end="2026-12-31", params=None,
        cache=None, **kw):
    return run_strategy(mod, folder, start, end, params or {}, tick_size=0.25,
                        cache=cache or DayCache(), verbose=False, **kw)


def ts(day, hhmm):
    return pd.Timestamp(f"{day} {hhmm}", tz=TZ)


# ── Trade + output frame ─────────────────────────────────────────────────────
def test_trade_rejects_bad_direction():
    with pytest.raises(ValueError, match="long' or 'short"):
        Trade("Long", ts(DAYS[0], "10:00"), ts(DAYS[0], "11:00"), 1.0, 2.0, "tp")


def test_trade_rejects_exit_before_entry():
    with pytest.raises(ValueError, match="before"):
        Trade("long", ts(DAYS[0], "11:00"), ts(DAYS[0], "10:00"), 1.0, 2.0, "tp")


def test_pnl_points_sign():
    t0, t1 = ts(DAYS[0], "10:00"), ts(DAYS[0], "11:00")
    assert Trade("long", t0, t1, 100.0, 103.5, "tp").pnl_points == 3.5
    assert Trade("short", t0, t1, 100.0, 103.5, "sl").pnl_points == -3.5


def test_frame_has_the_12_columns_even_when_empty():
    empty = trades_to_frame([])
    assert list(empty.columns) == OUTPUT_COLUMNS and empty.empty
    t = Trade("long", ts(DAYS[0], "10:00"), ts(DAYS[0], "11:00"), 1.0, 2.0, "tp",
              notes={"b": 1, "a": [1, 2], "c": None})
    df = trades_to_frame([(datetime.date(2026, 1, 2), t)])
    assert list(df.columns) == OUTPUT_COLUMNS
    row = df.iloc[0]
    assert row["date"] == datetime.date(2026, 1, 2)
    assert row["pnl_points"] == 1.0
    assert math.isnan(row["sl"]) and math.isnan(row["tp"])
    assert row["trade_type"] is None
    # notes: exactly json.dumps of the dict (key order kept)
    assert row["notes"] == json.dumps({"b": 1, "a": [1, 2], "c": None})


# ── declarations ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("attrs, match", [
    ({"process_day": None}, "process_day"),
    ({"DATA": {"indicators": ["x"]}}, "main"),
    ({"DATA": {"main": []}}, "non-empty list"),
    ({"DATA": {"main": ["a", "a"]}}, "twice"),
    ({"PREPARE_PARAMS": ["x"]}, "no prepare_day"),
])
def test_bad_declarations(attrs, match):
    base = {"process_day": lambda day, p: None}
    base.update(attrs)
    with pytest.raises(EngineError, match=match):
        strategy_spec(strategy(**base))


def test_additional_slots_in_declaration_order():
    mod = strategy(DATA={"main": ["close"], "indicators": "*", "big_trades": ["x"]})
    assert additional_slots(mod) == ["indicators", "big_trades"]


# ── the day loop ─────────────────────────────────────────────────────────────
def test_days_in_order_with_date_filter_and_output(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m")
    (folder / "notes.parquet").write_bytes(b"not a day file")    # ignored
    seen = []

    def process_day(day, params):
        seen.append(day.date)
        close = day.data["main"]["close"]
        return Trade("long", close.index[0], close.index[-1],
                     float(close.iloc[0]), float(close.iloc[-1]), "eod",
                     trade_type="toy", notes={"n": len(close)})

    res = run(strategy(process_day=process_day), folder,
              start="2026-01-05", end="2026-01-07")
    assert seen == [datetime.date(2026, 1, d) for d in (5, 6, 7)]
    assert res.days_in_range == 3 and res.days_run == 3
    df = res.trades
    assert list(df.columns) == OUTPUT_COLUMNS and len(df) == 3
    assert list(df["pnl_points"]) == [2.0, 2.0, 2.0]
    assert json.loads(df.iloc[0]["notes"]) == {"n": 3}
    assert res.warnings == []


def test_params_are_merged_and_tick_size_injected(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    got = []
    mod = strategy(PARAMS={"x": 1, "y": 2, "tick_size": 9.99},
                   process_day=lambda day, p: got.append(dict(p)))
    run(mod, folder, params={"y": 5})
    assert got == [{"x": 1, "y": 5, "tick_size": 0.25}]


def test_process_day_must_return_trades(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    mod = strategy(process_day=lambda day, p: [{"direction": "long"}])
    with pytest.raises(EngineError, match="modules.engine.Trade"):
        run(mod, folder)


def test_undeclared_column_is_explained(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    mod = strategy(process_day=lambda day, p: day.data["main"]["volume"])
    with pytest.raises(EngineError, match="add it to"):
        run(mod, folder)


def test_declared_column_missing_from_file_errors(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    mod = strategy(DATA={"main": ["close", "tick_volume"]},
                   process_day=lambda day, p: None)
    with pytest.raises(EngineError, match=r"\['tick_volume'\].*DATA\['main'\]"):
        run(mod, folder)


def test_all_columns_star(tmp_path):
    folder = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    cols = []
    mod = strategy(DATA={"main": "*"},
                   process_day=lambda day, p: cols.append(list(day.data["main"].columns)))
    run(mod, folder)
    assert cols == [["open", "high", "low", "close"]]


# ── additional data slots ────────────────────────────────────────────────────
def test_additional_slot_is_read_per_day_and_missing_days_are_skipped(tmp_path):
    main = write_dataset(tmp_path / "ES_1m")
    ind = write_dataset(tmp_path / "ES_1m_indicators", columns=("cvd",), base=500.0)
    (ind / "2026-01-06.parquet").unlink()
    seen = {}

    def process_day(day, params):
        seen[day.date.isoformat()] = float(day.data["indicators"]["cvd"].iloc[0])

    mod = strategy(DATA={"main": ["close"], "indicators": ["cvd"]},
                   process_day=process_day)
    res = run(mod, main, extra_folders={"indicators": ind})
    assert "2026-01-06" not in seen and len(seen) == 4
    assert seen["2026-01-05"] == 501.0
    assert res.skipped == {"indicators": ["2026-01-06"]}
    assert "SKIPPED" in res.warnings[0] and "ES_1m_indicators" in res.warnings[0]


def test_additional_slot_needs_a_folder(tmp_path):
    main = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    mod = strategy(DATA={"main": ["close"], "indicators": ["cvd"]},
                   process_day=lambda day, p: None)
    with pytest.raises(EngineError, match="no dataset\\s+was chosen|was chosen"):
        run(mod, main)
    with pytest.raises(EngineError, match="does not exist"):
        run(mod, main, extra_folders={"indicators": tmp_path / "nope"})


def test_unknown_slot_access_errors(tmp_path):
    main = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    mod = strategy(process_day=lambda day, p: day.data["options"])
    with pytest.raises(EngineError, match="No data slot 'options'"):
        run(mod, main)


# ── lookback ─────────────────────────────────────────────────────────────────
def test_previous_days_cross_the_start_date_and_cover_every_slot(tmp_path):
    main = write_dataset(tmp_path / "ES_1m")
    ind = write_dataset(tmp_path / "ES_1m_indicators", columns=("cvd",), base=500.0)
    out = {}

    def process_day(day, params):
        prev = day.previous
        out[day.date.isoformat()] = (
            None if prev is None else prev.date.isoformat(),
            [d.date.isoformat() for d in day.previous_days(3)],
            None if prev is None else float(prev.data["indicators"]["cvd"].iloc[0]),
            None if prev is None else float(prev.data["main"]["close"].iloc[-1]),
        )

    mod = strategy(DATA={"main": ["close"], "indicators": ["cvd"]},
                   process_day=process_day)
    run(mod, main, start="2026-01-06", end="2026-01-08",
        extra_folders={"indicators": ind})
    # 2026-01-06's previous day is 01-05 even though the run starts on 01-06
    assert out["2026-01-06"][0] == "2026-01-05"
    assert out["2026-01-06"][1] == ["2026-01-02", "2026-01-05"]   # only 2 exist before
    assert out["2026-01-08"][1] == ["2026-01-05", "2026-01-06", "2026-01-07"]
    assert out["2026-01-06"][2] == 501.0                          # yesterday's indicators
    assert out["2026-01-06"][3] == 103.0                          # yesterday's last close


def test_missing_slot_on_a_previous_day(tmp_path):
    from modules.engine import EngineError
    main = write_dataset(tmp_path / "ES_1m")
    ind = write_dataset(tmp_path / "ES_1m_indicators", columns=("cvd",))
    (ind / "2026-01-05.parquet").unlink()
    seen = {}

    def process_day(day, params):
        prev = day.previous
        seen[day.date.day] = (day.missing, prev.missing if prev else None)
        if prev is not None and prev.missing:
            with pytest.raises(EngineError, match="does not exist"):
                prev.data["indicators"]

    mod = strategy(DATA={"main": ["close"], "indicators": ["cvd"]},
                   process_day=process_day)
    run(mod, main, extra_folders={"indicators": ind})
    assert seen[6] == ([], ["indicators"])        # yesterday (01-05) lacks indicators
    assert 5 not in seen                          # 01-05 itself was skipped


def test_first_file_has_no_previous(tmp_path):
    main = write_dataset(tmp_path / "ES_1m")
    got = []
    mod = strategy(process_day=lambda day, p: got.append(
        (day.previous, day.previous_days(5), day.previous_days(0))))
    run(mod, main, start="2026-01-02", end="2026-01-02")
    assert got == [(None, [], [])]


# ── prepare_day + caching ────────────────────────────────────────────────────
def test_prepare_day_is_cached_and_keyed_by_prepare_params(tmp_path):
    main = write_dataset(tmp_path / "ES_1m")
    built = []

    def prepare_day(date, data, params):
        built.append((date, dict(params)))
        return {"last": float(data["main"]["close"].iloc[-1]), "anchor": params["anchor"]}

    trades_seen = []
    mod = strategy(PARAMS={"anchor": "a", "x": 1}, PREPARE_PARAMS=["anchor"],
                   prepare_day=prepare_day,
                   process_day=lambda day, p: trades_seen.append(day.prepared))
    cache = DayCache()
    r1 = run(mod, main, cache=cache)
    assert r1.days_prepared == 5 and r1.files_read == 5
    assert built[0] == (datetime.date(2026, 1, 2), {"anchor": "a"})  # only PREPARE_PARAMS
    r2 = run(mod, main, cache=cache, params={"x": 7})        # non-prepare param changed
    assert (r2.days_prepared, r2.files_read) == (0, 0)       # everything from RAM
    r3 = run(mod, main, cache=cache, params={"anchor": "b"})  # prepare param changed
    assert r3.days_prepared == 5 and r3.files_read == 0       # raw frames still cached
    assert trades_seen[-1]["anchor"] == "b"


def test_prepare_returning_none_skips_the_day(tmp_path):
    main = write_dataset(tmp_path / "ES_1m")
    seen = []
    mod = strategy(prepare_day=lambda date, data, p: None if date.day == 5 else date,
                   process_day=lambda day, p: seen.append(day.date.day))
    res = run(mod, main)
    assert seen == [2, 6, 7, 8] and res.days_run == 4


def test_changed_file_is_reread(tmp_path):
    main = write_dataset(tmp_path / "ES_1m", days=DAYS[:1])
    closes = []
    mod = strategy(process_day=lambda day, p: closes.append(
        float(day.data["main"]["close"].iloc[0])))
    cache = DayCache()
    run(mod, main, cache=cache)
    write_dataset(tmp_path / "ES_1m", days=DAYS[:1], base=200.0)   # rewrite the file
    res = run(mod, main, cache=cache)
    assert closes == [100.0, 200.0] and res.files_read == 1


# ── the cache itself ─────────────────────────────────────────────────────────
def test_cache_lru_budget_and_clear():
    cache = DayCache(budget_bytes=300)
    for k in "abc":
        cache.put(("raw", k), k, nbytes=100)
    assert cache.stats()["entries"] == 3
    cache.get(("raw", "a"))                   # refresh 'a' -> 'b' is now the oldest
    cache.put(("raw", "d"), "d", nbytes=100)  # over budget -> evict 'b'
    assert not cache.contains(("raw", "b"))
    assert all(cache.contains(("raw", k)) for k in "acd")
    cache.set_budget(150)                     # shrinking evicts immediately
    assert cache.stats()["bytes"] <= 150
    big = DayCache(budget_bytes=10)
    big.put(("raw", "x"), "x", nbytes=1000)   # one oversize entry stays
    assert big.contains(("raw", "x"))
    assert cache.clear() > 0 and cache.stats()["entries"] == 0


def test_cache_sizes_frames_and_arrays():
    from modules.engine.cache import estimate_nbytes
    arr = np.zeros(1000)
    assert estimate_nbytes(arr) == 8000
    df = pd.DataFrame({"a": arr})
    assert estimate_nbytes(df) >= 8000
    obj = types.SimpleNamespace(x=arr, y=[arr, arr])
    assert estimate_nbytes(obj) >= 24000


# ── additional-data auto-pick ────────────────────────────────────────────────
def _ref(root, dataset):
    return DatasetRef(root, "parquet", "Futures", "ES", dataset, dataset)


def test_default_dataset_for_slot(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    refs = [_ref(r1, "ES_1m_advanced"), _ref(r2, "ES_1m_indicators"),
            _ref(r1, "ES_1m_indicators"), _ref(r1, "ES_big_trades")]
    assert default_dataset_for_slot(refs, "indicators").root == r2        # first match
    assert default_dataset_for_slot(refs, "indicators", prefer_root=r1).root == r1
    assert default_dataset_for_slot(refs, "big_trades").dataset == "ES_big_trades"
    assert default_dataset_for_slot(refs, "BigTrades").dataset == "ES_big_trades"
    assert default_dataset_for_slot(refs, "options") is None


# ── settings ─────────────────────────────────────────────────────────────────
def test_settings_cache_gb_round_trip_and_clamp(tmp_path):
    from modules.common.backend.settings import (DEFAULT_CACHE_GB, Settings,
                                                 load_settings)
    path = tmp_path / "settings.json"
    s = load_settings(path)
    assert s.cache_gb == DEFAULT_CACHE_GB == 4.0
    s.cache_gb = 16.0
    s.save(path)
    assert load_settings(path).cache_gb == 16.0
    assert Settings({}, [], cache_gb=10_000).cache_gb == 512.0
    assert Settings({}, [], cache_gb="junk").cache_gb == DEFAULT_CACHE_GB
