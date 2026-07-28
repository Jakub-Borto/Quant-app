"""Contract tests for the regime-file schema helpers and the pure io layer.
All synthetic — no Qt, no real data."""

import types

import numpy as np
import pandas as pd
import pytest

from modules.regime_detector.backend import io as rio
from modules.regime_detector.backend import schema

NY = "America/New_York"


# ── column parsing (open scope set — documentation, not constraint) ─────────

def test_parse_column_scopes_and_hist():
    assert schema.parse_column("gbx_hist_up_ratio") == ("gbx", True, "up_ratio")
    assert schema.parse_column("rth_hist_daily_vol_center") == \
        ("rth", True, "daily_vol_center")
    assert schema.parse_column("on_vol") == ("on", False, "vol")
    assert schema.parse_column("gbx_ret_sign") == ("gbx", False, "ret_sign")


def test_parse_column_unknown_prefix_is_accepted():
    # never an error — unknown scopes parse and group under "other"
    assert schema.parse_column("foo_bar") == ("foo", False, "bar")
    assert schema.display_group("foo_bar") == "other"
    assert schema.display_group("rth_vol") == "rth"


def test_parse_column_unprefixed_and_always_columns():
    assert schema.parse_column("price") == (None, False, "price")
    assert schema.parse_column("is_final") == (None, False, "is_final")
    assert schema.parse_column("wibble") == (None, False, "wibble")


# ── detector-module validation ───────────────────────────────────────────────

def _good_module():
    return types.SimpleNamespace(
        PARAMS={"rth_start": "09:30", "rth_end": "16:00",
                "snapshot_minutes": 30, "lookback_days": 120},
        REGIME_STATES={"gbx_x": {"states": ["a", "b"],
                                 "colors": ["#111111", "#222222"]}},
        COLUMN_TIERS={"regime": ["gbx_x"], "score": ["gbx_x_score"],
                      "diagnostic": []},
        SCRIPT_VERSION="1.0",
        run_all=lambda *a, **k: None,
    )


def test_validate_detector_clean():
    assert schema.validate_detector(_good_module()) == []


def test_validate_detector_missing_declarations():
    mod = types.SimpleNamespace(PARAMS={})
    errors = schema.validate_detector(mod)
    assert any("REGIME_STATES" in e for e in errors)
    assert any("SCRIPT_VERSION" in e for e in errors)
    assert any("run_all" in e for e in errors)


def test_validate_detector_missing_required_params():
    mod = _good_module()
    del mod.PARAMS["lookback_days"]
    errors = schema.validate_detector(mod)
    assert any("lookback_days" in e for e in errors)


def test_validate_detector_rejects_declared_unknown():
    mod = _good_module()
    mod.REGIME_STATES["gbx_x"]["states"] = ["a", "unknown"]
    mod.REGIME_STATES["gbx_x"]["colors"] = ["#111111", "#222222"]
    errors = schema.validate_detector(mod)
    assert any("unknown" in e for e in errors)


def test_validate_detector_regime_column_must_have_states():
    mod = _good_module()
    mod.COLUMN_TIERS["regime"] = ["gbx_x", "gbx_y"]
    errors = schema.validate_detector(mod)
    assert any("gbx_y" in e for e in errors)


# ── day-frame validation ─────────────────────────────────────────────────────

def _frame(n=3, tz=NY):
    idx = pd.date_range("2026-01-05 18:30", periods=n, freq="30min", tz=tz)
    df = pd.DataFrame({
        "is_final": [False] * (n - 1) + [True],
        "price": np.linspace(100.0, 101.0, n),
        "contract": "ESH6",
        "n_bars_gbx": np.arange(30, 30 * (n + 1), 30),
        "n_bars_rth": 0,
        "lookback_days_used": 120,
        "gbx_x": ["a"] * n,
    }, index=idx)
    return df


def test_validate_day_frame_clean():
    states = {"gbx_x": {"states": ["a", "b"], "colors": ["#1", "#2"]}}
    assert schema.validate_day_frame(_frame(), states) == []


def test_validate_day_frame_is_final_rules():
    df = _frame()
    df["is_final"] = [True, False, True]
    assert any("is_final" in e for e in schema.validate_day_frame(df))
    df["is_final"] = False
    assert any("is_final" in e for e in schema.validate_day_frame(df))


def test_validate_day_frame_needs_ny_tz():
    df = _frame(tz="UTC")
    assert any("America/New_York" in e for e in schema.validate_day_frame(df))
    df2 = _frame()
    df2.index = df2.index.tz_localize(None)
    assert any("tz-aware" in e for e in schema.validate_day_frame(df2))


def test_validate_day_frame_undeclared_state():
    states = {"gbx_x": {"states": ["b"], "colors": ["#2"]}}
    errors = schema.validate_day_frame(_frame(), states)
    assert any("undeclared" in e for e in errors)


def test_validate_day_frame_unknown_is_always_allowed():
    df = _frame()
    df.loc[df.index[0], "gbx_x"] = schema.UNKNOWN_STATE
    states = {"gbx_x": {"states": ["a", "b"], "colors": ["#1", "#2"]}}
    assert schema.validate_day_frame(df, states) == []


# ── io helpers ───────────────────────────────────────────────────────────────

def test_propose_run_name_strips_scaffold_underscore():
    assert rio.propose_run_name("ES", "_scaffold_example") == "ES_scaffold_example"
    assert rio.propose_run_name("NQ", "vol_regimes") == "NQ_vol_regimes"


def test_meta_round_trip(tmp_path):
    meta = {
        "script_name": "x", "script_version": "1.0",
        "params": {"lookback_days": np.int64(120), "thr": np.float64(0.5),
                   "flag": np.bool_(True)},
        "run_started": pd.Timestamp("2026-07-24 10:00"),
        "counts": {"processed": 3, "skipped": 1, "missing_input": 0,
                   "short_lookback": 2},
        "states": {"gbx_x": {"states": ["a"], "colors": ["#1"]}},
        "schema": {"regime": ["gbx_x"], "score": [], "diagnostic": []},
        "meta_version": 1,
    }
    rio.write_meta(tmp_path, meta)
    back = rio.read_meta(tmp_path)
    assert back["params"] == {"lookback_days": 120, "thr": 0.5, "flag": True}
    assert back["run_started"].startswith("2026-07-24T10:00")
    assert back["counts"]["short_lookback"] == 2
    assert back["states"]["gbx_x"]["colors"] == ["#1"]
    assert back["schema"]["regime"] == ["gbx_x"]


def test_day_files_excludes_non_dated(tmp_path):
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "2026-01-05.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "2026-01-06.parquet")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "notes.parquet")
    assert list(rio.day_files(tmp_path)) == ["2026-01-05", "2026-01-06"]


def test_snapshot_grid_labels_globex():
    labels = rio.snapshot_grid_labels("18:00", "17:00", 30)
    assert labels[0] == "18:30"
    assert labels[-1] == "17:00"
    assert "09:30" in labels and "00:00" in labels
    assert len(labels) == 46          # 23h of 30-min steps

    hourly = rio.snapshot_grid_labels("18:00", "17:00", 60)
    assert hourly[0] == "19:00" and hourly[-1] == "17:00"


def test_pick_rows_asof_and_final():
    idx1 = pd.DatetimeIndex([
        pd.Timestamp("2026-01-04 18:30", tz=NY),   # evening BEFORE RTH date
        pd.Timestamp("2026-01-05 10:00", tz=NY),
        pd.Timestamp("2026-01-05 17:00", tz=NY),
    ])
    df1 = pd.DataFrame({"price": [1.0, 2.0, 3.0],
                        "gbx_x": ["a", "b", "a"],
                        "is_final": [False, False, True]}, index=idx1)
    frames = {"2026-01-05": df1}

    final = rio.pick_rows(frames, rio.FINAL, session_start="18:00")
    assert final.loc[pd.Timestamp("2026-01-05"), "price"] == 3.0

    ten = rio.pick_rows(frames, "10:00", session_start="18:00")
    assert ten.loc[pd.Timestamp("2026-01-05"), "price"] == 2.0

    # evening wall time resolves to the EVENING BEFORE the RTH date
    evening = rio.pick_rows(frames, "18:30", session_start="18:00")
    assert evening.loc[pd.Timestamp("2026-01-05"), "price"] == 1.0

    # a time before any snapshot exists drops the day
    early = rio.pick_rows(frames, "18:15", session_start="18:00")
    assert early.empty


def test_pick_rows_half_day_takes_last_available():
    # half day: file ends at 13:00 — an as-of of 15:30 sees the 13:00 row
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-03 10:00", tz=NY),
                            pd.Timestamp("2026-07-03 13:00", tz=NY)])
    df = pd.DataFrame({"price": [1.0, 2.0], "is_final": [False, True]},
                      index=idx)
    picked = rio.pick_rows({"2026-07-03": df}, "15:30", session_start="18:00")
    assert picked.loc[pd.Timestamp("2026-07-03"), "price"] == 2.0
