"""
Backend tests for the backtester's trades persistence — the temp-file
handoff writer used by the "Go to Analytics" / "Go to Monte Carlo" buttons.
Pure backend: no Qt.
"""

import pandas as pd
import pyarrow.parquet as pq

from modules.analytics.backend.io import read_filter_metadata
from modules.backtester.backend.persistence import save_temp_trades
from modules.common.backend.data_roots import clear_temp_files, list_trades_files


def _trades() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-01-05", "2026-01-06"],
        "direction": ["long", "short"],
        "pnl_points": [2.5, -1.0],
        "ticks": [10.0, -4.0],
    })


def test_save_temp_trades_increments_n(tmp_path):
    temp = tmp_path / "root" / "temp"
    other = _trades().assign(ticks=[1.0, 2.0])
    p1 = save_temp_trades(temp, _trades(), "ES", False, ["normal"], "all")
    p2 = save_temp_trades(temp, other, "ES", False, ["normal"], "all")
    assert p1.name == "ES_temp_file_1.parquet"
    assert p2.name == "ES_temp_file_2.parquet"
    assert p1.exists() and p2.exists()


def test_save_temp_trades_reuses_identical_file(tmp_path):
    temp = tmp_path / "root" / "temp"
    p1 = save_temp_trades(temp, _trades(), "ES", True, ["normal"], "all")
    p2 = save_temp_trades(temp, _trades(), "ES", True, ["normal"], "all")
    assert p2 == p1                       # same rows + same filter meta -> reuse
    assert len(list(temp.glob("*.parquet"))) == 1
    # same rows but DIFFERENT filter metadata -> a new file
    p3 = save_temp_trades(temp, _trades(), "ES", True, ["cpi"], "all")
    assert p3.name == "ES_temp_file_2.parquet"


def test_save_temp_trades_creates_dir_and_roundtrips(tmp_path):
    temp = tmp_path / "root" / "temp"
    assert not temp.exists()
    trades = _trades()
    path = save_temp_trades(temp, trades, "NQ", False, ["normal"], "all")
    assert path.parent == temp
    assert pd.read_parquet(path).equals(trades)


def test_save_temp_trades_stamps_filter_metadata(tmp_path):
    path = save_temp_trades(tmp_path / "temp", _trades(), "ES",
                            True, ["normal", "cpi"], ["A"])
    meta = pq.read_schema(path).metadata
    assert meta[b"filtered"] == b"true"
    # Analytics reads this back to show the "filtered file" caption
    assert read_filter_metadata(path) == {"day_types": ["normal", "cpi"],
                                          "trade_types": ["A"]}


def test_temp_files_invisible_to_trades_pickers(tmp_path):
    root = tmp_path / "root"
    save_temp_trades(root / "temp", _trades(), "ES", False, ["normal"], "all")
    assert list_trades_files([root]) == []


def test_clear_temp_files(tmp_path):
    root1, root2 = tmp_path / "r1", tmp_path / "r2"
    save_temp_trades(root1 / "temp", _trades(), "ES", False, ["normal"], "all")
    save_temp_trades(root2 / "temp", _trades(), "NQ", False, ["normal"], "all")
    keep = root1 / "temp" / "my_own_notes.parquet"
    _trades().to_parquet(keep)
    no_temp_root = tmp_path / "r3"          # missing temp/ must not raise
    assert clear_temp_files([root1, root2, no_temp_root]) == 2
    assert keep.exists()                    # only the app's pattern is deleted
    assert list((root1 / "temp").glob("*_temp_file_*.parquet")) == []
    assert list((root2 / "temp").glob("*_temp_file_*.parquet")) == []
