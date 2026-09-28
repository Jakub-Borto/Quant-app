"""
Picking whose reference data the trade report uses — the regime runs and the
Market Exposure benchmark. Both default to the backtest's asset, fall back to
its full-size parent when only the parent has data (MES -> ES, NNQ -> NQ), and
keep a manual pick until the backtest's asset changes.
"""

import json

import pandas as pd
import pytest

from modules.common.backend.asset_info import (ASSET_INFO, _micro_child,
                                               default_reference_asset,
                                               get_commission_info,
                                               root_asset, same_underlying)
from modules.common.trade_report.backend.benchmark import list_statistics_assets


# ── asset relationships ───────────────────────────────────────────────────────
def test_root_asset_follows_parent_links():
    assert root_asset("MES") == "ES"
    assert root_asset("MNQ") == "NQ"
    assert root_asset("NNQ") == "NQ"
    assert root_asset("ES") == "ES"
    assert root_asset("UNKNOWN") == "UNKNOWN"


def test_nq_has_three_contract_sizes():
    sizes = {a for a in ASSET_INFO if root_asset(a) == "NQ"}
    assert sizes == {"NQ", "MNQ", "NNQ"}
    assert ASSET_INFO["NNQ"]["tick_size"] == 0.50      # CME: 0.50 pts = $0.10
    assert ASSET_INFO["NNQ"]["dollars_per_tick"] == 0.10
    assert ASSET_INFO["NNQ"]["ticks_per_point"] == 2


def test_adding_nnq_does_not_change_nqs_micro():
    assert _micro_child("NQ") == "MNQ"
    assert get_commission_info("NNQ_x.parquet") == (None, None)


def test_same_underlying():
    assert same_underlying("NNQ", "MNQ")
    assert same_underlying("ES", "MES")
    assert not same_underlying("ES", "NQ")
    assert not same_underlying(None, "ES")


@pytest.mark.parametrize("asset, available, expected", [
    ("MES", {"ES", "NQ"}, "ES"),        # micro reads its parent
    ("NNQ", {"ES", "NQ"}, "NQ"),
    ("MES", {"MES", "ES"}, "MES"),      # own data wins
    ("ES", {"ES"}, "ES"),
    ("CL", {"ES"}, "CL"),               # nothing related -> itself ("no data")
    ("MES", set(), "MES"),
    (None, {"ES"}, None),
])
def test_default_reference_asset(asset, available, expected):
    assert default_reference_asset(asset, available) == expected


def _stats_file(root, asset):
    path = root / "Futures" / asset / f"{asset}_statistics" / "statistics.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    return path


def test_list_statistics_assets(tmp_path):
    _stats_file(tmp_path, "NQ")
    _stats_file(tmp_path, "ES")
    (tmp_path / "Futures" / "MES" / "MES_1m").mkdir(parents=True)
    assert list_statistics_assets(tmp_path) == ["ES", "NQ"]
    assert list_statistics_assets(tmp_path / "missing") == []
    assert list_statistics_assets(None) == []


# ── Market Exposure benchmark picker ──────────────────────────────────────────
pytest.importorskip("PySide6")


def _trades():
    entry = pd.date_range("2026-01-05 10:00", periods=4, freq="D",
                          tz="America/New_York")
    return pd.DataFrame({"date": entry.tz_localize(None).normalize(),
                         "entry_time": entry, "ticks": [1.0, -2.0, 3.0, 4.0]})


@pytest.fixture
def exposure(qtbot, tmp_path, monkeypatch):
    from modules.common.trade_report.ui import exposure_section as ex
    _stats_file(tmp_path, "ES")
    _stats_file(tmp_path, "NQ")
    calls = []
    monkeypatch.setattr(ex, "load_asset_statistics",
                        lambda root, asset: calls.append(("load", asset)) or object())
    monkeypatch.setattr(ex, "market_exposure_data",
                        lambda trades, stats, tick: calls.append(("tick", tick)) or None)
    section = ex.ExposureSection()
    qtbot.addWidget(section)
    return section, calls, tmp_path


def test_micro_backtest_defaults_to_the_parent_benchmark(exposure):
    section, calls, root = exposure
    section.update_exposure(_trades(), "MES", 0.25, root)
    assert section.benchmark_asset() == "ES"
    assert ("load", "ES") in calls


def test_own_statistics_win_and_the_backtest_tick_size_is_used(exposure):
    section, calls, root = exposure
    section.update_exposure(_trades(), "NQ", 0.25, root)
    assert section.benchmark_asset() == "NQ"
    calls.clear()
    section.update_exposure(_trades(), "NNQ", 0.50, root)
    assert section.benchmark_asset() == "NQ"
    # NNQ P&L is in 0.50 ticks, so NQ's move must be too
    assert ("tick", 0.50) in calls


def test_a_manual_pick_sticks_until_the_asset_changes(exposure):
    section, calls, root = exposure
    section.update_exposure(_trades(), "MES", 0.25, root)
    section._bench.setCurrentIndex(section._bench.findText("NQ"))
    assert section.benchmark_asset() == "NQ"
    calls.clear()
    section.update_exposure(_trades(), "MES", 0.25, root)     # re-run / filter
    assert section.benchmark_asset() == "NQ" and ("load", "NQ") in calls
    section.update_exposure(_trades(), "MNQ", 0.25, root)     # new instrument
    assert section.benchmark_asset() == "NQ"                  # its default
    section.update_exposure(_trades(), "MES", 0.25, root)
    assert section.benchmark_asset() == "ES"                  # pick was reset


def test_an_asset_without_statistics_is_still_offered(exposure):
    section, calls, root = exposure
    section.update_exposure(_trades(), "CL", 0.01, root)
    assert section.benchmark_asset() == "CL"
    assert [section._bench.itemText(i) for i in range(section._bench.count())] \
        == ["CL", "ES", "NQ"]


# ── regime asset picker ───────────────────────────────────────────────────────
@pytest.fixture
def regime(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.common.trade_report.ui.regime_section import RegimeSection
    for asset, run in (("ES", "ES_vol"), ("ES", "ES_trend"), ("NQ", "NQ_vol")):
        d = tmp_path / "regimes" / asset / run
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({}))
    section = RegimeSection(Settings({}, [str(tmp_path)]), lambda w: None)
    qtbot.addWidget(section)
    return section


def _run_assets(section):
    return {section._run.itemData(i).asset for i in range(section._run.count())}


def test_regime_runs_default_to_the_parent_for_a_micro(regime):
    regime.set_context("MES", None, None)
    assert regime.regime_asset() == "ES"
    assert _run_assets(regime) == {"ES"}
    assert "different underlying" not in regime._banner.text()


def test_regime_asset_pick_sticks_until_the_asset_changes(regime):
    regime.set_context("MES", None, None)
    regime._regime_asset.setCurrentIndex(regime._regime_asset.findText("NQ"))
    assert _run_assets(regime) == {"NQ"}
    assert "different underlying" in regime._banner.text()
    regime.set_context("MES", None, None)                     # re-run
    assert regime.regime_asset() == "NQ"
    regime.set_context("ES", None, None)                      # new instrument
    assert regime.regime_asset() == "ES"


def test_show_all_assets_still_lists_everything(regime):
    regime.set_context("MES", None, None)
    regime._all_assets.setChecked(True)
    assert _run_assets(regime) == {"ES", "NQ"}
    assert not regime._regime_asset.isEnabled()
    regime._all_assets.setChecked(False)
    assert _run_assets(regime) == {"ES"}
