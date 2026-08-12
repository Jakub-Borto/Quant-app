"""
The shared trade report: the canonical frame contract, and the guarantee that
there is exactly ONE filter chain behind all three modules.

The headline test is the one that was impossible to write while the chain
existed twice — feed the Backtester's frame shape and an optimizer run's
frame shape into the same public API and assert the reports agree row for row.
"""

import json
from datetime import date

import pandas as pd
import pytest

from modules.common.trade_report import (DEFAULT_SECTIONS, KEEP, ReportContext,
                                         SaveTarget, TradeFrameError,
                                         canonical_trades, from_optimizer_rows,
                                         save_name)
from modules.common.trade_report.backend.frame import (apply_mask, entry_frame,
                                                       narrow)

pytest.importorskip("PySide6")

from modules.common.backend.settings import Settings          # noqa: E402
from modules.common.trade_report.backend.trade_stats import (  # noqa: E402
    DAY_TYPE_ORDER)
from modules.common.trade_report.ui import TradeReport         # noqa: E402
from modules.common.trade_report.ui.regime_section import (     # noqa: E402
    FILTER_COLUMN)

ALL_DAY_TYPES = {tag for tag, _ in DAY_TYPE_ORDER}

# One logical trade set. Row 2 is deliberately an `other_high_impact` day, so
# the optimizer shape exercises the bucket rename.
_ROWS = [
    # ticks, day_type,      trade_type, tp_type,     flip
    (40.0,  "normal",       "alpha",    "tp_vwap_2", 0),
    (-10.0, "normal",       "beta",     "tp_vwap_3", 3),
    (120.0, "high_impact",  "alpha",    "1:1",       3),
    (60.0,  "normal",       "alpha",    "tp_vwap_2", 1),
]


def _rows(n=len(_ROWS)):
    return _ROWS[:n]


def _common(n):
    days = [f"2026-01-{5 + i:02d}" for i in range(n)]
    return {
        "entry_time": pd.to_datetime([f"{d} 09:{30 + i:02d}"
                                      for i, d in enumerate(days)]),
        "exit_time": pd.to_datetime([f"{d} 10:30" for d in days]),
        "direction": ["long"] * n, "entry_price": [5000.0] * n,
        "exit_price": [5002.5] * n, "exit_reason": ["target"] * n,
        "trade_type": [r[2] for r in _rows(n)],
        "notes": [json.dumps({"tp_type": r[3], "flip_count": r[4]})
                  for r in _rows(n)],
    }, days


def backtester_frame(n=len(_ROWS)) -> pd.DataFrame:
    """What tag_trades hands over: ticks + cumulative_ticks + day_type, and a
    `date` that is a STRING (strategies emit date_str)."""
    data, days = _common(n)
    frame = pd.DataFrame({"date": days, **data})
    frame["ticks"] = [r[0] for r in _rows(n)]
    frame["cumulative_ticks"] = frame["ticks"].cumsum()
    frame["day_type"] = [r[1] for r in _rows(n)]
    return frame


def optimizer_frame(n=len(_ROWS)) -> pd.DataFrame:
    """What an optimizer run stores: pnl_ticks + day_bucket, `date`
    normalized to datetime64[ns]."""
    data, days = _common(n)
    frame = pd.DataFrame({"date": pd.to_datetime(days), **data})
    frame["pnl_ticks"] = [r[0] for r in _rows(n)]
    frame["day_bucket"] = [("other_high_impact" if r[1] == "high_impact"
                            else r[1]) for r in _rows(n)]
    return frame


@pytest.fixture
def report(qtbot, tmp_path):
    widget = TradeReport(Settings({}, [str(tmp_path)]),
                         track_worker=lambda w: None)
    qtbot.addWidget(widget)
    widget.set_context(ReportContext(
        ticker="ES", tick_size=0.25, ticks_per_point=4.0, root=tmp_path,
        candles_folder=tmp_path / "parquet"))
    return widget


# ══ the canonical shape ══════════════════════════════════════════════════════
def test_both_producers_normalise_to_the_same_frame():
    bt = canonical_trades(backtester_frame())
    opt = from_optimizer_rows(optimizer_frame())
    for column in ("ticks", "cumulative_ticks", "day_type", "trade_type"):
        pd.testing.assert_series_equal(bt[column], opt[column],
                                       check_dtype=False)
    assert bt.index.equals(opt.index)


def test_canonical_trades_is_idempotent():
    once = canonical_trades(backtester_frame())
    twice = canonical_trades(once)
    pd.testing.assert_frame_equal(once, twice)


def test_canonical_trades_never_touches_the_date_column():
    """Coercing it would change the schema of every Backtester trades file,
    and save_trades dedups by comparing whole frames against what is on disk."""
    bt = backtester_frame()
    assert canonical_trades(bt)["date"].dtype == bt["date"].dtype
    opt = optimizer_frame()
    out = from_optimizer_rows(opt)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_the_sort_is_stable_so_tied_entry_times_keep_their_order():
    frame = backtester_frame(3)
    frame["entry_time"] = pd.to_datetime(["2026-01-05 09:30"] * 3)
    out = canonical_trades(frame)
    assert out["ticks"].tolist() == [r[0] for r in _rows(3)]
    assert out["cumulative_ticks"].tolist() == [40.0, 30.0, 150.0]


def test_a_missing_column_is_named_rather_than_raising_keyerror():
    with pytest.raises(TradeFrameError) as excinfo:
        canonical_trades(backtester_frame().drop(columns=["day_type"]))
    assert "day_type" in str(excinfo.value)
    with pytest.raises(TradeFrameError) as excinfo:
        from_optimizer_rows(backtester_frame())      # has no pnl_ticks
    assert "pnl_ticks" in str(excinfo.value)


def test_narrow_recomputes_cumulative_ticks():
    frame = canonical_trades(backtester_frame())
    out = narrow(frame, frame["day_type"] == "normal")
    assert out["cumulative_ticks"].tolist() == [40.0, 30.0, 90.0]
    assert out.index.tolist() == [0, 1, 3]           # labels preserved


def test_entry_frame_and_apply_mask():
    frame = canonical_trades(backtester_frame())
    assert len(entry_frame(frame, "day_type", ["normal"])) == 3
    assert entry_frame(frame, "nope", ["x"]) is frame
    assert apply_mask(frame, None) is frame
    assert len(apply_mask(frame, frame["ticks"] > 0)) == 3


@pytest.mark.parametrize("parts, expected", [
    (["ES_1m_ohlcv", "orb", date(2026, 1, 5), date(2026, 2, 1)],
     "ES_1m_ohlcv_orb_2026-01-05_2026-02-01"),
    (["ES", "run", None, "", "k2"], "ES_run_k2"),
    (["ES data", "my strategy"], "ES-data_my-strategy"),
])
def test_save_name(parts, expected):
    assert save_name(parts) == expected


def test_section_keys_are_frozen():
    """These strings ARE the persistence: resolve_layout silently drops keys
    it does not recognise, so a rename resets every user's saved layout
    without raising. Change this literal only on purpose."""
    assert tuple(s.key for s in DEFAULT_SECTIONS) == (
        "trade_type_filter", "day_type_filter", "metrics", "equity",
        "chart_controls", "trade_detail", "regime", "entry_breakdown", "news",
        "exposure", "exit_breakdown", "rr", "trades_table", "actions")


# ══ ONE chain, two callers ═══════════════════════════════════════════════════
def _drive(report, frame, **kwargs):
    report.show_trades(frame, save_target=SaveTarget(["ES", "run"]), **kwargs)
    return report


def test_backtester_and_optimizer_shapes_produce_the_same_report(qtbot, tmp_path):
    """The point of the refactor: the two modules cannot drift, because there
    is only one implementation left to drift."""
    def build(frame):
        widget = TradeReport(Settings({}, [str(tmp_path)]),
                             track_worker=lambda w: None)
        qtbot.addWidget(widget)
        widget.set_context(ReportContext(
            ticker="ES", tick_size=0.25, ticks_per_point=4.0, root=tmp_path,
            candles_folder=tmp_path / "parquet"))
        _drive(widget, frame, day_types=ALL_DAY_TYPES)
        return widget

    bt = build(canonical_trades(backtester_frame()))
    opt = build(from_optimizer_rows(optimizer_frame()))

    for column in ("ticks", "cumulative_ticks", "day_type", "trade_type"):
        pd.testing.assert_series_equal(bt.filtered_trades()[column],
                                       opt.filtered_trades()[column],
                                       check_dtype=False)
    assert bt.filtered_trades().index.equals(opt.filtered_trades().index)
    assert bt.is_filtered() == opt.is_filtered() is False
    assert bt.selected_day_types() == opt.selected_day_types()
    assert bt.selected_trade_types() == opt.selected_trade_types() == "all"

    # and the same narrowing lands identically on both
    for widget in (bt, opt):
        widget._tt_filter._boxes["beta"].setChecked(False)
    pd.testing.assert_series_equal(bt.filtered_trades()["cumulative_ticks"],
                                   opt.filtered_trades()["cumulative_ticks"],
                                   check_dtype=False)
    assert bt.is_filtered() is opt.is_filtered() is True


@pytest.mark.parametrize("shape", ["backtester", "optimizer"])
def test_cumulative_ticks_is_correct_at_the_news_stage(report, shape, monkeypatch):
    """The Optimizer's chain used to skip the recompute after the trade-type
    filter, so the news table saw a stale cumsum — latent only because that
    table happens to read `ticks`."""
    seen = {}
    original = report._news.set_trades
    monkeypatch.setattr(report._news, "set_trades",
                        lambda df: (seen.setdefault("df", df.copy()),
                                    original(df))[1])
    frame = (canonical_trades(backtester_frame()) if shape == "backtester"
             else from_optimizer_rows(optimizer_frame()))
    _drive(report, frame, day_types=ALL_DAY_TYPES)
    report._tt_filter._boxes["beta"].setChecked(False)

    got = seen["df"]
    assert got["cumulative_ticks"].tolist() == got["ticks"].cumsum().tolist()


def test_the_chain_preserves_index_labels(report):
    """TradeNotesSection.mask_for reindexes a source-indexed mask onto each
    stage's frame; a stage that reset the index would silently yield an
    all-False mask, i.e. 'no trades match' out of nowhere."""
    _drive(report, from_optimizer_rows(optimizer_frame()),
           day_types=ALL_DAY_TYPES)
    source_labels = set(report._source.index)
    report._tt_filter._boxes["beta"].setChecked(False)
    assert set(report.filtered_trades().index) <= source_labels
    assert report._source.index.is_unique


# ══ the day-type / trade-type both-behaviours requirement ════════════════════
def test_day_types_keep_leaves_the_row_alone_and_a_set_resets_it(report):
    _drive(report, canonical_trades(backtester_frame()),
           day_types=ALL_DAY_TYPES)
    report._dt_filter._boxes["holiday"].setChecked(False)

    # KEEP is the default — this is what makes a Backtester re-run keep the
    # user's day-type choice
    _drive(report, canonical_trades(backtester_frame()))
    assert not report._dt_filter._boxes["holiday"].isChecked()

    # an explicit set rebuilds it — how the Optimizer follows the heatmap
    _drive(report, canonical_trades(backtester_frame()),
           day_types={"normal"})
    assert report._dt_filter.selected() == ["normal"]


def test_keep_trade_types_carries_ticks_and_auto_checks_new_types(report):
    frames = {"all": backtester_frame(), "oos": backtester_frame(2)}
    _drive(report, canonical_trades(frames["all"]), day_types=ALL_DAY_TYPES)
    assert set(report._tt_filter.selected()) == {"alpha", "beta"}

    report._tt_filter._boxes["beta"].setChecked(False)
    _drive(report, canonical_trades(frames["oos"]), keep_trade_types=True,
           day_types=KEEP)
    assert set(report._tt_filter.selected()) == {"alpha"}    # uncheck survives

    # without the flag a re-slice starts fresh (the Backtester's per-run reset)
    _drive(report, canonical_trades(frames["all"]))
    assert set(report._tt_filter.selected()) == {"alpha", "beta"}


def test_a_type_the_previous_slice_never_offered_arrives_checked(report):
    """A member with no trades in one scope must not come back silently
    filtered out."""
    narrow_frame = backtester_frame(2)
    narrow_frame = narrow_frame[narrow_frame["trade_type"] == "alpha"]
    _drive(report, canonical_trades(narrow_frame), day_types=ALL_DAY_TYPES)
    assert set(report._tt_filter.selected()) == {"alpha"}
    _drive(report, canonical_trades(backtester_frame()), keep_trade_types=True)
    assert set(report._tt_filter.selected()) == {"alpha", "beta"}


# ══ what the duplication was hiding ══════════════════════════════════════════
def test_a_save_confirmation_survives_a_filter_toggle(report):
    """The old optimizer host shared ONE banner between filter messages and
    the actions row, so any toggle after a save wiped the confirmation."""
    _drive(report, canonical_trades(backtester_frame()),
           day_types=ALL_DAY_TYPES)
    report._save_banner.show_message("success", "Saved to trades/ES_run.parquet")

    report._dt_filter._boxes["holiday"].setChecked(False)
    assert "Saved to" in report._save_banner.text()
    report._tt_filter._boxes["beta"].setChecked(False)
    assert "Saved to" in report._save_banner.text()

    # …and a NEW selection does clear it
    _drive(report, canonical_trades(backtester_frame()))
    assert report._save_banner.text() == ""


def test_the_regime_filter_column_floats_for_every_caller(report, monkeypatch):
    """The old optimizer host never called set_display_hints, so its trades
    table hid regime_filter exactly when it disagreed with regime."""
    monkeypatch.setattr(report._regime, "timings_differ", lambda: True)
    frame = from_optimizer_rows(optimizer_frame())
    frame[FILTER_COLUMN] = "trend"
    _drive(report, frame, day_types=ALL_DAY_TYPES)
    assert FILTER_COLUMN in report._notes._lead_columns


# ══ empty / filtered-out states ══════════════════════════════════════════════
def test_an_empty_selection_shows_the_message_and_nulls_the_state(report):
    _drive(report, canonical_trades(backtester_frame()))
    report.show_trades(backtester_frame().iloc[0:0])
    assert report.filtered_trades() is None
    assert report._source is None
    assert "No trades" in report._filter_banner.text()


def test_an_empty_filter_selection_is_a_warning_for_every_stage(report):
    _drive(report, canonical_trades(backtester_frame()),
           day_types=ALL_DAY_TYPES)
    for box in report._dt_filter._boxes.values():
        box.setChecked(False)
    assert "No day types selected" in report._filter_banner.text()
    assert not report.panel.sections.frame("metrics").isVisible()


def test_clear_hides_and_forgets(report):
    _drive(report, canonical_trades(backtester_frame()))
    report.clear()
    assert report.filtered_trades() is None
    assert not report.isVisible()


# ══ the save handoff ═════════════════════════════════════════════════════════
def test_the_actions_context_is_built_once_for_every_caller(report):
    _drive(report, from_optimizer_rows(optimizer_frame()),
           day_types=ALL_DAY_TYPES)
    report._save = SaveTarget(["ES", "run", "k2", "all"],
                              drop_columns=("day_bucket", "combine_vid"))
    ctx = report._actions_context()
    assert ctx["asset"] == "ES"
    assert ctx["save_name"] == "ES_run_k2_all"
    assert ctx["filtered"] is False
    assert ctx["trade_types"] == "all"
    assert "day_bucket" not in ctx["trades"].columns   # module-private column
    assert "ticks" in ctx["trades"].columns


def test_no_save_target_makes_the_buttons_no_ops(report):
    report.show_trades(canonical_trades(backtester_frame()))
    assert report._actions_context() is None
