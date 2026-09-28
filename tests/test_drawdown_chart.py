"""
The drawdown pane under the report's equity curve, and the ONE drawdown
definition it shares with the metrics tile (trade_stats.drawdown_series).

The peak starts at 0: a run that opens with losses is in drawdown from its
first trade — the chart and the Max Drawdown tile must agree on that.
"""

import numpy as np
import pandas as pd
import pytest

from modules.common.trade_report.backend.trade_stats import (compute_metrics,
                                                             drawdown_series)


def _trades(ticks):
    ticks = np.asarray(ticks, dtype=float)
    n = len(ticks)
    entry = pd.date_range("2026-01-05 10:00", periods=n, freq="D",
                          tz="America/New_York")
    return pd.DataFrame({
        "date": entry.tz_localize(None).normalize(),
        "direction": ["long"] * n,
        "entry_time": entry, "exit_time": entry + pd.Timedelta(hours=2),
        "entry_price": 5000.0, "exit_price": 5010.0,
        "sl": 4990.0, "tp": 5030.0, "pnl_points": ticks / 4.0,
        "exit_reason": ["tp"] * n,
        "ticks": ticks, "cumulative_ticks": ticks.cumsum(),
        "day_type": "normal",
    })


# ── the definition ────────────────────────────────────────────────────────────
def test_drawdown_counts_from_a_starting_peak_of_zero():
    dd = drawdown_series([-5, -8, 3, 1, 6])
    assert dd.tolist() == [-5, -8, 0, -2, 0]


def test_drawdown_of_a_curve_that_only_rises_is_zero():
    assert drawdown_series([2, 5, 5, 9]).tolist() == [0, 0, 0, 0]


def test_drawdown_of_nothing_is_empty():
    assert drawdown_series([]).size == 0


def test_max_drawdown_metric_uses_the_same_definition():
    # opens with -5, -3 (cum -8), then recovers: the old first-trade peak
    # said -3; measured from 0 it is -8
    m = compute_metrics(_trades([-5, -3, 11, -2, 5]))
    assert m["max_drawdown"] == -8


# ── the chart ─────────────────────────────────────────────────────────────────
pytest.importorskip("PySide6")


@pytest.fixture
def chart(qtbot):
    from modules.common.ui.charts.equity_curve import EquityCurveChart
    c = EquityCurveChart()
    qtbot.addWidget(c)
    return c


def test_drawdown_pane_is_x_linked_and_marks_the_worst_point(chart):
    trades = _trades([10, -4, -9, 6, 12, -3])
    chart.set_trades(trades)
    dd = drawdown_series(trades["cumulative_ticks"])
    assert chart._dd_plot.getViewBox().linkedView(0) is chart._plot.getViewBox()
    assert chart._dd.tolist() == dd.tolist()
    i = int(np.argmin(dd))
    assert chart._max_dd_text is not None
    assert "13" in chart._max_dd_text.textItem.toPlainText()     # 10 -> -3
    assert chart._max_dd_text.pos().x() == pytest.approx(chart._x[i])


def test_no_max_marker_when_never_in_drawdown(chart):
    chart.set_trades(_trades([1, 2, 3]))
    assert chart._max_dd_text is None


def test_selected_trade_is_mirrored_on_the_drawdown_pane(chart):
    chart.set_trades(_trades([10, -4, -9, 6, 12, -3]))
    chart.select(2)
    xs, ys = chart._dd_marker.getData()
    assert xs.tolist() == [chart._x[2]] and ys.tolist() == [chart._dd[2]]
    chart.select(None)
    xs, _ = chart._dd_marker.getData()
    assert len(xs) == 0


def test_hover_shows_the_drawdown_in_ticks(chart):
    chart.set_trades(_trades([10, -4, -9, 6, 12, -3]))
    text = chart._dd_hover_text(chart._x[2], 0.0)
    assert text == "Drawdown: −13 ticks"
    assert chart._dd_hover_text(chart._x[0], 0.0) == "Drawdown: 0 ticks"


def test_axis_toggle_keeps_the_link_the_marker_and_the_selection(chart, qtbot):
    chart.set_trades(_trades([10, -4, -9, 6, 12, -3]))
    chart.select(2)
    chart._axis_btn.click()
    qtbot.wait(20)
    assert chart._x.tolist() == [1, 2, 3, 4, 5, 6]            # trade-number mode
    assert chart._dd_plot.getViewBox().linkedView(0) is chart._plot.getViewBox()
    assert chart._max_dd_text is not None
    xs, _ = chart._dd_marker.getData()
    assert xs.tolist() == [3.0]


def test_crosshair_snaps_to_the_nearest_trade_on_both_plots(chart):
    chart.set_trades(_trades([10, -4, -9, 6, 12, -3]))
    chart._set_crosshair(float(chart._x[4]))
    assert all(line.isVisible() and line.value() == chart._x[4]
               for line in chart._vlines)
    chart._set_crosshair(None)
    assert not any(line.isVisible() for line in chart._vlines)
