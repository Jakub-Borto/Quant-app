"""
Performance metric tiles + exit breakdown table — the Qt analog of the old
render_metrics. Row layout, labels, format strings and help texts are
verbatim from legacy_streamlit/views/trade_report.py.
"""

import pandas as pd
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from modules.common.backend.trade_stats import (compute_metrics,
                                                exit_breakdown_table)
from ..dataframe_model import make_table_view, update_table_view
from ..widgets import MetricTile, SectionHeader

_SHARPE_DAILY_HELP = ("daily P&L over every business day between first and "
                      "last trade — days without trades count as 0; ×√252")
_SHARPE_TRADE_HELP = "daily P&L over days with at least one trade; ×√252"


class MetricsSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(SectionHeader("Performance"))

        self._tiles: dict[str, MetricTile] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        rows = [
            ["Total Ticks", "Total Trades", "Avg Trade/Expectancy", "Avg Win",
             "Avg Loss", "Largest Win", "Largest Loss", "Profit Factor"],
            ["Win Rate", "Loss Rate", "Breakeven Rate", "Sharpe (daily)",
             "Sharpe (traded days)", "Calmar"],
            ["Planned Avg RR", "Planned Median RR", "Realised Avg RR",
             "Realised Median RR"],
            ["Max Drawdown", "Max Peak", "Consec. Wins", "Consec. Losses",
             "Avg Duration", "Median Duration"],
            ["Long Win Rate", "Short Win Rate", "Long Trades", "Short Trades"],
        ]
        helps = {"Sharpe (daily)": _SHARPE_DAILY_HELP,
                 "Sharpe (traded days)": _SHARPE_TRADE_HELP}
        for r, labels in enumerate(rows):
            for c, label in enumerate(labels):
                tile = MetricTile(label, "—", helps.get(label, ""))
                self._tiles[label] = tile
                grid.addWidget(tile, r, c)
        lay.addLayout(grid)

        lay.addWidget(SectionHeader("Exit Breakdown"))
        self._exit_table = make_table_view(pd.DataFrame(), height=190)
        lay.addWidget(self._exit_table)

    def set_trades(self, trades: pd.DataFrame) -> None:
        m = compute_metrics(trades)

        pf_display     = "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
        calmar_display = "∞" if m["calmar"]        == float("inf") else f"{m['calmar']:.2f}"
        avg_dur    = f"{m['avg_duration_min']:.0f}m"    if m["avg_duration_min"]    is not None else "N/A"
        median_dur = f"{m['median_duration_min']:.0f}m" if m["median_duration_min"] is not None else "N/A"

        def _rr(key):
            return f"{m[key]:.2f}" if m[key] is not None else "N/A"

        values = {
            "Total Ticks":          f"{m['total_ticks']:.0f}",
            "Total Trades":         str(m["total_trades"]),
            "Avg Trade/Expectancy": f"{m['avg_trade']:.2f}",
            "Avg Win":              f"{m['avg_win']:.1f}",
            "Avg Loss":             f"{m['avg_loss']:.1f}",
            "Largest Win":          f"{m['largest_win']:.0f}",
            "Largest Loss":         f"{m['largest_loss']:.0f}",
            "Profit Factor":        pf_display,
            "Win Rate":             f"{m['win_rate']:.1%}",
            "Loss Rate":            f"{m['loss_rate']:.1%}",
            "Breakeven Rate":       f"{m['breakeven_rate']:.1%}",
            "Sharpe (daily)":       f"{m['sharpe_daily']:.2f}",
            "Sharpe (traded days)": f"{m['sharpe_trade']:.2f}",
            "Calmar":               calmar_display,
            "Planned Avg RR":       _rr("avg_rr_planned"),
            "Planned Median RR":    _rr("median_rr_planned"),
            "Realised Avg RR":      _rr("avg_rr_realised"),
            "Realised Median RR":   _rr("median_rr_realised"),
            "Max Drawdown":         f"{m['max_drawdown']:.0f} ticks",
            "Max Peak":             f"{m['max_peak']:.0f} ticks",
            "Consec. Wins":         str(m["max_consec_wins"]),
            "Consec. Losses":       str(m["max_consec_losses"]),
            "Avg Duration":         avg_dur,
            "Median Duration":      median_dur,
            "Long Win Rate":        f"{m['long_winrate']:.1%}",
            "Short Win Rate":       f"{m['short_winrate']:.1%}",
            "Long Trades":          str(m["long_trades"]),
            "Short Trades":         str(m["short_trades"]),
        }
        for label, value in values.items():
            self._tiles[label].set_value(value)

        update_table_view(self._exit_table, exit_breakdown_table(trades))
