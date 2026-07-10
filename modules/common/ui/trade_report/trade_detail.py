"""
Single-trade drill-in for the equity-curve click — metric tiles, parsed
trade notes and the candlestick chart of the trade's session. The Qt analog
of the old render_trade_detail (@st.fragment); all value computations and
format strings are verbatim.
"""

import json
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from modules.common.backend.chart_window import _is_timestamp, resolve_chart_window
from ..charts.candlestick import TradeChart
from ..widgets import Banner, Caption, MetricTile, SectionHeader, hline


class TradeDetailView(QWidget):
    """Hidden until show_trade() is called with an equity-curve click."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self.setVisible(False)

    def clear(self) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.setVisible(False)

    def show_trade(self, trade, chart_settings: dict, folder_path,
                   ticks_per_point: float) -> None:
        self.clear()
        lay = self._lay
        lay.addWidget(hline())
        # date may be a string (orb), date object (ivb) or Timestamp (optimizer)
        lay.addWidget(SectionHeader(
            f"Trade Detail — {pd.Timestamp(trade['date']).date()}"))

        duration  = trade["exit_time"] - trade["entry_time"]
        hours     = int(duration.total_seconds() // 3600)
        minutes   = int((duration.total_seconds() % 3600) // 60)
        sl_ticks  = abs(trade["entry_price"] - trade["sl"]) * ticks_per_point
        tp_ticks  = abs(trade["entry_price"] - trade["tp"]) * ticks_per_point
        actual_rr = tp_ticks / sl_ticks if sl_ticks > 0 else 0

        metric_items = [
            ("Direction",   trade["direction"].upper()),
            ("Tick PnL",    f"{trade['ticks']:.0f}"),
            ("SL Ticks",    f"{sl_ticks:.0f}"),
            ("TP Ticks",    f"{tp_ticks:.0f}"),
            ("RR",          f"{actual_rr:.2f}"),
            ("Duration",    f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"),
            ("Exit Reason", str(trade["exit_reason"])),
        ]
        if "trade_type" in trade.index and pd.notna(trade["trade_type"]):
            metric_items.append(("Trade Type", str(trade["trade_type"])))
        if "day_type" in trade.index and pd.notna(trade["day_type"]):
            metric_items.append(("Day Type", str(trade["day_type"])))

        per_row = 5
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for n, (label, value) in enumerate(metric_items):
            grid.addWidget(MetricTile(label, value), n // per_row, n % per_row)
        lay.addLayout(grid)

        # ── trade notes (JSON column, values wrap — a metric tile would clip) ─
        if "notes" in trade.index and pd.notna(trade["notes"]):
            try:
                notes = json.loads(trade["notes"])
                items = list(notes.items())
                if items:
                    title = QLabel("**Trade notes**")
                    title.setTextFormat(Qt.MarkdownText)
                    lay.addWidget(title)
                    notes_grid = QGridLayout()
                    notes_grid.setHorizontalSpacing(18)
                    per_row = 4
                    for n, (key, val) in enumerate(items):
                        if isinstance(val, list):
                            display = ", ".join(
                                pd.Timestamp(v).strftime("%H:%M") if _is_timestamp(v) else str(v)
                                for v in val
                            )
                        elif _is_timestamp(val):
                            display = pd.Timestamp(val).strftime("%H:%M")
                        else:
                            display = str(val)
                        cell = QWidget()
                        v_lay = QVBoxLayout(cell)
                        v_lay.setContentsMargins(0, 0, 0, 0)
                        v_lay.setSpacing(1)
                        v_lay.addWidget(Caption(key))
                        val_lbl = QLabel(f"**{display}**")
                        val_lbl.setTextFormat(Qt.MarkdownText)
                        val_lbl.setWordWrap(True)
                        v_lay.addWidget(val_lbl)
                        notes_grid.addWidget(cell, n // per_row, n % per_row)
                    lay.addLayout(notes_grid)
            except Exception as e:  # noqa: BLE001 — old view showed a warning
                banner = Banner("warning", str(e))
                lay.addWidget(banner)

        # ── session candlestick chart ─────────────────────────────────────────
        trade_date = pd.Timestamp(trade["date"])
        day_file   = Path(folder_path) / f"{trade_date.date().isoformat()}.parquet"
        if not day_file.exists():
            lay.addWidget(Banner(
                "info", f"Candle file not found: {day_file} — chart unavailable."))
            self.setVisible(True)
            return
        session       = pd.read_parquet(day_file)
        session       = session[session.index.date == trade_date.date()]
        chart_candles = resolve_chart_window(session, trade["entry_time"],
                                             trade["exit_time"], chart_settings)
        chart = TradeChart()
        chart.set_trade(trade, chart_candles, trade["entry_time"], trade["exit_time"])
        lay.addWidget(chart)
        self.setVisible(True)
