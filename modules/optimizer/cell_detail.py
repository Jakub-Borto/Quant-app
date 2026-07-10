"""
Optimizer cell drill-down — everything the Backtester shows, for the clicked
heatmap cell's trades.

The PySide6 port of _render_cell_detail: verbatim x/y/slider/half filtering,
the ticks alias (pnl_ticks) + the day_bucket -> day_type historical rename,
then the backtester-shaped chain: trade-type filter -> news/holiday table ->
day-type filter (defaults follow the heatmap's day-bucket selection) ->
shared TradeReportPanel -> trades table. No saving — the run's trades already
live in the optimization folder.
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout, QWidget

from modules.common.backend.data_roots import DatasetRef  # noqa: F401 (typing)
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.trade_report.filters import (make_day_type_filter,
                                                    make_trade_type_filter)
from modules.common.ui.trade_report.news_section import NewsBreakdownTable
from modules.common.ui.trade_report.panel import TradeReportPanel
from modules.common.ui.widgets import Banner, Caption, SectionHeader, hline
from modules.optimizer.backend.heatmap_model import _fmt_axis_value


class CellDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cell_df: pd.DataFrame | None = None   # cell trades pre type/day filters

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(hline())
        self._header = SectionHeader("Cell detail")
        lay.addWidget(self._header)
        self._banner = Banner()
        lay.addWidget(self._banner)

        self._tt_caption = Caption("Filter by trade type")
        self._tt_caption.setVisible(False)
        lay.addWidget(self._tt_caption)
        self._tt_holder = QVBoxLayout()
        lay.addLayout(self._tt_holder)
        self._tt_filter = None

        self._news = NewsBreakdownTable()
        lay.addWidget(self._news)

        self._dt_caption = Caption("Filter by day type")
        lay.addWidget(self._dt_caption)
        self._dt_holder = QVBoxLayout()
        lay.addLayout(self._dt_holder)
        self._dt_filter = None

        self._panel = TradeReportPanel()
        lay.addWidget(self._panel)

        self._table_header = SectionHeader("Trades")
        lay.addWidget(self._table_header)
        self._table = make_table_view(pd.DataFrame(), height=380)
        lay.addWidget(self._table)
        self.setVisible(False)

    # ── entry point from the heatmap click ────────────────────────────────────
    def show_cell(self, trades: pd.DataFrame, meta: dict, x_axis, y_axis,
                  slider_axes, slider_values, half, split, cell_ij,
                  selected_buckets, run_root) -> None:
        xi, yj = cell_ij
        if xi >= len(x_axis["values"]) or (y_axis and yj >= len(y_axis["values"])):
            return                                  # stale selection after reload

        desc = []
        df = trades
        x_val = x_axis["values"][xi]
        df = df[df[x_axis["param"]] == x_val]
        desc.append(f"{x_axis['param']} = {_fmt_axis_value(x_val)}")
        if y_axis is not None:
            y_val = y_axis["values"][yj]
            df = df[df[y_axis["param"]] == y_val]
            desc.append(f"{y_axis['param']} = {_fmt_axis_value(y_val)}")
        for ax in slider_axes:
            value = slider_values[ax["param"]]
            df = df[df[ax["param"]] == value]
            desc.append(f"{ax['param']} = {_fmt_axis_value(value)}")
        if half != "both" and split is not None:
            dates = pd.to_datetime(df["date"])
            split_ts = pd.Timestamp(split)
            df = df[dates <= split_ts] if half == "1st" else df[dates > split_ts]
            desc.append(f"{half} half")

        self._header.setText("Cell detail — " + " · ".join(desc))
        self.setVisible(True)
        self._banner.clear_message()

        if df.empty:
            self._banner.show_message("info", "No trades in this cell.")
            self._set_report_visible(False)
            self._cell_df = None
            return

        # backtester-shaped columns: ticks alias + historical day_type names
        df = df.sort_values("entry_time").reset_index(drop=True)
        df["ticks"] = df["pnl_ticks"]
        df["day_type"] = df["day_bucket"].replace(
            {"other_high_impact": "high_impact"})
        self._cell_df = df

        # (re)build the filter rows for this cell
        if self._tt_filter is not None:
            self._tt_filter.deleteLater()
            self._tt_filter = None
        unique_types = []
        if "trade_type" in df.columns:
            unique_types = sorted(df["trade_type"].dropna().unique().tolist())
        self._tt_caption.setVisible(bool(unique_types))
        if unique_types:
            self._tt_filter = make_trade_type_filter(unique_types)
            self._tt_filter.selectionChanged.connect(self._apply_filters)
            self._tt_holder.addWidget(self._tt_filter)

        # day-type defaults follow the heatmap's day-bucket selection
        heat_defaults = {("high_impact" if b == "other_high_impact" else b)
                         for b in selected_buckets}
        if self._dt_filter is not None:
            self._dt_filter.deleteLater()
        self._dt_filter = make_day_type_filter(checked_tags=heat_defaults)
        self._dt_filter.selectionChanged.connect(self._apply_filters)
        self._dt_holder.addWidget(self._dt_filter)

        self._panel.set_context(
            meta.get("ticker"), meta.get("tick_size"),
            meta.get("ticks_per_point"),
            candles_folder=run_root / "parquet" / meta.get("dataset", ""),
            parquet_root=run_root / "parquet")
        self._apply_filters()

    def hide_detail(self) -> None:
        self.setVisible(False)
        self._cell_df = None

    # ── filters (verbatim backtester ordering) ────────────────────────────────
    def _apply_filters(self) -> None:
        if self._cell_df is None:
            return
        df = self._cell_df
        self._banner.clear_message()

        if self._tt_filter is not None:
            selected_types = self._tt_filter.selected()
            if not selected_types:
                self._banner.show_message("warning", "No trade types selected.")
                self._set_report_visible(False)
                return
            df = df[df["trade_type"].isin(selected_types)]

        # news & holiday breakdown — before the day filter (backtester order)
        self._news.set_trades(df)

        selected_day_types = self._dt_filter.selected()
        if not selected_day_types:
            self._banner.show_message("warning", "No day types selected.")
            self._set_report_visible(False)
            return
        df = df[df["day_type"].isin(selected_day_types)].copy()
        if df.empty:
            self._banner.show_message("info", "No trades match the selected filters.")
            self._set_report_visible(False)
            return
        df["cumulative_ticks"] = df["ticks"].cumsum()

        self._set_report_visible(True)
        self._panel.set_trades(df)

        display_cols = [c for c in ["date", "direction", "entry_time",
                                    "exit_time", "entry_price", "exit_price",
                                    "exit_reason", "ticks", "trade_type",
                                    "day_type"] if c in df.columns]
        table = df[display_cols].copy()
        table["date"] = pd.to_datetime(table["date"]).dt.date
        update_table_view(self._table, table)

    def _set_report_visible(self, visible: bool) -> None:
        self._panel.setVisible(visible)
        self._table.setVisible(visible)
        self._table_header.setVisible(visible)
