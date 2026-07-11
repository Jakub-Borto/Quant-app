"""
Optimizer cell drill-down — everything the Backtester shows, for the clicked
heatmap cell's trades.

The PySide6 port of _render_cell_detail: verbatim x/y/slider/half filtering,
the ticks alias (pnl_ticks) + the day_bucket -> day_type historical rename,
then the backtester-shaped chain: trade-type filter -> news/holiday table ->
day-type filter (defaults follow the heatmap's day-bucket selection) ->
shared TradeReportPanel -> trades table -> the shared TradeActionsRow:
Save Trades writes the cell's filtered trades into the data root's trades/
(named ticker_strategy_dates + the cell's param combination), and Go to
Analytics / Go to Monte Carlo hand them off via a temp file.
"""

import re

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from modules.common.backend.data_roots import DatasetRef  # noqa: F401 (typing)
from modules.common.backend.trade_stats import DAY_TYPE_ORDER
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.trade_report.filters import (make_day_type_filter,
                                                    make_trade_type_filter)
from modules.common.ui.trade_report.actions_row import TradeActionsRow
from modules.common.ui.trade_report.news_section import NewsBreakdownTable
from modules.common.ui.trade_report.panel import TradeReportPanel
from modules.common.ui.widgets import Banner, Caption, SectionHeader, hline
from modules.optimizer.backend.heatmap_model import _fmt_axis_value


class CellDetailPanel(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._cell_df: pd.DataFrame | None = None   # cell trades pre type/day filters
        # handoff context for the Go to Analytics / Monte Carlo row
        self._filtered_trades: pd.DataFrame | None = None
        self._filtered = False
        self._selected_day_types: list = []
        self._selected_trade_types_meta = "all"
        self._run_root = None
        self._asset = None
        self._meta: dict = {}
        self._cell_desc: list[str] = []

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

        self._actions_row = TradeActionsRow(settings, self._actions_context,
                                            self._banner)
        actions_lay = QHBoxLayout()
        actions_lay.addStretch()
        actions_lay.addWidget(self._actions_row)
        actions_lay.addStretch()
        lay.addLayout(actions_lay)
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
        self._run_root = run_root
        self._asset = meta.get("ticker")
        self._meta = meta
        self._cell_desc = desc

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
        self._filtered_trades = None

    # ── filters (verbatim backtester ordering) ────────────────────────────────
    def _apply_filters(self) -> None:
        if self._cell_df is None:
            return
        df = self._cell_df
        self._banner.clear_message()

        self._selected_trade_types_meta = "all"
        trade_type_filtered = False
        if self._tt_filter is not None:
            unique_types = sorted(df["trade_type"].dropna().unique().tolist())
            selected_types = self._tt_filter.selected()
            if not selected_types:
                self._banner.show_message("warning", "No trade types selected.")
                self._set_report_visible(False)
                return
            df = df[df["trade_type"].isin(selected_types)]
            trade_type_filtered = len(selected_types) < len(unique_types)
            if trade_type_filtered:
                self._selected_trade_types_meta = selected_types

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

        self._filtered = (trade_type_filtered
                          or len(selected_day_types) < len(DAY_TYPE_ORDER))
        self._selected_day_types = selected_day_types
        self._filtered_trades = df

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
        self._actions_row.setVisible(visible)

    def _actions_context(self) -> dict | None:
        if (self._filtered_trades is None or self._run_root is None
                or not self._asset):
            return None
        # day_bucket is derived (like day_type, which the shared row strips)
        trades = self._filtered_trades.drop(columns=["day_bucket"],
                                            errors="ignore")
        # ticker_strategy_dates + the cell's param combination, filename-safe
        # (the ticker MUST stay the first underscore token — asset lookups
        # downstream key off it)
        parts = [self._asset, self._meta.get("strategy"),
                 self._meta.get("start_date"), self._meta.get("end_date"),
                 *self._cell_desc]
        save_name = "_".join(
            re.sub(r"[^A-Za-z0-9._\-]+", "-", str(p)).strip("-")
            for p in parts if p)
        return {"trades": trades, "asset": self._asset,
                "root": self._run_root, "save_name": save_name,
                "filtered": self._filtered,
                "day_types": self._selected_day_types,
                "trade_types": self._selected_trade_types_meta}
