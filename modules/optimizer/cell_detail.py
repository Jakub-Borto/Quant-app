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
from modules.common.ui.trade_report.entry_section import EntryBreakdownSection
from modules.common.ui.trade_report.news_section import NewsBreakdownTable
from modules.common.ui.trade_report.panel import TradeReportPanel
from modules.common.ui.trade_report.regime_section import (FILTER_COLUMN,
                                                           RegimeSection)
from modules.common.ui.widgets import (Banner, Caption, SectionHeader, hline,
                                       pin_minimum_height)
from modules.optimizer.backend.heatmap_model import _fmt_axis_value


def _entry_frame(frame, column: str, selected):
    """One filter applied to the entry-breakdown frame — everything the report
    shows EXCEPT the trade-type filter, so entry types stay comparable."""
    if selected is None or column not in frame.columns:
        return frame
    return frame[frame[column].isin(selected)]


class CellDetailPanel(QWidget):
    def __init__(self, settings, track_worker=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        # the Explore tab runs no background work of its own; regime loading
        # needs one, so the window's tracker is threaded down to here
        self._track_worker = track_worker or (lambda w: None)
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
        # the report is nested deeper here than in the Backtester
        # (page -> tabs -> ExploreTab -> CellDetailPanel -> panel -> stack);
        # every link needs pinning or the squeeze reappears at that level
        pin_minimum_height(self)
        lay.addWidget(hline())
        self._header = SectionHeader("Cell detail")
        lay.addWidget(self._header)
        self._banner = Banner()
        lay.addWidget(self._banner)

        self._panel = TradeReportPanel(settings)

        # both filter rows are rebuilt per cell, so each lives in a stable
        # container that gets registered once
        self._tt_container = QWidget()
        self._tt_holder = QVBoxLayout(self._tt_container)
        self._tt_holder.setContentsMargins(0, 0, 0, 0)
        self._tt_holder.addWidget(Caption("Filter by trade type"))
        self._tt_filter = None

        self._dt_container = QWidget()
        self._dt_holder = QVBoxLayout(self._dt_container)
        self._dt_holder.setContentsMargins(0, 0, 0, 0)
        self._dt_holder.addWidget(Caption("Filter by day type"))
        self._dt_filter = None

        self._news = NewsBreakdownTable()
        self._entry = EntryBreakdownSection()
        self._regime = RegimeSection(settings, self._track_worker)
        self._regime.sourceChanged.connect(self._on_regime_source_changed)
        self._regime.selectionChanged.connect(self._apply_filters)
        self._table = make_table_view(pd.DataFrame(), height=380)

        self._actions_row = TradeActionsRow(settings, self._actions_context,
                                            self._banner)
        actions_holder = QWidget()
        actions_lay = QHBoxLayout(actions_holder)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.addStretch()
        actions_lay.addWidget(self._actions_row)
        actions_lay.addStretch()

        for key, widget in (("trade_type_filter", self._tt_container),
                            ("day_type_filter", self._dt_container),
                            ("news", self._news),
                            ("entry_breakdown", self._entry),
                            ("regime", self._regime),
                            ("trades_table", self._table),
                            ("actions", actions_holder)):
            self._panel.attach_host_section(key, widget)
        self._panel.build_sections()
        lay.addWidget(self._panel)
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
        self._panel.set_section_visible("trade_type_filter", bool(unique_types))
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

        cell_dates = pd.to_datetime(df["date"])
        self._regime.set_context(meta.get("ticker"),
                                 cell_dates.min().strftime("%Y-%m-%d"),
                                 cell_dates.max().strftime("%Y-%m-%d"))
        self._regime_df = self._regime.annotate(df)
        self._apply_filters()

    def hide_detail(self) -> None:
        self.setVisible(False)
        self._cell_df = None
        self._filtered_trades = None

    # ── filters (verbatim backtester ordering) ────────────────────────────────
    def _on_regime_source_changed(self) -> None:
        """Re-join once per regime source change, not per filter toggle."""
        if self._cell_df is None:
            return
        self._regime_df = self._regime.annotate(self._cell_df)
        self._apply_filters()

    def _apply_filters(self) -> None:
        if self._cell_df is None:
            return
        df = getattr(self, "_regime_df", None)
        if df is None or len(df) != len(self._cell_df):
            df = self._cell_df
        self._banner.clear_message()
        all_entries = df          # every entry type; see _entry_frame

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
        self._panel.set_section_visible("news", self._news.set_trades(df))

        selected_day_types = self._dt_filter.selected()
        if not selected_day_types:
            self._banner.show_message("warning", "No day types selected.")
            self._set_report_visible(False)
            return
        df = df[df["day_type"].isin(selected_day_types)].copy()
        all_entries = _entry_frame(all_entries, "day_type", selected_day_types)
        if df.empty:
            self._banner.show_message("info", "No trades match the selected filters.")
            self._set_report_visible(False)
            return
        df["cumulative_ticks"] = df["ticks"].cumsum()

        # regime breakdown before the regime filter (same rule as news)
        regime_states = self._regime.states()
        self._regime.set_trades(df)

        regime_filtered = False
        selected_regimes = self._regime.selected()
        if selected_regimes is not None and FILTER_COLUMN in df.columns:
            if not selected_regimes:
                self._banner.show_message("warning", "No regime states selected.")
                self._set_report_visible(False)
                return
            df = df[df[FILTER_COLUMN].isin(selected_regimes)].copy()
            all_entries = _entry_frame(all_entries, FILTER_COLUMN, selected_regimes)
            if df.empty:
                self._banner.show_message(
                    "info", "No trades match the selected regime states.")
                self._set_report_visible(False)
                return
            df["cumulative_ticks"] = df["ticks"].cumsum()
            regime_filtered = len(selected_regimes) < len(regime_states) + 1

        self._filtered = (trade_type_filtered or regime_filtered
                          or len(selected_day_types) < len(DAY_TYPE_ORDER))
        self._selected_day_types = selected_day_types
        self._filtered_trades = df

        self._panel.set_section_visible(
            "entry_breakdown", self._entry.set_trades(all_entries))

        self._set_report_visible(True)
        self._panel.set_trades(df)

        display_cols = [c for c in ["date", "direction", "entry_time",
                                    "exit_time", "entry_price", "exit_price",
                                    "exit_reason", "ticks", "trade_type",
                                    "day_type", "regime"] if c in df.columns]
        table = df[display_cols].copy()
        table["date"] = pd.to_datetime(table["date"]).dt.date
        update_table_view(self._table, table)

    def _set_report_visible(self, visible: bool) -> None:
        self._panel.set_report_visible(visible)

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
