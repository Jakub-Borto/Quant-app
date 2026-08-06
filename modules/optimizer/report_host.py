"""
The optimizer's drill-down report host — everything the Backtester shows, for
whatever slice of trades a subclass hands it.

Extracted from CellDetailPanel so the Explore tab's cell drill-down and the
Combine tab's combined-set report run the SAME filter chain: trade-type filter
-> news/holiday table -> day-type filter -> shared TradeReportPanel -> regime
filter -> trades table -> the shared TradeActionsRow. The chain's ordering is
a contract (the news and regime breakdowns are computed BEFORE their own
filters, so they stay comparable); one copy of it is the point of this file.

Subclasses supply two things: a frame (via set_source) and a save context
(via _actions_context).
"""

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from modules.common.backend.trade_stats import DAY_TYPE_ORDER
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.trade_report.actions_row import TradeActionsRow
from modules.common.ui.trade_report.entry_section import EntryBreakdownSection
from modules.common.ui.trade_report.filters import (CheckboxFilterRow,
                                                    make_day_type_filter)
from modules.common.ui.trade_report.news_section import NewsBreakdownTable
from modules.common.ui.trade_report.panel import TradeReportPanel
from modules.common.ui.trade_report.regime_section import (FILTER_COLUMN,
                                                           RegimeSection)
from modules.common.ui.widgets import (Banner, Caption, SectionHeader, hline,
                                       pin_minimum_height)


def _entry_frame(frame, column: str, selected):
    """One filter applied to the entry-breakdown frame — everything the report
    shows EXCEPT the trade-type filter, so entry types stay comparable."""
    if selected is None or column not in frame.columns:
        return frame
    return frame[frame[column].isin(selected)]


class TradeReportHost(QWidget):
    def __init__(self, settings, track_worker=None, title="Detail",
                 empty_message="No trades in this selection.", parent=None):
        super().__init__(parent)
        self._settings = settings
        # the host tabs run no background work of their own; regime loading
        # needs one, so the window's tracker is threaded down to here
        self._track_worker = track_worker or (lambda w: None)
        self._empty_message = empty_message
        self._source_df: pd.DataFrame | None = None  # pre type/day filters
        self._regime_df: pd.DataFrame | None = None  # source + regime columns
        # handoff context for the Go to Analytics / Monte Carlo row
        self._filtered_trades: pd.DataFrame | None = None
        self._filtered = False
        self._selected_day_types: list = []
        self._selected_trade_types_meta = "all"
        self._asset = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        # the report is nested deeper here than in the Backtester
        # (page -> tabs -> HostTab -> this panel -> panel -> stack);
        # every link needs pinning or the squeeze reappears at that level
        pin_minimum_height(self)
        lay.addWidget(hline())
        self._header = SectionHeader(title)
        lay.addWidget(self._header)
        self._banner = Banner()
        lay.addWidget(self._banner)
        # subclasses hang their own controls between the header and the report
        self.above_panel = QVBoxLayout()
        lay.addLayout(self.above_panel)

        self._panel = TradeReportPanel(settings)

        # both filter rows are rebuilt per selection, so each lives in a stable
        # container that gets registered once
        self._tt_container = QWidget()
        self._tt_holder = QVBoxLayout(self._tt_container)
        self._tt_holder.setContentsMargins(0, 0, 0, 0)
        self._tt_holder.addWidget(Caption("Filter by trade type"))
        self._tt_filter = None
        self._tt_types: list = []       # what the row last OFFERED, not chose

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

    # ── what subclasses call ──────────────────────────────────────────────────
    def set_report_context(self, *, ticker, tick_size, ticks_per_point,
                           dataset, root, regime_start, regime_end) -> None:
        """
        Asset/candle context for the report and the regime picker. Destructive
        for the regime section (it rescans and drops the chosen run), so call
        it once per selection — never per filter toggle.
        """
        self._asset = ticker
        self._panel.set_context(
            ticker, tick_size, ticks_per_point,
            candles_folder=root / "parquet" / (dataset or ""),
            parquet_root=root / "parquet")
        self._regime.set_context(ticker, regime_start, regime_end)

    def set_source(self, df: pd.DataFrame, *, header: str,
                   day_bucket_defaults, preserve_filters: bool = False) -> None:
        """
        Show `df` (raw optimizer-run rows) in the report. `preserve_filters`
        keeps the user's checkbox selections across a re-slice of the SAME
        selection — a scope switch, not a new selection.
        """
        self._header.setText(header)
        self.setVisible(True)
        self._banner.clear_message()

        if df is None or df.empty:
            self._banner.show_message("info", self._empty_message)
            self._set_report_visible(False)
            self._source_df = None
            self._regime_df = None
            self._filtered_trades = None
            return

        # backtester-shaped columns: ticks alias + historical day_type names
        df = df.sort_values("entry_time").reset_index(drop=True)
        df["ticks"] = df["pnl_ticks"]
        df["day_type"] = df["day_bucket"].replace(
            {"other_high_impact": "high_impact"})
        self._source_df = df

        prev_tt = self._tt_filter.selected() \
            if preserve_filters and self._tt_filter is not None else None
        prev_dt = self._dt_filter.selected() \
            if preserve_filters and self._dt_filter is not None else None

        # (re)build the filter rows for this selection
        if self._tt_filter is not None:
            self._tt_filter.deleteLater()
            self._tt_filter = None
        prev_types, unique_types = self._tt_types, []
        if "trade_type" in df.columns:
            unique_types = sorted(df["trade_type"].dropna().unique().tolist())
        self._tt_types = unique_types
        self._panel.set_section_visible("trade_type_filter", bool(unique_types))
        if unique_types:
            # carry over what the user actually decided, but a type the PREVIOUS
            # slice never offered arrives checked — an entry that simply has no
            # trades in one scope must not come back silently filtered out
            keep = None
            if prev_tt is not None:
                keep = set(prev_tt) | (set(unique_types) - set(prev_types))
            self._tt_filter = CheckboxFilterRow(
                [(t, t) for t in unique_types], checked_tags=keep or None,
                per_row=6)
            self._tt_filter.selectionChanged.connect(self._apply_filters)
            self._tt_holder.addWidget(self._tt_filter)

        if self._dt_filter is not None:
            self._dt_filter.deleteLater()
        checked = set(prev_dt) if prev_dt else set(day_bucket_defaults)
        self._dt_filter = make_day_type_filter(checked_tags=checked or None)
        self._dt_filter.selectionChanged.connect(self._apply_filters)
        self._dt_holder.addWidget(self._dt_filter)

        self._regime_df = self._regime.annotate(df)
        self._apply_filters()

    def hide_detail(self) -> None:
        self.setVisible(False)
        self._source_df = None
        self._filtered_trades = None

    # ── filters (verbatim backtester ordering) ────────────────────────────────
    def _on_regime_source_changed(self) -> None:
        """Re-join once per regime source change, not per filter toggle."""
        if self._source_df is None:
            return
        self._regime_df = self._regime.annotate(self._source_df)
        self._apply_filters()

    def _apply_filters(self) -> None:
        if self._source_df is None:
            return
        df = self._regime_df
        if df is None or len(df) != len(self._source_df):
            df = self._source_df
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

    # ── subclass contract ─────────────────────────────────────────────────────
    def _actions_context(self) -> dict | None:
        """{trades, asset, root, save_name, filtered, day_types, trade_types}
        for the shared TradeActionsRow, or None to make its buttons no-ops."""
        raise NotImplementedError
