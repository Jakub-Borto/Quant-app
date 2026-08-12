"""
TradeReport — the shared trade report, and the ONE implementation of its
filter chain.

Used by the Backtester, the Optimizer's cell drill-down and the Combine
drill-down. There is nothing to subclass: construct it, hand it a
ReportContext once per selection, then call show_trades() with a frame.

    report = TradeReport(settings, track_worker=self.track_worker)
    report.set_context(ReportContext(ticker="ES", tick_size=0.25, …))
    report.show_trades(trades, save_target=SaveTarget([...]))

═══════════════════════════════════════════════════════════════════════════
 THE CHAIN'S ORDER IS A CONTRACT
═══════════════════════════════════════════════════════════════════════════

    trade-type filter
      -> News table          (deliberately sees PRE-day-filter trades)
      -> day-type filter
      -> regime breakdown    (PRE regime filter, same rule as News)
      -> regime filter
      -> trade-notes query   (LAST, so a condition can reference every
                              derived column, day_type and regime included)
      -> panel.set_trades()

Every narrowing step goes through backend.frame.narrow, which recomputes
cumulative_ticks — the equity curve reads it, and a gapped cumsum draws the
pre-filter path. The two breakdown tables that run BEFORE their own filter do
so on purpose: they stay comparable across the choice they are helping the
user make. Dragging a section in the layout dialog moves a widget; it does
not change which trades that widget was handed (see backend/layout.py).

Two banners, not one: the filter banner is cleared at the top of every chain
run, so a shared banner would wipe the actions row's "Saved to …"
confirmation on the next filter toggle.
"""

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..backend.frame import (KEEP, ReportContext, SaveTarget, apply_mask,
                             canonical_trades, entry_frame, narrow, save_name)
from ..backend.trade_stats import DAY_TYPE_ORDER
from .actions_row import TradeActionsRow
from .entry_section import EntryBreakdownSection
from .filters import CheckboxFilterRow, make_day_type_filter
from .news_section import NewsBreakdownTable
from .notes_table_section import TradeNotesSection
from .panel import TradeReportPanel
from .regime_section import FILTER_COLUMN, RegimeSection
from modules.common.ui.widgets import (Banner, Caption, SectionHeader, hline,
                                       pin_minimum_height)


class TradeReport(QWidget):
    """The whole report as one widget. See the module docstring for the API."""

    def __init__(self, settings, *, track_worker=None, header=None,
                 empty_message="No trades in this selection.", parent=None):
        super().__init__(parent)
        self.settings = settings
        # the report runs no background work of its own; the regime section
        # does, so the window's worker tracker is threaded down to here
        self._track_worker = track_worker or (lambda w: None)
        self._empty_message = empty_message

        self._source: pd.DataFrame | None = None   # canonical, pre-filters
        self._regime_df: pd.DataFrame | None = None  # source + regime columns
        self._ctx: ReportContext | None = None
        self._save: SaveTarget | None = None
        # handoff state for the Save / Go to… row
        self._filtered_trades: pd.DataFrame | None = None
        self._filtered = False
        self._day_types: list = []
        self._trade_types_meta = "all"
        # the notes section is a filter stage AND lives inside the report it
        # filters; the latch makes a re-entrant chain run impossible
        self._in_apply = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        # the report is nested deeper in the Optimizer than in the Backtester
        # (page -> tabs -> tab -> report -> panel -> stack); every link needs
        # pinning or the collapse/expand squeeze reappears at that level
        pin_minimum_height(self)

        # header chrome is opt-in: the Backtester has a window title and wants
        # nothing above the report at all
        self._header = None
        if header is not None:
            lay.addWidget(hline())
            self._header = SectionHeader(header)
            lay.addWidget(self._header)

        self._filter_banner = Banner()
        lay.addWidget(self._filter_banner)

        self.panel = TradeReportPanel(settings)

        # both filter rows are rebuilt per selection, so each lives in a
        # stable container that is registered once
        self._tt_container, self._tt_holder = self._filter_slot(
            "Filter by trade type")
        self._tt_filter = None
        self._tt_types: list = []       # what the row last OFFERED, not chose
        self._dt_container, self._dt_holder = self._filter_slot(
            "Filter by day type")
        # built once, all checked. day_types=KEEP then leaves it alone, which
        # is what makes a selection survive across runs.
        self._dt_filter = make_day_type_filter()
        self._dt_filter.selectionChanged.connect(self._apply_filters)
        self._dt_holder.addWidget(self._dt_filter)

        self._news = NewsBreakdownTable()
        self._entry = EntryBreakdownSection()
        self._regime = RegimeSection(settings, self._track_worker)
        self._regime.sourceChanged.connect(self._on_regime_source_changed)
        self._regime.selectionChanged.connect(self._apply_filters)
        self._notes = TradeNotesSection(settings, self._track_worker)
        self._notes.queryChanged.connect(self._apply_filters)

        # the save banner lives INSIDE the actions section, so a confirmation
        # travels with the buttons wherever the layout dialog puts them
        self._save_banner = Banner()
        self._actions_row = TradeActionsRow(settings, self._actions_context,
                                            self._save_banner)
        actions_holder = QWidget()
        actions_col = QVBoxLayout(actions_holder)
        actions_col.setContentsMargins(0, 0, 0, 0)
        actions_row = QHBoxLayout()
        actions_row.addStretch()
        actions_row.addWidget(self._actions_row)
        actions_row.addStretch()
        actions_col.addLayout(actions_row)
        actions_col.addWidget(self._save_banner)

        for key, widget in (("trade_type_filter", self._tt_container),
                            ("day_type_filter", self._dt_container),
                            ("news", self._news),
                            ("entry_breakdown", self._entry),
                            ("regime", self._regime),
                            ("trades_table", self._notes),
                            ("actions", actions_holder)):
            self.panel.attach_host_section(key, widget)
        self.panel.build_sections()
        lay.addWidget(self.panel)

    def _filter_slot(self, caption: str):
        container = QWidget()
        holder = QVBoxLayout(container)
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addWidget(Caption(caption))
        return container, holder

    # ══ public API ════════════════════════════════════════════════════════════
    def set_context(self, ctx: ReportContext) -> None:
        """
        Asset + data-root context. Destructive for the regime section (it
        rescans and drops the chosen run), so call it once per selection —
        never per filter toggle.
        """
        self._ctx = ctx
        self.panel.set_context(ctx.ticker, ctx.tick_size, ctx.ticks_per_point,
                               candles_folder=ctx.candles_folder,
                               parquet_root=ctx.root / "parquet")
        self._regime.set_context(ctx.ticker, ctx.regime_start, ctx.regime_end)

    def show_trades(self, trades, *, header=None, day_types=KEEP,
                    keep_trade_types=False, save_target=None) -> None:
        """
        Show `trades` (any shape backend.frame.canonical_trades accepts).

        day_types=KEEP        leave the day-type row exactly as the user left
                              it — how a Backtester re-run keeps the choice.
        day_types={"normal"}  rebuild the row with that set checked — how the
                              Optimizer follows the heatmap's bucket selection.
        keep_trade_types      carry the user's trade-type ticks across a
                              re-slice of the SAME selection; a type the
                              previous slice never OFFERED arrives checked, so
                              a member with no trades in one scope does not
                              come back silently filtered out.
        save_target=None      the Save / Go to… buttons become no-ops.
        """
        self._save = save_target
        if header is not None and self._header is not None:
            self._header.setText(header)
        self.setVisible(True)
        self._filter_banner.clear_message()
        # a new selection invalidates the previous "Saved to …" confirmation;
        # a mere filter toggle does NOT, which is why the two banners are separate
        self._save_banner.clear_message()

        if trades is None or len(trades) == 0:
            self._filter_banner.show_message("info", self._empty_message)
            self.panel.set_report_visible(False)
            self._source = self._regime_df = self._filtered_trades = None
            self._notes.set_source(None)
            return

        self._source = canonical_trades(trades)
        self._rebuild_trade_type_filter(keep=keep_trade_types)
        if day_types is not KEEP:
            self._rebuild_day_type_filter(set(day_types))

        self._regime_df = self._regime.annotate(self._source)
        self._notes.set_source(self._regime_df)
        self._apply_filters()

    def clear(self) -> None:
        """Nothing selected any more — hide and forget."""
        self.setVisible(False)
        self._source = self._filtered_trades = None
        self._notes.set_source(None)

    # ── read-back ─────────────────────────────────────────────────────────────
    def filtered_trades(self):
        return self._filtered_trades

    def is_filtered(self) -> bool:
        return self._filtered

    def selected_day_types(self) -> list:
        return list(self._day_types)

    def selected_trade_types(self):
        """The save form: "all", or the list the user narrowed to."""
        return self._trade_types_meta

    # ══ filter rows ═══════════════════════════════════════════════════════════
    def _rebuild_trade_type_filter(self, *, keep: bool) -> None:
        previous = self._tt_filter.selected() if (keep and self._tt_filter) else None
        if self._tt_filter is not None:
            self._tt_filter.deleteLater()
            self._tt_filter = None
        offered, types = self._tt_types, []
        if "trade_type" in self._source.columns:
            types = sorted(self._source["trade_type"].dropna().unique().tolist())
        self._tt_types = types
        self.panel.set_section_visible("trade_type_filter", bool(types))
        if not types:
            return
        # carry over what the user actually decided, but a type the PREVIOUS
        # slice never offered arrives checked — an entry that simply has no
        # trades in one scope must not come back silently filtered out
        checked = None
        if previous is not None:
            checked = set(previous) | (set(types) - set(offered))
        self._tt_filter = CheckboxFilterRow([(t, t) for t in types],
                                            checked_tags=checked or None,
                                            per_row=6)
        self._tt_filter.selectionChanged.connect(self._apply_filters)
        self._tt_holder.addWidget(self._tt_filter)

    def _rebuild_day_type_filter(self, checked: set) -> None:
        if self._dt_filter is not None:
            self._dt_filter.deleteLater()
        self._dt_filter = make_day_type_filter(checked_tags=checked or None)
        self._dt_filter.selectionChanged.connect(self._apply_filters)
        self._dt_holder.addWidget(self._dt_filter)

    # ══ the chain ═════════════════════════════════════════════════════════════
    def _on_regime_source_changed(self) -> None:
        """Re-join once per regime source change, not per filter toggle."""
        if self._source is None:
            return
        self._regime_df = self._regime.annotate(self._source)
        self._notes.set_source(self._regime_df)
        self._apply_filters()

    def _apply_filters(self) -> None:
        if self._source is None or self._in_apply:
            return
        self._in_apply = True
        try:
            self._run_filters()
        finally:
            self._in_apply = False

    def _reject(self, message: str) -> None:
        self._filter_banner.show_message("warning", message)
        self.panel.set_report_visible(False)

    def _run_filters(self) -> None:
        self.panel.set_section_forced_visible("trades_table", False)
        df = self._regime_df
        if df is None or len(df) != len(self._source):
            df = self._source
        self._filter_banner.clear_message()
        all_entries = df          # every entry type; see backend.frame

        # ── trade-type filter ─────────────────────────────────────────────────
        self._trade_types_meta = "all"
        trade_type_filtered = False
        if self._tt_filter is not None:
            offered = sorted(df["trade_type"].dropna().unique().tolist())
            selected = self._tt_filter.selected()
            if not selected:
                return self._reject("No trade types selected.")
            df = narrow(df, df["trade_type"].isin(selected))
            trade_type_filtered = len(selected) < len(offered)
            if trade_type_filtered:
                self._trade_types_meta = selected

        # ── news & holiday breakdown — BEFORE the day-type filter ─────────────
        self.panel.set_section_visible("news", self._news.set_trades(df))

        # ── day-type filter ───────────────────────────────────────────────────
        day_types = self._dt_filter.selected()
        if not day_types:
            return self._reject("No day types selected.")
        df = narrow(df, df["day_type"].isin(day_types))
        all_entries = entry_frame(all_entries, "day_type", day_types)
        if df.empty:
            return self._reject("No trades match the selected filters.")
        day_type_filtered = len(day_types) < len(DAY_TYPE_ORDER)

        # ── regime breakdown — BEFORE the regime filter, so the table always
        #    shows every state (same rule as the news table above) ────────────
        regime_states = self._regime.states()
        self._regime.set_trades(df)

        # ── regime filter ─────────────────────────────────────────────────────
        regime_filtered = False
        selected_regimes = self._regime.selected()
        if selected_regimes is not None and FILTER_COLUMN in df.columns:
            if not selected_regimes:
                return self._reject("No regime states selected.")
            df = narrow(df, df[FILTER_COLUMN].isin(selected_regimes))
            all_entries = entry_frame(all_entries, FILTER_COLUMN,
                                      selected_regimes)
            if df.empty:
                return self._reject("No trades match the selected regime states.")
            regime_filtered = len(selected_regimes) < len(regime_states) + 1

        # ── trade-notes query — LAST, so a condition can reference every
        #    derived column, day_type and regime included ────────────────────
        notes_filtered = False
        notes_mask = self._notes.report_mask(df)
        if notes_mask is not None:
            df = narrow(df, notes_mask)
            all_entries = apply_mask(all_entries,
                                     self._notes.mask_for(all_entries))
            if df.empty:
                self._reject("No trades match the trade-notes query.")
                # keep the section itself on screen — it holds the query the
                # user needs in order to undo it
                self.panel.set_section_forced_visible("trades_table", True)
                self._notes.set_trades(df)
                return
            notes_filtered = self._notes.is_narrowing()

        self._filtered = (day_type_filtered or trade_type_filtered
                          or regime_filtered or notes_filtered)
        self._day_types = day_types
        self._filtered_trades = df

        # entry breakdown: every entry type, under the day/regime/notes slice
        self.panel.set_section_visible("entry_breakdown",
                                       self._entry.set_trades(all_entries))
        self.panel.set_report_visible(True)
        self.panel.set_trades(df)

        # regime_filter is only worth a column of its own when it can disagree
        # with `regime`
        self._notes.set_display_hints(
            show_regime_filter=(self._regime.timings_differ()
                                and FILTER_COLUMN in df.columns))
        self._notes.set_trades(df)

    # ══ the Save / Go to… row ═════════════════════════════════════════════════
    def _actions_context(self) -> dict | None:
        if (self._filtered_trades is None or self._ctx is None
                or self._save is None or not self._ctx.ticker):
            return None
        return {"trades": self._filtered_trades.drop(
                    columns=list(self._save.drop_columns), errors="ignore"),
                "asset": self._ctx.ticker, "root": self._ctx.root,
                "save_name": save_name(self._save.name_parts),
                "filtered": self._filtered,
                "day_types": self._day_types,
                "trade_types": self._trade_types_meta}
