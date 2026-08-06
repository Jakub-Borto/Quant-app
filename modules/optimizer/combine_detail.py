"""
The Combine tab's drill-down: the full trade report for one path point of a
saved combine run, at one data scope.

The combiner never persists its merged trades, so the frame is rebuilt on
demand (combine.materialize) and handed to the shared TradeReportHost. Scope
switching goes through set_scope, which re-slices in place and KEEPS the
user's filters and regime source — the scope radios live in the Combine tab
itself, so the choice is visible before the report is ever opened.
"""

import re

from modules.optimizer.backend.combine.materialize import (SCOPE_LABELS,
                                                           VID_COLUMN)
from modules.optimizer.report_host import TradeReportHost


class CombineDetailPanel(TradeReportHost):
    def __init__(self, settings, track_worker=None, parent=None):
        super().__init__(
            settings, track_worker, title="Combined trade report",
            empty_message="No trades in this slice of the combined set.",
            parent=parent)
        self._resolve = None            # callable(scope) -> DataFrame
        self._active = False            # a set is loaded (NOT Qt visibility:
                                        # a background tab reads as hidden)
        self._scope = "all"
        self._header_stem = ""
        self._save_stem: list = []
        self._root = None
        self._day_bucket_defaults: set = set()

    # ── entry point from the Combine tab ──────────────────────────────────────
    def show_set(self, *, resolve, scope, header_stem, save_stem, ticker,
                 tick_size, ticks_per_point, dataset, root, regime_start,
                 regime_end, day_bucket_defaults) -> None:
        self._resolve = resolve
        self._header_stem = header_stem
        self._save_stem = list(save_stem)
        self._root = root
        self._day_bucket_defaults = set(day_bucket_defaults)
        self._active = True
        self.set_report_context(
            ticker=ticker, tick_size=tick_size,
            ticks_per_point=ticks_per_point, dataset=dataset, root=root,
            regime_start=regime_start, regime_end=regime_end)
        self._load(scope, preserve_filters=False)

    def set_scope(self, scope: str) -> None:
        """Live re-slice. Deliberately does NOT re-run set_report_context —
        that would rescan the regime section and drop the chosen run."""
        if self._resolve is None or not self._active:
            return
        self._load(scope, preserve_filters=True)

    def hide_detail(self) -> None:
        self._active = False
        super().hide_detail()

    def _load(self, scope: str, *, preserve_filters: bool) -> None:
        self._scope = scope
        self.set_source(self._resolve(scope),
                        header=f"{self._header_stem} · {SCOPE_LABELS[scope]}",
                        day_bucket_defaults=self._day_bucket_defaults,
                        preserve_filters=preserve_filters)

    def _actions_context(self) -> dict | None:
        if (self._filtered_trades is None or self._root is None
                or not self._asset):
            return None
        # day_bucket is derived (like day_type, which the shared row strips);
        # combine_vid is this module's provenance column, not trade data
        trades = self._filtered_trades.drop(
            columns=["day_bucket", VID_COLUMN], errors="ignore")
        # ticker_combinerun_k_scope — the ticker MUST stay the first
        # underscore token, asset lookups downstream key off it
        save_name = "_".join(
            re.sub(r"[^A-Za-z0-9._\-]+", "-", str(p)).strip("-")
            for p in [*self._save_stem, self._scope] if p)
        return {"trades": trades, "asset": self._asset,
                "root": self._root, "save_name": save_name,
                "filtered": self._filtered,
                "day_types": self._selected_day_types,
                "trade_types": self._selected_trade_types_meta}
