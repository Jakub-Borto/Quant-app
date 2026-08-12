"""
The Combine tab's drill-down: what a scope switch needs in order to re-slice a
saved combine run, driving a shared TradeReport.

The combiner never persists its merged trades, so the frame is rebuilt on
demand (combine.materialize). Scope switching goes through set_scope, which
re-slices in place and KEEPS the user's filters and regime source — the scope
radios live in the Combine tab itself, so the choice is visible before the
report is ever opened.

Qt-free: it owns no widgets, only the state a re-slice needs.
"""

from modules.common.trade_report import (KEEP, ReportContext, SaveTarget,
                                         from_optimizer_rows)
from modules.optimizer.backend.combine.materialize import (SCOPE_LABELS,
                                                           VID_COLUMN)


class CombineReportSource:
    """Drives a TradeReport for one combine run, across data scopes."""

    def __init__(self, report):
        self._report = report
        self._resolve = None            # callable(scope) -> DataFrame
        self._active = False            # a set is loaded (NOT Qt visibility:
                                        # a background tab reads as hidden)
        self._scope = "all"
        self._header_stem = ""
        self._save_stem: list = []
        self._day_bucket_defaults: set = set()

    # ── entry point from the Combine tab ──────────────────────────────────────
    def show_set(self, *, resolve, scope, header_stem, save_stem, ticker,
                 tick_size, ticks_per_point, dataset, root, regime_start,
                 regime_end, day_bucket_defaults) -> None:
        self._resolve = resolve
        self._header_stem = header_stem
        self._save_stem = list(save_stem)
        self._day_bucket_defaults = set(day_bucket_defaults)
        self._active = True
        self._report.set_context(ReportContext(
            ticker=ticker, tick_size=tick_size, ticks_per_point=ticks_per_point,
            root=root, candles_folder=root / "parquet" / (dataset or ""),
            regime_start=regime_start, regime_end=regime_end))
        self._load(scope, day_types=self._day_bucket_defaults,
                   keep_trade_types=False)

    def set_scope(self, scope: str) -> None:
        """Live re-slice. Deliberately does NOT redo set_context — that would
        rescan the regime section and drop the chosen run."""
        if self._resolve is None or not self._active:
            return
        self._load(scope, day_types=KEEP, keep_trade_types=True)

    def clear(self) -> None:
        self._active = False
        self._report.clear()

    # ── internals ─────────────────────────────────────────────────────────────
    def _load(self, scope: str, *, day_types, keep_trade_types: bool) -> None:
        self._scope = scope
        df = self._resolve(scope)
        # ticker_combinerun_k_scope — the ticker MUST stay the first
        # underscore token, asset lookups downstream key off it.
        # combine_vid is this module's provenance column, not trade data.
        self._report.show_trades(
            from_optimizer_rows(df) if df is not None and len(df) else df,
            header=f"{self._header_stem} · {SCOPE_LABELS[scope]}",
            day_types=day_types, keep_trade_types=keep_trade_types,
            save_target=SaveTarget([*self._save_stem, scope],
                                   drop_columns=("day_bucket", VID_COLUMN)))
