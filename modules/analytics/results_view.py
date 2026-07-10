"""
Analytics results — per-instance 4-curve equity chart + metric tile grid,
the combined overlay chart, and the metrics comparison table.

The enrichment (cost curves) is recomputed here from the sized runs whenever
account size / slippage / stats-curve change — WITHOUT re-running the sizers,
exactly like the old render-time enrich_run. Curve colors/widths/dashes are
the old _CURVE_STYLE; the stats tile grid and comparison table are driven by
METRIC_REGISTRY.
"""

import pandas as pd
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from modules.analytics.backend.costs import enrich_run
from modules.analytics.backend.io import read_filter_metadata
from modules.analytics.backend.metrics import (DAY_TYPE_LABELS,
                                               METRIC_REGISTRY, _curve_frame,
                                               compute_metric_values)
from modules.common.ui.charts.equity_curve import MultiLineEquityChart
from modules.common.ui.dataframe_model import make_table_view
from modules.common.ui.widgets import Banner, Caption, MetricTile, SectionHeader

# Distinct styling so the gross-vs-net cost drag reads at a glance
# (verbatim colors from the old _CURVE_STYLE).
_CURVE_STYLE = {
    "Gross":         dict(width=2,   color="#1f77b4", style="solid"),
    "+ Commissions": dict(width=1.5, color="#ff7f0e", style="dot"),
    "+ Slippage":    dict(width=1.5, color="#9467bd", style="dash"),
    "+ Both (net)":  dict(width=2.5, color="#2ca02c", style="solid"),
}

_SLIPPAGE_CAPTION = (
    "Slippage is a first-order post-hoc deduction on the recorded trades: it "
    "cannot model that a worse entry might have prevented a TP from filling at "
    "all. Read it as a cost overlay / lower bound on damage, not a "
    "re-simulation — true path-dependent slippage lives in the backtester."
)

_KEYS = [k for k, _l, _f in METRIC_REGISTRY]
_LABELS = {k: l for k, l, _f in METRIC_REGISTRY}
_FMT = {k: f for k, _l, f in METRIC_REGISTRY}


def _filter_caption_text(trades_path) -> str | None:
    """Filtered-file warning (verbatim wording) or None."""
    try:
        meta = read_filter_metadata(trades_path)
    except Exception:  # noqa: BLE001 — unreadable metadata is not fatal
        return None
    if meta is None:
        return None
    day_labels = ", ".join(DAY_TYPE_LABELS.get(k, k) for k in meta["day_types"]) or "—"
    if meta["trade_types"] == "all":
        tt_labels = "all"
    else:
        tt_labels = ", ".join(str(t) for t in meta["trade_types"]) or "—"
    return (f"⚠ Filtered file: day types = {day_labels}; trade types = {tt_labels}. "
            "This is a sub-strategy slice — equity is stitched across excluded trades, "
            "so drawdown durations are compressed; do not read it as the live timeline.")


def _metric_grid(values: dict) -> QWidget:
    """Tile grid of every registry metric (rows of 5, registry order)."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(8)
    for n, key in enumerate(_KEYS):
        grid.addWidget(MetricTile(_LABELS[key], _FMT[key](values[key])),
                       n // 5, n % 5)
    return box


class ResultsView(QWidget):
    """Rebuilt wholesale by refresh() — mirrors the old render_results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(10)
        self.setVisible(False)

    def clear(self) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.setVisible(False)

    def refresh(self, runs: list[dict], account_size: float, slippage_n: int,
                stats_curve: str) -> None:
        self.clear()
        if not runs:
            return
        lay = self._lay
        enriched = [enrich_run(r, account_size, slippage_n) for r in runs]

        # cost-model warnings (missing commission, fractional non-micro, ...)
        for run in enriched:
            for w in run["warnings"]:
                lay.addWidget(Banner("warning", w))

        # ── individual equity curves ──────────────────────────────────────────
        lay.addWidget(SectionHeader("Individual equity curves"))
        for run in enriched:
            lay.addWidget(SectionHeader(run["label"]))
            caption = _filter_caption_text(run["trades_path"])
            if caption:
                lay.addWidget(Banner("warning", caption))

            values = compute_metric_values(_curve_frame(run, stats_curve),
                                           run["gross_total"])
            trades = run["trades"]
            if trades.empty:
                lay.addWidget(Banner("info", "No trades to display for this instance."))
                lay.addWidget(_metric_grid(values))   # safe zeros, keeps shape
                continue

            chart = MultiLineEquityChart(height=340)
            x = trades["entry_time"] if "entry_time" in trades.columns else trades.index
            has_contracts = "contracts" in trades.columns
            for label, (_pnl, equity) in run["curves"].items():
                extra = None
                if has_contracts and label == "Gross":   # contracts identical across curves; show once
                    extra = [f"Contracts: {c:.1f}" for c in trades["contracts"]]
                chart.add_series(label, x, equity, hover_extra=extra,
                                 **_CURVE_STYLE.get(label, dict(width=2, color="#cccccc", style="solid")))
            chart.add_start_line(account_size)
            chart.finish()
            lay.addWidget(chart)
            lay.addWidget(Caption(_SLIPPAGE_CAPTION))
            lay.addWidget(Caption(f"Statistics below are computed on the "
                                  f"{stats_curve} curve."))
            lay.addWidget(_metric_grid(values))

        # ── combined overlay (selected curve, one trace per non-empty run) ───
        non_empty = [r for r in enriched if not r["trades"].empty]
        lay.addWidget(SectionHeader(f"Combined equity curve ({stats_curve})"))
        if not non_empty:
            lay.addWidget(Banner("info", "No non-empty runs to combine."))
        else:
            combined = MultiLineEquityChart(height=340)
            for run in non_empty:
                frame = _curve_frame(run, stats_curve)
                combined.add_series(run["label"], frame["entry_time"],
                                    frame["equity"], color=None, width=2)
            combined.add_start_line(account_size)
            combined.finish()
            lay.addWidget(combined)

        # ── comparison table ──────────────────────────────────────────────────
        lay.addWidget(SectionHeader("Metrics comparison"))
        lay.addWidget(Caption(f"Computed on the {stats_curve} curve."))
        rows = []
        for run in enriched:
            vals = compute_metric_values(_curve_frame(run, stats_curve),
                                         run["gross_total"])
            row = {"Label": run["label"]}
            for key, label, _f in METRIC_REGISTRY:
                row[label] = vals[key]
            rows.append(row)
        lay.addWidget(make_table_view(pd.DataFrame(rows),
                                      height=min(90 + 30 * len(rows), 420)))
        self.setVisible(True)
