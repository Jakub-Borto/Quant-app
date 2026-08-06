"""
Optimizer cell drill-down — everything the Backtester shows, for the clicked
heatmap cell's trades.

Only the cell-shaped part lives here: verbatim x/y/slider/half filtering and
the save name (ticker_strategy_dates + the cell's param combination, so Save
Trades writes an identifiable file into the data root's trades/). The report
itself — filter chain, sections, the Analytics / Monte Carlo handoff — is
TradeReportHost, shared with the Combine tab's combined-set report.
"""

import re

import pandas as pd

from modules.common.backend.data_roots import DatasetRef  # noqa: F401 (typing)
from modules.optimizer.backend.heatmap_model import _fmt_axis_value
from modules.optimizer.report_host import TradeReportHost


class CellDetailPanel(TradeReportHost):
    def __init__(self, settings, track_worker=None, parent=None):
        super().__init__(settings, track_worker, title="Cell detail",
                         empty_message="No trades in this cell.", parent=parent)
        self._run_root = None
        self._meta: dict = {}
        self._cell_desc: list[str] = []

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

        self._run_root = run_root
        self._meta = meta
        self._cell_desc = desc

        if not df.empty:
            cell_dates = pd.to_datetime(df["date"])
            self.set_report_context(
                ticker=meta.get("ticker"), tick_size=meta.get("tick_size"),
                ticks_per_point=meta.get("ticks_per_point"),
                dataset=meta.get("dataset", ""), root=run_root,
                regime_start=cell_dates.min().strftime("%Y-%m-%d"),
                regime_end=cell_dates.max().strftime("%Y-%m-%d"))

        # day-type defaults follow the heatmap's day-bucket selection
        heat_defaults = {("high_impact" if b == "other_high_impact" else b)
                         for b in selected_buckets}
        self.set_source(df, header="Cell detail — " + " · ".join(desc),
                        day_bucket_defaults=heat_defaults)

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
