"""
Optimizer cell drill-down — the cell-shaped slicing and the save name, and
nothing else.

The report itself (filter chain, sections, the Analytics / Monte Carlo
handoff) is modules.common.trade_report, shared verbatim with the Combine tab
and the Backtester. This module is Qt-free: it drives a TradeReport that the
Explore tab owns.
"""

import pandas as pd

from modules.common.trade_report import (ReportContext, SaveTarget,
                                         from_optimizer_rows)
from modules.optimizer.backend.heatmap_model import _fmt_axis_value

# the heatmap's bucket tags -> the report's day-type tags
BUCKET_RENAMES = {"other_high_impact": "high_impact"}


def show_cell(report, trades: pd.DataFrame, meta: dict, x_axis, y_axis,
              slider_axes, slider_values, half, split, cell_ij,
              selected_buckets, run_root) -> None:
    """Slice `trades` down to the clicked heatmap cell and show it."""
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

    if not df.empty:
        cell_dates = pd.to_datetime(df["date"])
        report.set_context(ReportContext(
            ticker=meta.get("ticker"), tick_size=meta.get("tick_size"),
            ticks_per_point=meta.get("ticks_per_point"), root=run_root,
            candles_folder=run_root / "parquet" / (meta.get("dataset") or ""),
            regime_start=cell_dates.min().strftime("%Y-%m-%d"),
            regime_end=cell_dates.max().strftime("%Y-%m-%d")))

    report.show_trades(
        from_optimizer_rows(df) if not df.empty else df,
        header="Cell detail — " + " · ".join(desc),
        # day-type defaults follow the heatmap's day-bucket selection
        day_types={BUCKET_RENAMES.get(b, b) for b in selected_buckets},
        # ticker_strategy_dates + the cell's param combination; the ticker
        # MUST stay the first underscore token, asset lookups downstream key
        # off it. day_bucket is this module's column, not trade data.
        save_target=SaveTarget(
            [meta.get("ticker"), meta.get("strategy"), meta.get("start_date"),
             meta.get("end_date"), *desc],
            drop_columns=("day_bucket",)))
