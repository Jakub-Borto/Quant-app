"""
Strategy execution for the Backtester module.

The day loop, reads, caching and the trade frame are the engine's
(modules.engine.run_strategy); this adds the Backtester's derived columns.

Contract reminder (CLAUDE.md): strategies output prices and pnl in POINTS; the
backtester converts to ticks via `ticks = pnl_points * ticks_per_point`.
"""

from modules.engine import RunResult, run_strategy


def run_backtest(strategy, folder_path, start_date, end_date, params: dict,
                 tick_size: float, ticks_per_point: float, *,
                 extra_folders: dict | None = None,
                 on_progress=None) -> RunResult:
    """
    Run `strategy` over the dataset folder (plus its additional-data folders)
    and return the engine's RunResult, whose `trades` carry the derived
    `ticks` / `cumulative_ticks` columns. An empty `trades` frame means the
    strategy produced no trades — the caller decides how to surface that, and
    `warnings` (e.g. days skipped for a missing additional-data file) too.
    """
    result = run_strategy(strategy, folder_path, start_date, end_date, params,
                          tick_size=tick_size, extra_folders=extra_folders,
                          on_progress=on_progress)
    trades = result.trades
    if not trades.empty:
        trades["ticks"]            = trades["pnl_points"] * ticks_per_point
        trades["cumulative_ticks"] = trades["ticks"].cumsum()
    return result
