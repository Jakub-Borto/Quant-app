"""
Strategy execution for the Backtester module.

Extracted verbatim from legacy_streamlit/views/backtester.py::execute_run —
the tick_size injection, the strategy.run() call and the pnl_points -> ticks
conversion are the load-bearing logic; date validation and empty-result
messaging stay in the window (they were st.error/st.warning calls).

Contract reminder (CLAUDE.md): strategies output `pnl_points` only; the
backtester converts to ticks via `ticks = pnl_points * ticks_per_point`.
"""

import pandas as pd


def run_backtest(strategy, folder_path, start_date, end_date, params: dict,
                 tick_size: float, ticks_per_point: float) -> pd.DataFrame:
    """
    Run `strategy` over the dataset folder and return its trades with the
    derived `ticks` / `cumulative_ticks` columns appended.

    `params` is mutated with the injected tick_size (same as the old view).
    An empty DataFrame means the strategy produced no trades — the caller
    decides how to surface that.
    """
    params["tick_size"] = tick_size

    trades = strategy.run(
        folder_path=folder_path,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        params=params,
    )

    if trades.empty:
        return trades

    trades["ticks"]            = trades["pnl_points"] * ticks_per_point
    trades["cumulative_ticks"] = trades["ticks"].cumsum()
    return trades
