"""
modules.engine — the strategy engine.

Strategies import what they need from here:

    from modules.engine import Trade, timed

and the app runs them with run_strategy(). The engine owns the day loop, the
column-limited parquet reads with background prefetch, the RAM day cache, the
timing table and the output frame; a strategy only declares its data and
writes prepare_day() / process_day(). STRATEGY_GUIDE.md (repo root) is the
complete reference for strategy authors.

Pure Python, NO Qt — optimizer worker processes import this package.
"""

from .cache import (DEFAULT_BUDGET_GB, GB, DayCache, clear_cache, get_cache,
                    set_budget_gb)
from .day import Day, DayData
from .runner import (ALL_COLUMNS, MAIN_SLOT, EngineError, RunResult,
                     StrategySpec, additional_slots, day_files, run_strategy,
                     strategy_spec)
from .timing import timed
from .trade import OUTPUT_COLUMNS, Trade, trades_to_frame

__all__ = [
    "Trade", "OUTPUT_COLUMNS", "trades_to_frame", "timed",
    "Day", "DayData",
    "run_strategy", "RunResult", "EngineError", "StrategySpec",
    "strategy_spec", "additional_slots", "day_files", "MAIN_SLOT", "ALL_COLUMNS",
    "DayCache", "get_cache", "set_budget_gb", "clear_cache", "GB",
    "DEFAULT_BUDGET_GB",
]
