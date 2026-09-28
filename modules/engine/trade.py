"""
The ONE trade record every strategy produces, and its conversion to the
platform's trades DataFrame.

A strategy's process_day() returns Trade objects; the engine stamps each with
the trading day's date, computes pnl_points and builds the frame column by
column. Column-wise construction measured slightly FASTER than the old
pd.DataFrame(list_of_dicts) (2.8 ms vs 3.1 ms per 300 trades), so the object
costs nothing at optimizer scale.

This class lives in a normally-imported module on purpose: the plugin loader
execs strategy files without sys.modules registration, where a @dataclass
breaks, and optimizer workers must be able to import it by name.
"""

import json
import math
from dataclasses import dataclass

import pandas as pd

# The platform's trade schema, in order. Every run — empty or not — returns
# exactly these columns.
OUTPUT_COLUMNS = [
    "date",
    "direction",
    "trade_type",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "exit_reason",
    "pnl_points",
    "notes",
]

DIRECTIONS = ("long", "short")


@dataclass(slots=True)
class Trade:
    """One round-trip trade. Prices are in POINTS (the instrument's quote
    units), never ticks — the platform converts with ticks_per_point.

    direction    "long" or "short" (lowercase — anything else raises)
    entry_time   tz-aware pd.Timestamp (America/New_York), the fill time
    exit_time    tz-aware pd.Timestamp, the exit fill time
    entry_price  float, fill price in points
    exit_price   float, exit price in points
    exit_reason  short free-text label ("tp", "sl", "eod", ...)
    sl, tp       stop / target price in points; NaN when not applicable
    trade_type   optional label (e.g. which entry pattern fired)
    notes        optional flat dict of JSON-serializable values; stored as
                 json.dumps(notes) in the `notes` column
    """
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    sl: float = math.nan
    tp: float = math.nan
    trade_type: str | None = None
    notes: dict | None = None

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"Trade.direction must be 'long' or 'short', "
                             f"got {self.direction!r}")

    @property
    def pnl_points(self) -> float:
        """exit - entry for a long, entry - exit for a short (the formula
        every strategy used before the engine took it over)."""
        if self.direction == "long":
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price


def trades_to_frame(rows: list) -> pd.DataFrame:
    """rows = [(date, Trade), ...] in output order -> the 12-column frame."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    dates = [d for d, _ in rows]
    trades = [t for _, t in rows]
    return pd.DataFrame({
        "date":        dates,
        "direction":   [t.direction for t in trades],
        "trade_type":  [t.trade_type for t in trades],
        "entry_time":  [t.entry_time for t in trades],
        "exit_time":   [t.exit_time for t in trades],
        "entry_price": [t.entry_price for t in trades],
        "exit_price":  [t.exit_price for t in trades],
        "sl":          [t.sl for t in trades],
        "tp":          [t.tp for t in trades],
        "exit_reason": [t.exit_reason for t in trades],
        "pnl_points":  [t.pnl_points for t in trades],
        "notes":       [None if t.notes is None else json.dumps(t.notes)
                        for t in trades],
    })
