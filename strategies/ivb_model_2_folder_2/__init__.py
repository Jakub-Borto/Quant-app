"""
IVB Model 2 — modular package.

The backtester loads this package via __init__.py and expects:
  - run(folder_path, start_date, end_date, params) -> pd.DataFrame
  - PARAMS, PARAM_SECTIONS

Internal layout:
  params.py      PARAMS, PARAM_SECTIONS, OUTPUT_COLUMNS
  profile.py     compute_ivb_profile
  baselines.py   rolling / passive / two-bar baselines + tick volume merge
  absorption.py  shared absorption grading + trigger extraction
  entries/       one module per entry type, registered in FINDER_REGISTRY
  risk/          compute_sl_tp, run_trade (+ trailing placeholder)
  core.py        breakout/retest detection, entry dispatcher, process_day
"""

from pathlib import Path
import pandas as pd

from .params import PARAMS, PARAM_SECTIONS, OUTPUT_COLUMNS
from .core   import process_day


def run(
    folder_path: Path,
    start_date:  pd.Timestamp,
    end_date:    pd.Timestamp,
    params:      dict | None = None,
) -> pd.DataFrame:
    merged_params = {**PARAMS, **(params or {})}

    files = sorted(Path(folder_path).glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    if not files:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        if session.index.tz is None:
            continue
        trade = process_day(session, merged_params)
        if trade is not None:
            trade["date"] = pd.Timestamp(f.stem).date()
            trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.DataFrame(trades)[OUTPUT_COLUMNS]


__all__ = ["run", "PARAMS", "PARAM_SECTIONS"]
