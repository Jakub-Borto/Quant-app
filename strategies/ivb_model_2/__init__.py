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

    folder_path = Path(folder_path)

    files = sorted(folder_path.glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    if not files:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --- resolve the CVD indicators folder (sibling dataset under same type/asset) ---
    # candle folder_path is .../parquet/{type}/{asset}/{dataset}; the indicators live in a
    # folder the user names. Empty param => CVD divergence finder disabled for the whole run.
    cvd_folder_name   = merged_params.get("cvd_indicators_folder", "")
    indicators_folder = folder_path.parent / cvd_folder_name if cvd_folder_name else None

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        if session.index.tz is None:
            continue

        # per-day CVD: matching YYYY-MM-DD.parquet in the indicators folder; any problem
        # (no folder, missing file, no column, bad read) => None => finder disabled this day.
        cvd_raw = None
        if indicators_folder is not None:
            ind_file = indicators_folder / f.name
            if ind_file.exists():
                try:
                    ind_df = pd.read_parquet(ind_file)
                    if "cumulative_delta" in ind_df.columns:
                        cvd_raw = ind_df["cumulative_delta"]
                except Exception:
                    cvd_raw = None

        trade = process_day(session, merged_params, cvd_raw)
        if trade is not None:
            trade["date"] = pd.Timestamp(f.stem).date()
            trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.DataFrame(trades)[OUTPUT_COLUMNS]


__all__ = ["run", "PARAMS", "PARAM_SECTIONS"]
