"""
Per-instance sizing pipeline for the Analytics module.

Extracted verbatim from legacy_streamlit/views/analytics.py. A "sizer" is any
.py file in a position_sizing plugin folder except NON_SIZER_MODULES; the
sizer contract is apply(trades, params) -> sized copy with trade_pnl / equity
/ contracts columns.

Change vs the old view: the trades file arrives as a full Path and the sizer
arrives as an already-loaded module (discovery/loading now goes through
modules.common.backend.plugins with the settings' folder list).
"""

from pathlib import Path

import pandas as pd

from .io import load_trades

# Filenames in a position-sizing folder that aren't sizers themselves.
NON_SIZER_MODULES: set[str] = {"__init__", "base"}


def run_instance(trades_path: Path, sizer_module, params: dict) -> pd.DataFrame:
    """
    Load trades, apply the chosen sizer, return the sized DataFrame.

    Non-mutation guarantee: the sizer is contractually required to return a
    copy (see position_sizing/base.py). We also reload from parquet every
    time, so nothing persists between Run clicks — clean slate on each run.
    """
    raw = load_trades(trades_path)
    return sizer_module.apply(raw, params)
