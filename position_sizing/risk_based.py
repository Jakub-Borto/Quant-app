# position_sizing/risk_based.py
import math
import pandas as pd

PARAMS = {
    "risk_pct": 0.01,
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()

    risk_pct = params["risk_pct"]
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    # ticks_per_point is an instrument constant — derive it once from the file
    # rather than requiring it as a param, since it's already encoded in ticks/pnl_points
    nonzero = trades[trades["pnl_points"] != 0]
    if nonzero.empty:
        trades["trade_pnl"] = 0.0
        trades["equity"] = account_size
        return trades
    ticks_per_point = (nonzero["ticks"] / nonzero["pnl_points"]).iloc[0]

    trade_pnl_list = []
    skipped = 0
    equity = account_size

    for _, trade in trades.iterrows():
        sl_ticks = abs(trade["entry_price"] - trade["sl"]) * ticks_per_point
        sl_dollars = sl_ticks * dollars_per_tick

        if sl_dollars > 0:
            size = math.floor((equity * risk_pct) / sl_dollars)
        else:
            size = 0

        if size == 0:
            skipped += 1

        pnl = trade["ticks"] * dollars_per_tick * size
        trade_pnl_list.append(pnl)
        equity += pnl

    trades["trade_pnl"] = trade_pnl_list
    trades["equity"] = account_size + pd.Series(trade_pnl_list).cumsum().values
    trades["skipped"] = trades["trade_pnl"] == 0  # flag for UI warning

    # attach skipped count as metadata for the analytics UI to surface
    trades.attrs["skipped_trades"] = skipped

    return trades