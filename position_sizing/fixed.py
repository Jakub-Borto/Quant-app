import pandas as pd

PARAMS = {
    "contracts": 1.0,  # float => decimal input; allows mini contracts (e.g. 1.5)
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()

    size = round(params["contracts"], 1)  # carry one decimal (mini contracts)
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    trades["trade_pnl"] = trades["ticks"] * dollars_per_tick * size
    trades["equity"] = account_size + trades["trade_pnl"].cumsum()
    trades["contracts"] = size  # per-trade size (broadcast); shown on hover

    return trades