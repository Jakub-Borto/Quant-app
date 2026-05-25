import pandas as pd

PARAMS = {
    "contracts": 1,
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()
    
    size = params["contracts"]
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    trades["trade_pnl"] = trades["ticks"] * dollars_per_tick * size
    trades["equity"] = account_size + trades["trade_pnl"].cumsum()

    return trades