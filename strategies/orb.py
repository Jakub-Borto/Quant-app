# strategies/orb.py
import pandas as pd
from strategies.base import is_bullish, is_bearish
from pathlib import Path

PARAMS = {
    "range_minutes": 15,
    "sl_factor": 0.5,
    "rr": 2.0,
    "timeout_minutes": 240,
}

RTH_OPEN  = "09:30"
RTH_CLOSE = "16:00"


def run(folder_path: Path, start_date: pd.Timestamp,
        end_date: pd.Timestamp, params: dict = None) -> pd.DataFrame:
    p = {**PARAMS, **(params or {})}

    files = sorted(Path(folder_path).glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        trade = _process_session(session, pd.Timestamp(f.stem).date(), p)
        if trade:
            trades.append(trade)

    return pd.DataFrame(trades)


def _process_session(session: pd.DataFrame, rth_date, p: dict) -> dict:
    rth_start = pd.Timestamp(f"{rth_date} {RTH_OPEN}",  tz=session.index.tz)
    rth_end   = pd.Timestamp(f"{rth_date} {RTH_CLOSE}", tz=session.index.tz)

    rth = session.loc[rth_start:rth_end]

    if len(rth) < 2:
        return None

    range_end = rth_start + pd.Timedelta(minutes=p["range_minutes"])

    opening_range = rth[rth.index < range_end]
    after_range   = rth[rth.index >= range_end]

    if opening_range.empty or after_range.empty:
        return None

    orb_high  = opening_range["high"].max()
    orb_low   = opening_range["low"].min()
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return None

    signal_idx = None
    direction  = None

    for i in range(len(after_range) - 1):
        candle = after_range.iloc[i]

        if candle["close"] > orb_high and is_bullish(candle):
            signal_idx = i
            direction  = "long"
            break

        if candle["close"] < orb_low and is_bearish(candle):
            signal_idx = i
            direction  = "short"
            break

    if signal_idx is None:
        return None

    entry_candle = after_range.iloc[signal_idx + 1]
    entry_price  = entry_candle["open"]
    entry_time   = entry_candle.name

    sl_distance = orb_range * p["sl_factor"]

    if direction == "long":
        sl = entry_price - sl_distance
        tp = entry_price + sl_distance * p["rr"]
    else:
        sl = entry_price + sl_distance
        tp = entry_price - sl_distance * p["rr"]

    timeout_time   = entry_time + pd.Timedelta(minutes=p["timeout_minutes"])
    trade_candles  = after_range.iloc[signal_idx + 1:]
    exit_price     = None
    exit_time      = None
    exit_reason    = None
    tp_moved_to_be = False

    for i, (ts, candle) in enumerate(trade_candles.iterrows()):
        if ts >= timeout_time and not tp_moved_to_be:
            in_profit = (candle["close"] > entry_price if direction == "long"
                         else candle["close"] < entry_price)
            if in_profit:
                exit_price, exit_time, exit_reason = candle["close"], ts, "timeout_profit"
                break
            else:
                tp             = entry_price
                tp_moved_to_be = True

        if direction == "long":
            if candle["low"] <= sl:
                exit_price, exit_time, exit_reason = sl, ts, "sl"
                break
            if candle["high"] >= tp:
                exit_price, exit_time, exit_reason = tp, ts, "tp"
                break
        else:
            if candle["high"] >= sl:
                exit_price, exit_time, exit_reason = sl, ts, "sl"
                break
            if candle["low"] <= tp:
                exit_price, exit_time, exit_reason = tp, ts, "tp"
                break

        if i == len(trade_candles) - 1:
            exit_price, exit_time, exit_reason = candle["close"], ts, "eod"

    if exit_price is None:
        return None

    pnl_points = exit_price - entry_price if direction == "long" else entry_price - exit_price

    return {
        "date":        str(rth_date),
        "direction":   direction,
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "sl":          sl,
        "tp":          tp,
        "exit_reason": exit_reason,
        "pnl_points":  pnl_points,
    }