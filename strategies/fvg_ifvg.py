# strategies/fvg_ifvg.py
import pandas as pd
from pathlib import Path
from strategies.base import is_bullish, is_bearish

PARAMS = {
    "15m_fvg_timeout" : 60,
    "fvg_timeout_minutes": 60,
    "eod_timeout_minutes": 180,
}

# ---------------------------------------------------------------------------
# 15m helpers
# ---------------------------------------------------------------------------

def _build_15m(session: pd.DataFrame) -> pd.DataFrame:
    return (
        session.resample("15min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open"])
    )


def _find_first_15m_fvg(candles_15m: pd.DataFrame) -> tuple | None:
    if len(candles_15m) < 3:
        return None

    highs = candles_15m["high"].to_numpy()
    lows = candles_15m["low"].to_numpy()
    times = candles_15m.index

    for i in range(2, len(candles_15m)):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        if c3_low > c1_high:
            return ("bullish", float(c3_low), float(c1_high), times[i] + pd.Timedelta(minutes=15))
        if c3_high < c1_low:
            return ("bearish", float(c1_low), float(c3_high), times[i] + pd.Timedelta(minutes=15))

    return None


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _fvg_invalidated(fvg_type: str, fvg_high: float, fvg_low: float, candle: pd.Series) -> bool:
    body_min = min(candle["open"], candle["close"])
    body_max = max(candle["open"], candle["close"])
    if fvg_type == "bullish":
        return body_min < fvg_low
    return body_max > fvg_high


def _price_enters_fvg(fvg_type: str, fvg_high: float, fvg_low: float, candle: pd.Series) -> bool:
    if fvg_type == "bullish":
        return candle["low"] <= fvg_high and candle["close"] > fvg_low
    return candle["high"] >= fvg_low and candle["close"] < fvg_high


def _find_latest_1m_fvg(candles_so_far: pd.DataFrame, fvg_type: str) -> tuple | None:
    n = len(candles_so_far)
    if n < 3:
        return None

    for i in range(n - 3, -1, -1):
        c1 = candles_so_far.iloc[i]
        c2 = candles_so_far.iloc[i + 1]
        c3 = candles_so_far.iloc[i + 2]

        if fvg_type == "bullish" and c3["low"] > c1["high"] and is_bullish(c2):
            return (c3["low"], c1["high"])
        if fvg_type == "bearish" and c3["high"] < c1["low"] and is_bearish(c2):
            return (c1["low"], c3["high"])

    return None


# ---------------------------------------------------------------------------
# Phase 1 — wait for price to enter 15m FVG zone
# ---------------------------------------------------------------------------

def _phase_1(session: pd.DataFrame, fvg: dict, fvg_cutoff: pd.Timestamp) -> tuple[bool, pd.Timestamp | None]:
    post_fvg = session[session.index > fvg["confirmed_time"]]

    for ts, candle in post_fvg.iterrows():
        if ts > fvg_cutoff:
            return False, None
        if _fvg_invalidated(fvg["type"], fvg["high"], fvg["low"], candle):
            return False, None
        if _price_enters_fvg(fvg["type"], fvg["high"], fvg["low"], candle):
            return True, ts

    return False, None


# ---------------------------------------------------------------------------
# Phase 2 — find 1m IFVG inside zone and enter
# ---------------------------------------------------------------------------

def _phase_2(session: pd.DataFrame, fvg: dict, time_entered: pd.Timestamp, eod_cutoff: pd.Timestamp) -> dict | None:
    fvg_type = fvg["type"]
    fvg_high = fvg["high"]
    fvg_low  = fvg["low"]

    # seed candles_list with all 1m candles from fvg confirmed to zone entry
    seed = session[
        (session.index > fvg["confirmed_time"]) &
        (session.index <= time_entered)
    ]
    candles_list = [row for _, row in seed.iterrows()]

    post_entry = session[
        (session.index > time_entered) &
        (session.index <= eod_cutoff)
    ]

    for ts, candle in post_entry.iterrows():
        candles_list.append(candle)

        # invalidation still applies inside the zone
        if _fvg_invalidated(fvg_type, fvg_high, fvg_low, candle):
            return None

        # look for IFVG only on candles in the right direction
        if fvg_type == "bullish" and not is_bullish(candle):
            continue
        if fvg_type == "bearish" and not is_bearish(candle):
            continue

        df_so_far = pd.DataFrame(candles_list)
        # for bullish 15m FVG → look for bearish 1m FVG being run through upward
        # for bearish 15m FVG → look for bullish 1m FVG being run through downward
        search_type = "bearish" if fvg_type == "bullish" else "bullish"
        result = _find_latest_1m_fvg(df_so_far, search_type)

        if result is None:
            continue

        ifvg_high, ifvg_low = result

        if fvg_type == "bullish" and candle["close"] > ifvg_high:
            return _build_trade(session, fvg, candles_list, ts, candle, "long", eod_cutoff)
        if fvg_type == "bearish" and candle["close"] < ifvg_low:
            return _build_trade(session, fvg, candles_list, ts, candle, "short", eod_cutoff)

    return None


# ---------------------------------------------------------------------------
# Trade builder + exit
# ---------------------------------------------------------------------------

def _build_trade(
    session: pd.DataFrame,
    fvg: dict,
    candles_list: list,
    entry_time: pd.Timestamp,
    entry_candle: pd.Series,
    direction: str,
    eod_cutoff: pd.Timestamp,
) -> dict | None:
    entry_price = float(entry_candle["close"])
    candles_df  = pd.DataFrame(candles_list)

    if direction == "long":
        sl = float(candles_df["low"].min())
        tp = float(session.loc[session.index <= entry_time, "high"].max())
    else:
        sl = float(candles_df["high"].max())
        tp = float(session.loc[session.index <= entry_time, "low"].min())

    # sanity check
    if direction == "long" and tp <= entry_price:
        return None
    if direction == "short" and tp >= entry_price:
        return None

    # vectorized exit
    post = session[session.index > entry_time]

    if direction == "long":
        tp_hit = post[post["high"] >= tp].index.min()
        sl_hit = post[post["low"]  <= sl].index.min()
    else:
        tp_hit = post[post["low"]  <= tp].index.min()
        sl_hit = post[post["high"] >= sl].index.min()

    if pd.isna(tp_hit) and pd.isna(sl_hit):
        exit_price  = float(post["close"].iloc[-1])
        exit_time   = post.index[-1]
        exit_reason = "eod"
    elif pd.isna(sl_hit) or tp_hit <= sl_hit:
        exit_price  = tp
        exit_time   = tp_hit
        exit_reason = "tp"
    else:
        exit_price  = sl
        exit_time   = sl_hit
        exit_reason = "sl"

    pnl_points = exit_price - entry_price if direction == "long" else entry_price - exit_price

    return {
        "date":        session.index[0].date(),
        "direction":   direction,
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "sl":          sl,
        "tp":          tp,
        "fvg_type":    fvg["type"],
        "fvg_high":    fvg["high"],
        "fvg_low":     fvg["low"],
        "exit_reason": exit_reason,
        "pnl_points":  pnl_points,
    }


# ---------------------------------------------------------------------------
# Per-day driver
# ---------------------------------------------------------------------------

def _process_day(session: pd.DataFrame, params: dict) -> dict | None:
    if len(session) < 3:
        return None

    session_start      = session.index[0]
    eod_cutoff         = session_start + pd.Timedelta(minutes=params["eod_timeout_minutes"])
    fvg_timeout_minutes = params["fvg_timeout_minutes"]

    fvg_search_cutoff = session_start + pd.Timedelta(minutes=params["15m_fvg_timeout"])
    candles_15m = _build_15m(session[session.index <= fvg_search_cutoff])
    fvg_result = _find_first_15m_fvg(candles_15m)
    if fvg_result is None:
        return None

    fvg_type, fvg_high, fvg_low, fvg_confirmed_time = fvg_result
    fvg = {
        "type":           fvg_type,
        "high":           fvg_high,
        "low":            fvg_low,
        "confirmed_time": fvg_confirmed_time,
    }

    fvg_cutoff = fvg_confirmed_time + pd.Timedelta(minutes=fvg_timeout_minutes)

    entered, time_entered = _phase_1(session, fvg, fvg_cutoff)
    if not entered:
        return None

    return _phase_2(session, fvg, time_entered, eod_cutoff)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "date", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "sl", "tp",
    "fvg_type", "fvg_high", "fvg_low",
    "exit_reason", "pnl_points",
]


def run(folder_path: Path, start_date: pd.Timestamp,
        end_date: pd.Timestamp, params: dict | None = None) -> pd.DataFrame:
    merged_params = {**PARAMS, **(params or {})}

    files = sorted(Path(folder_path).glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    if not files:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        trade = _process_day(session, merged_params)
        if trade is not None:
            trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    return pd.DataFrame(trades)[_OUTPUT_COLUMNS]