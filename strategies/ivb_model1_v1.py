import pandas as pd
from pathlib import Path
from datetime import time

PARAMS = {
    "ib_minutes":      30,     # IB range duration: 15, 30, or 60
    "delta_threshold": 30.0,   # minimum volume_delta_pct for entry candle
    "body_threshold":   0.5,   # body must cover 50% of bar range
    "rr":              1.0,    # fixed risk to reward ratio
    "sl_type":         0,      # 0 = VAL, 1 = swing low
    "retest_window":   30,     # max bars to wait for retest after breakout
    "entry_window":     15,    # bars to scan for entry after retest
    "trade_timeout":    60,     # bars before timeout logic kicks in
}

_OUTPUT_COLUMNS = [
    "date",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "exit_reason",
    "pnl_points",
]

def _compute_ivb_profile(ib_bars: pd.DataFrame) -> tuple:
    """
    Compute POC, VAH, VAL from tick_volume within the IB bars.
    Returns (poc, vah, val) or (None, None, None) if data is missing.
    """
    import json

    levels = {}

    for _, row in ib_bars.iterrows():
        tv = row["tick_volume"]
        if not tv or tv == "{}":
            continue
        try:
            raw = json.loads(tv)
        except Exception:
            continue
        for price_str, (buy_qty, sell_qty) in raw.items():
            price = float(price_str)
            total = buy_qty + sell_qty
            if price not in levels:
                levels[price] = 0
            levels[price] += total

    if not levels:
        return None, None, None

    # POC — price with highest volume
    poc = max(levels, key=levels.get)

    # Value area — 70% of total volume expanding from POC
    total_volume = sum(levels.values())
    target       = total_volume * 0.70
    sorted_prices = sorted(levels.keys())
    poc_idx       = sorted_prices.index(poc)

    lo_idx = poc_idx
    hi_idx = poc_idx
    va_volume = levels[poc]

    while va_volume < target:
        down_vol = levels[sorted_prices[lo_idx - 1]] if lo_idx > 0 else 0
        up_vol   = levels[sorted_prices[hi_idx + 1]] if hi_idx < len(sorted_prices) - 1 else 0

        if down_vol == 0 and up_vol == 0:
            break

        if up_vol >= down_vol:
            hi_idx += 1
            va_volume += up_vol
        else:
            lo_idx -= 1
            va_volume += down_vol

    vah = sorted_prices[hi_idx]
    val = sorted_prices[lo_idx]

    return poc, vah, val

def _detect_breakout(post_ib: pd.DataFrame, ivb_high: float, ivb_low: float) -> tuple:
    """
    Detects the first breakout of the IVB high or low.
    Returns (direction, breakout_pos) or (None, None) if no breakout.
    direction: "long" or "short"
    breakout_pos: position in post_ib (integer)
    """
    long_breakout  = post_ib["close"] > ivb_high
    short_breakout = post_ib["close"] < ivb_low

    long_pos  = int(long_breakout.argmax())  if long_breakout.any()  else None
    short_pos = int(short_breakout.argmax()) if short_breakout.any() else None

    if long_pos is None and short_pos is None:
        return None, None

    if long_pos is not None and short_pos is not None:
        direction = "long" if long_pos <= short_pos else "short"
    elif long_pos is not None:
        direction = "long"
    else:
        direction = "short"

    breakout_pos = long_pos if direction == "long" else short_pos

    return direction, breakout_pos

def _detect_retest(
    post_ib:      pd.DataFrame,
    breakout_pos: int,
    direction:    str,
    vah:          float,
    val:          float,
    poc:          float,
    retest_window: int,
) -> int:
    """
    Detects the first retest of the reload zone after the breakout.
    Long: bar low touches VAH-POC zone (low <= vah)
    Short: bar high touches VAL-POC zone (high >= val)
    Returns (retest_pos) or None if no retest found within window.
    retest_pos: position in post_ib
    """
    # Only look at bars after the breakout, within the retest window
    scan_start = breakout_pos + 1
    scan_end   = scan_start + retest_window
    scan       = post_ib.iloc[scan_start:scan_end]

    if scan.empty:
        return None

    if direction == "long":
        # Bar low must touch or enter the VAH-POC zone
        retest_mask = scan["low"] <= vah
    else:
        # Bar high must touch or enter the VAL-POC zone
        retest_mask = scan["high"] >= val

    if not retest_mask.any():
        return None

    # First bar that touches the zone
    retest_pos = scan_start + int(retest_mask.argmax())
    return retest_pos

def _find_entry(
    post_retest: pd.DataFrame,
    direction:   str,
    ivb_high:    float,
    ivb_low:     float,
    poc:         float,
    vah:         float,
    val:         float,
    params:      dict,
) -> tuple:
    """
    Scans post_retest window for a valid entry candle.
    Returns (entry_timestamp, entry_price) or (None, None).
    """
    if post_retest.empty:
        return None, None

    bar_range = post_retest["high"] - post_retest["low"]
    body      = (post_retest["close"] - post_retest["open"]).abs()
    full_body = (body / bar_range.replace(0, float("nan"))) >= params["body_threshold"]

    if direction == "long":
        delta_mask   = post_retest["volume_delta_pct"] >= params["delta_threshold"]
        dir_mask     = post_retest["close"] > post_retest["open"]
        invalid_mask = post_retest["close"] < val

        # First bar closing above ivb_high is allowed, later ones are not
        above_ivb = post_retest["close"] > ivb_high
        if above_ivb.any():
            blocked_above                         = above_ivb.copy()
            blocked_above.iloc[int(above_ivb.argmax())] = False
        else:
            blocked_above = above_ivb

        zone_mask = (post_retest["close"] >= poc) & ~blocked_above

    else:
        delta_mask   = post_retest["volume_delta_pct"] <= -params["delta_threshold"]
        dir_mask     = post_retest["close"] < post_retest["open"]
        invalid_mask = post_retest["close"] > vah

        # First bar closing below ivb_low is allowed, later ones are not
        below_ivb = post_retest["close"] < ivb_low
        if below_ivb.any():
            blocked_below                          = below_ivb.copy()
            blocked_below.iloc[int(below_ivb.argmax())] = False
        else:
            blocked_below = below_ivb

        zone_mask = (post_retest["close"] <= poc) & ~blocked_below

    candidates = delta_mask & dir_mask & zone_mask & full_body

    if not candidates.any():
        return None, None

    entry_rel = int(candidates.argmax())

    # Invalidation — any bar closes below VAL (or above VAH) before entry
    if invalid_mask.iloc[:entry_rel].any():
        return None, None

    # Entry is next bar open — if no next bar exists, no trade
    if entry_rel + 1 >= len(post_retest):
        return None, None

    entry_ts    = post_retest.index[entry_rel + 1]
    entry_price = float(post_retest.iloc[entry_rel + 1]["open"])

    return entry_ts, entry_price

def _compute_sl_tp(
    post_retest:  pd.DataFrame,   # ← fixed: was post_entry
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    val:          float,
    vah:          float,
    params:       dict,
) -> tuple:
    if params["sl_type"] == 0:
        sl = val if direction == "long" else vah

    else:
        bars_to_entry = post_retest.loc[:entry_ts]
        if bars_to_entry.empty:
            sl = val if direction == "long" else vah
        else:
            sl = float(bars_to_entry["low"].min())  if direction == "long" \
            else float(bars_to_entry["high"].max())

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None, None

    tp = entry_price + risk * params["rr"] if direction == "long" \
    else entry_price - risk * params["rr"]

    return sl, tp

def _run_trade(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    sl:           float,
    tp:           float,
    params:       dict,
) -> dict:
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    n           = len(pre_timeout)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl
        tp_hit = pre_timeout["high"] >= tp
    else:
        sl_hit = pre_timeout["high"] >= sl
        tp_hit = pre_timeout["low"]  <= tp

    sl_pos = int(sl_hit.argmax()) if sl_hit.any() else None
    tp_pos = int(tp_hit.argmax()) if tp_hit.any() else None

    # --- helper — takes explicit sl/tp so post-timeout adjusted levels work --
    def make_trade(exit_ts, exit_price, exit_reason, used_sl, used_tp):
        pnl = (exit_price - entry_price) if direction == "long" \
         else (entry_price - exit_price)
        return {
            "direction":   direction,
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          used_sl,
            "tp":          used_tp,
            "exit_reason": exit_reason,
            "pnl_points":  pnl,
        }

    # --- resolved within pre-timeout window ---------------------------------
    if sl_pos is not None or tp_pos is not None:
        if sl_pos is None:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        if tp_pos is None:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)
        if tp_pos <= sl_pos:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        else:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)

    # --- timeout reached ----------------------------------------------------
    if n < timeout:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", sl, tp)

    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout", sl, tp)

    # In loss — adjust to break even TP, tighten SL
    new_tp = entry_price
    new_sl = float(pre_timeout["low"].min())  if direction == "long" \
        else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= new_sl
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= new_sl
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


def _process_day(session: pd.DataFrame, params: dict):
    rth_session = session[
        (session.index.time >= time(9, 30)) &
        (session.index.time < time(16, 0))
    ]

    if len(rth_session) < params["ib_minutes"]:
        return None

    # --- Step 2: Compute Initial Balance ------------------------------------
    ib_bars   = rth_session.iloc[:params["ib_minutes"]]
    ivb_high  = float(ib_bars["high"].max())
    ivb_low   = float(ib_bars["low"].min())
    ivb_range = ivb_high - ivb_low

    if ivb_range <= 0:
        return None

    poc, vah, val = _compute_ivb_profile(ib_bars)
    if poc is None:                               # ← bug 4 fix
        return None

    # --- Step 3: Breakout detection -----------------------------------------
    post_ib = rth_session.iloc[params["ib_minutes"]:]
    if post_ib.empty:
        return None

    direction, breakout_pos = _detect_breakout(post_ib, ivb_high, ivb_low)
    if direction is None:
        return None

    # --- Step 4: Retest detection -------------------------------------------
    retest_pos = _detect_retest(
        post_ib       = post_ib,
        breakout_pos  = breakout_pos,
        direction     = direction,
        vah           = vah,
        val           = val,
        poc           = poc,
        retest_window = params["retest_window"],
    )

    if retest_pos is None:
        return None

    # --- Step 5: Entry detection --------------------------------------------
    post_retest = pd.concat([
        post_ib.iloc[[breakout_pos]],
        post_ib.iloc[retest_pos : retest_pos + params["entry_window"]]
    ])

    entry_ts, entry_price = _find_entry(
        post_retest = post_retest,
        direction   = direction,
        ivb_high    = ivb_high,
        ivb_low     = ivb_low,
        poc         = poc,
        vah         = vah,
        val         = val,
        params      = params,
    )

    if entry_ts is None:
        return None

    # --- Step 6: Compute SL and TP ------------------------------------------
    sl, tp = _compute_sl_tp(
        post_retest = post_retest,               # ← bug 1 fix
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        val         = val,
        vah         = vah,
        params      = params,
    )

    if sl is None:
        return None

    # --- Step 7: Trade execution --------------------------------------------
    post_entry = post_ib.loc[entry_ts:]

    trade = _run_trade(
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )

    return trade


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
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        if session.index.tz is None:
            continue
        trade = _process_day(session, merged_params)
        if trade is not None:
            trade["date"] = pd.Timestamp(f.stem).date()
            trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    return pd.DataFrame(trades)[_OUTPUT_COLUMNS]