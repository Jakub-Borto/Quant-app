"""Zone-based stop script: SL from where the pullback bottomed/topped vs the value-area zones.

risk_script: 2. The SL is derived from the PULLBACK candles only (retest candle .. bar before
entry — no lookahead). Then a fixed RR (`zone_rr`) sets the target. Execution reuses the shared
run_trade fill-simulation, so the ONLY difference from basic_risk is HOW the stop is chosen.

Zones (value area): VAL = bottom, POC = middle, VAH = top.
  Upper zone = POC..VAH,  Lower zone = VAL..POC.
"""

from .sl_tp import run_trade


def _zone_sl(post_retest, entry_ts, direction, levels):
    """Pick the stop from the pullback window's extremes vs the VAL/POC/VAH zones."""
    poc = levels["poc"]
    vah = levels["vah"]
    val = levels["val"]

    # --- SL window: drop the breakout bar (index 0), keep retest .. bar before entry ---
    # entry is taken at the entry bar's OPEN, so that bar's low/close are future data => excluded.
    window = post_retest.iloc[1:]
    window = window[window.index < entry_ts]

    # degenerate: nothing to measure => fall back to the basic VAL/VAH stop
    if window.empty:
        return val if direction == "long" else vah

    if direction == "long":
        lowest_close = float(window["close"].min())   # where the pullback bottomed (by close)
        lowest_low   = float(window["low"].min())      # how far the wick reached

        if poc <= lowest_close <= vah:                 # bottomed in the UPPER zone
            return poc if lowest_low >= poc else lowest_low
        elif val <= lowest_close < poc:                # bottomed in the LOWER zone
            return val if lowest_low >= val else lowest_low
        else:                                          # shouldn't occur (pullback re-enters VA)
            return val

    else:
        highest_close = float(window["close"].max())   # where the pullback topped (by close)
        highest_high  = float(window["high"].max())     # how far the wick reached

        if val <= highest_close <= poc:                # topped in the LOWER zone
            return poc if highest_high <= poc else highest_high
        elif poc < highest_close <= vah:               # topped in the UPPER zone
            return vah if highest_high <= vah else highest_high
        else:                                          # shouldn't occur (pullback re-enters VA)
            return vah


def run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params):
    """Returns the standard trade dict, or None if risk is non-positive."""
    sl = _zone_sl(post_retest, entry_ts, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    rr = params["zone_rr"]
    tp = entry_price + risk * rr if direction == "long" else entry_price - risk * rr

    return run_trade(
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )
