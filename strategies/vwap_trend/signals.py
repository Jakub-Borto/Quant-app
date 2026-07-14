"""Pure signal logic: bars -> per-bar side decisions. No I/O, no pandas.

Two stages:

  classify()  — vectorized: each bar's close vs the VWAP neutral band
                -> LONG / SHORT / NEUTRAL / NO_SIGNAL.
  resolve()   — O(bars) state walk turning raw signals into the effective
                side decided ON each bar's close (what the position should
                be from the NEXT bar on), applying band_rule, no-signal
                carry and the exclusion window.

Codes are small ints so the walk stays cheap inside the optimizer loop.
"""

import numpy as np

LONG      = 1
SHORT     = -1
NEUTRAL   = 0     # close inside the band (inclusive)
NO_SIGNAL = 2     # NaN VWAP, or zero-volume bar with skip_zero_volume on
FLAT      = 0     # decision value (shares 0 with NEUTRAL deliberately)


def classify(close: np.ndarray, vwap: np.ndarray, volume: np.ndarray,
             band_points: float, skip_zero_volume: bool) -> np.ndarray:
    """Raw per-bar signal. band_points = vwap_band_ticks * tick_size.

    band == 0 reduces to the paper's strict close > vwap / close < vwap,
    with an exact-equality close falling into NEUTRAL.
    """
    upper = vwap + band_points
    lower = vwap - band_points
    codes = np.full(len(close), NEUTRAL, dtype=np.int8)
    codes[close > upper] = LONG
    codes[close < lower] = SHORT
    # NaN VWAP compares False on both -> would read NEUTRAL; make it NO_SIGNAL
    no_sig = np.isnan(vwap)
    if skip_zero_volume:
        # forward-filled synthetic bars: close is a copy of the prior close and
        # carries no information — they must not manufacture flips
        no_sig = no_sig | (volume == 0)
    codes[no_sig] = NO_SIGNAL
    return codes


def resolve(codes: np.ndarray, excl: np.ndarray, carry_forward: bool) -> np.ndarray:
    """Raw signals -> effective side decided on each bar's close.

    decision[t] is the side the position should have DURING bar t+1 (fills are
    at the next bar's open — the engine consumes decisions with that shift).

      - NO_SIGNAL carries the previous decision (both band rules).
      - NEUTRAL carries under 'carry_forward', forces FLAT under 'flat'.
      - Exclusion bars force FLAT and reset the carry state, so nothing is
        carried across the exclusion window.
    """
    n = len(codes)
    decisions = np.empty(n, dtype=np.int8)
    state = FLAT
    codes_l = codes.tolist()
    excl_l  = excl.tolist()
    for t in range(n):
        if excl_l[t]:
            state = FLAT
        else:
            c = codes_l[t]
            if c == LONG or c == SHORT:
                state = c
            elif c == NEUTRAL and not carry_forward:
                state = FLAT
            # NEUTRAL under carry_forward and NO_SIGNAL: keep state
        decisions[t] = state
    return decisions
