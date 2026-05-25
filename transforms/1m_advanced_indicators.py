"""
transforms/indicators_1m.py

Reads enriched 1-minute candle Parquet files (one per day).
Writes one indicators Parquet per day with 29 columns.

Output columns:
    vwap_bar_globex,  _std1/2/3_up/dn          (7)
    vwap_bar_rth,     _std1/2/3_up/dn          (7)
    vwap_tick_globex, _std1/2/3_up/dn          (7)
    vwap_tick_rth,    _std1/2/3_up/dn          (7)
    cumulative_delta                            (1)
                                          total: 29
"""

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# BAR VWAP
# ---------------------------------------------------------------------------

def _compute_bar_vwap(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bar-based VWAP and +/-1/2/3 sigma bands for two anchors:
      - globex : first bar of the file (18:00 NY)
      - rth    : first bar at or after 09:30 NY (NaN before that)

    Typical price per bar = (high + low + close) / 3
    VWAP at bar N = cumsum(tp * volume)[N] / cumsum(volume)[N]

    Std bands:
      variance[N] = cumsum(tp^2 * volume)[N] / cumsum(volume)[N]  -  vwap[N]^2
      std[N]      = sqrt(variance[N])
      band_up[N]  = vwap[N] + k * std[N]
      band_dn[N]  = vwap[N] - k * std[N]

    Returns a DataFrame with the same index as `candles`, 14 columns.
    """

    out = pd.DataFrame(index=candles.index)

    # --- typical price (one value per bar) ---------------------------------
    tp  = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    vol = candles["volume"].astype(float)

    # weighted price and weighted price-squared (needed for variance)
    tp_vol  = tp * vol        # shape: (n_bars,)
    tp2_vol = tp * tp * vol   # shape: (n_bars,)  <- tp squared x vol

    # =======================================================================
    # GLOBEX anchor — accumulate from bar 0 (no masking needed)
    # =======================================================================

    cum_vol_g  = vol.cumsum()
    cum_tpv_g  = tp_vol.cumsum()
    cum_tp2v_g = tp2_vol.cumsum()

    vwap_g = cum_tpv_g / cum_vol_g

    # variance = E[x^2] - E[x]^2  (population variance of price, vol-weighted)
    var_g = (cum_tp2v_g / cum_vol_g) - (vwap_g ** 2)
    # numerical noise can push variance slightly below zero -> clip
    var_g = var_g.clip(lower=0.0)
    std_g = np.sqrt(var_g)

    out["vwap_bar_globex"]         = vwap_g
    out["vwap_bar_globex_std1_up"] = vwap_g + 1 * std_g
    out["vwap_bar_globex_std1_dn"] = vwap_g - 1 * std_g
    out["vwap_bar_globex_std2_up"] = vwap_g + 2 * std_g
    out["vwap_bar_globex_std2_dn"] = vwap_g - 2 * std_g
    out["vwap_bar_globex_std3_up"] = vwap_g + 3 * std_g
    out["vwap_bar_globex_std3_dn"] = vwap_g - 3 * std_g

    # =======================================================================
    # RTH anchor — accumulate only from 09:30 NY onward
    # =======================================================================

    # Boolean mask: True for every bar at or after 09:30
    rth_mask = pd.Series(
        (candles.index.time >= time(9, 30)) & (candles.index.time < time(16, 0)),
        index=candles.index
    )

    # Zero out pre-RTH bars so their weight does not contaminate the cumsum.
    # After 09:30 the values are identical to the raw series.
    tp_vol_r  = tp_vol.where(rth_mask, other=0.0)
    tp2_vol_r = tp2_vol.where(rth_mask, other=0.0)
    vol_r     = vol.where(rth_mask, other=0.0)

    cum_vol_r  = vol_r.cumsum()
    cum_tpv_r  = tp_vol_r.cumsum()
    cum_tp2v_r = tp2_vol_r.cumsum()

    # np.where avoids pandas ZeroDivisionWarning when cum_vol_r == 0
    vwap_r_raw = np.where(
        cum_vol_r > 0,
        cum_tpv_r / cum_vol_r,
        np.nan
    )
    vwap_r = pd.Series(vwap_r_raw, index=candles.index)

    var_r_raw = np.where(
        cum_vol_r > 0,
        cum_tp2v_r / cum_vol_r - vwap_r_raw ** 2,
        np.nan
    )
    var_r = pd.Series(np.maximum(var_r_raw, 0.0), index=candles.index)
    std_r = np.sqrt(var_r)

    # Enforce NaN before 09:30 explicitly (belt-and-suspenders)
    vwap_r = vwap_r.where(rth_mask, other=np.nan)
    std_r  = std_r.where(rth_mask,  other=np.nan)

    out["vwap_bar_rth"]         = vwap_r
    out["vwap_bar_rth_std1_up"] = vwap_r + 1 * std_r
    out["vwap_bar_rth_std1_dn"] = vwap_r - 1 * std_r
    out["vwap_bar_rth_std2_up"] = vwap_r + 2 * std_r
    out["vwap_bar_rth_std2_dn"] = vwap_r - 2 * std_r
    out["vwap_bar_rth_std3_up"] = vwap_r + 3 * std_r
    out["vwap_bar_rth_std3_dn"] = vwap_r - 3 * std_r

    return out


# ---------------------------------------------------------------------------
# TICK VWAP
# ---------------------------------------------------------------------------

def _parse_tick_volume(tv_json: str) -> tuple:
    """
    Parse one bar's tick_volume JSON string.

    Format: {"price_as_str": [buy_qty, sell_qty], ...}
    Returns (prices, quantities) as float64 arrays.
    Returns two empty arrays if the value is missing or malformed.
    """
    if not tv_json or tv_json != tv_json:   # handles None and NaN
        return np.array([]), np.array([])
    try:
        raw    = json.loads(tv_json)
        prices = np.array(list(raw.keys()), dtype=np.float64)
        qtys   = np.array(
            [b + s for b, s in raw.values()], dtype=np.float64
        )
        return prices, qtys
    except Exception:
        return np.array([]), np.array([])


def _compute_tick_vwap(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tick-level VWAP and +/-1/2/3 sigma bands for globex and RTH anchors.

    For each bar we unpack tick_volume to get the actual price distribution
    within the bar, rather than using a single typical price.

    Per bar contribution to the running totals:
        bar_wt[i]   = sum(qty)                total contracts in bar
        bar_wpx[i]  = sum(price * qty)        volume-weighted price sum
        bar_wpx2[i] = sum(price^2 * qty)      volume-weighted price-squared sum

    VWAP[N] = cumsum(bar_wpx)[N] / cumsum(bar_wt)[N]
    std[N]  = sqrt( cumsum(bar_wpx2)[N] / cumsum(bar_wt)[N] - VWAP[N]^2 )
    """

    out = pd.DataFrame(index=candles.index)
    n   = len(candles)

    # Pre-allocate per-bar aggregates (float64, one value per bar)
    bar_wt   = np.zeros(n, dtype=np.float64)   # total volume
    bar_wpx  = np.zeros(n, dtype=np.float64)   # sum(price * qty)
    bar_wpx2 = np.zeros(n, dtype=np.float64)   # sum(price^2 * qty)

    # This loop is over bars (~500 per day), not over ticks — acceptable cost.
    # Each iteration does vectorized numpy ops on the price levels within a bar.
    for i, tv_json in enumerate(candles["tick_volume"]):
        prices, qtys = _parse_tick_volume(tv_json)
        if len(prices) == 0:
            continue
        bar_wt[i]   = qtys.sum()
        bar_wpx[i]  = (prices * qtys).sum()
        bar_wpx2[i] = (prices * prices * qtys).sum()

    # --- inner helper: build vwap + std given optional RTH mask ------------
    def _build(wt, wpx, wpx2, mask=None):
        """
        mask = None  -> globex (use all bars)
        mask = bool array -> rth (zero out pre-RTH bars before cumsum)
        """
        if mask is not None:
            wt   = np.where(mask, wt,   0.0)
            wpx  = np.where(mask, wpx,  0.0)
            wpx2 = np.where(mask, wpx2, 0.0)

        cum_wt   = np.cumsum(wt)
        cum_wpx  = np.cumsum(wpx)
        cum_wpx2 = np.cumsum(wpx2)

        valid    = cum_wt > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap_arr = np.where(valid, cum_wpx  / cum_wt, np.nan)
            var_arr  = np.where(valid, cum_wpx2 / cum_wt - vwap_arr ** 2, np.nan)
        var_arr  = np.maximum(var_arr, 0.0)
        std_arr  = np.sqrt(var_arr)

        if mask is not None:
            vwap_arr = np.where(mask, vwap_arr, np.nan)
            std_arr  = np.where(mask, std_arr,  np.nan)

        return (
            pd.Series(vwap_arr, index=candles.index),
            pd.Series(std_arr,  index=candles.index),
        )

    # =======================================================================
    # GLOBEX
    # =======================================================================
    vwap_g, std_g = _build(bar_wt, bar_wpx, bar_wpx2)

    out["vwap_tick_globex"]         = vwap_g
    out["vwap_tick_globex_std1_up"] = vwap_g + 1 * std_g
    out["vwap_tick_globex_std1_dn"] = vwap_g - 1 * std_g
    out["vwap_tick_globex_std2_up"] = vwap_g + 2 * std_g
    out["vwap_tick_globex_std2_dn"] = vwap_g - 2 * std_g
    out["vwap_tick_globex_std3_up"] = vwap_g + 3 * std_g
    out["vwap_tick_globex_std3_dn"] = vwap_g - 3 * std_g

    # =======================================================================
    # RTH
    # =======================================================================
    rth_mask = (
        (candles.index.time >= time(9, 30)) &
        (candles.index.time < time(16, 0))
    )

    vwap_r, std_r = _build(bar_wt, bar_wpx, bar_wpx2, rth_mask)

    out["vwap_tick_rth"]         = vwap_r
    out["vwap_tick_rth_std1_up"] = vwap_r + 1 * std_r
    out["vwap_tick_rth_std1_dn"] = vwap_r - 1 * std_r
    out["vwap_tick_rth_std2_up"] = vwap_r + 2 * std_r
    out["vwap_tick_rth_std2_dn"] = vwap_r - 2 * std_r
    out["vwap_tick_rth_std3_up"] = vwap_r + 3 * std_r
    out["vwap_tick_rth_std3_dn"] = vwap_r - 3 * std_r

    return out


# ---------------------------------------------------------------------------
# CUMULATIVE DELTA
# ---------------------------------------------------------------------------

def _compute_cumulative_delta(candles: pd.DataFrame) -> pd.DataFrame:
    """
    CVD = cumsum(buy_volume - sell_volume), anchored at first bar (18:00 NY).
    Resets every file. No cross-day accumulation.
    """
    out = pd.DataFrame(index=candles.index)
    out["cumulative_delta"] = (
        candles["buy_volume"].astype(float)
        - candles["sell_volume"].astype(float)
    ).cumsum()
    return out


# ---------------------------------------------------------------------------
# ABSORPTION
# ---------------------------------------------------------------------------

def _compute_absorption(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling OFI-beta residual absorption score.

    Columns output:
        beta              — rolling impact coefficient (returns per unit delta)
        residual          — actual return minus expected return (ticks)
        absorption_score  — normalized sign-adjusted residual
                            positive = absorption detected (either side)
                            NaN outside valid signal window (10:35 - 15:50)

    Only computed on RTH bars (09:30-16:00).
    Signal valid only after 60-bar warmup (from 10:35).
    ETH bars are always NaN.
    """

    WINDOW          = 60
    TICK_SIZE       = 0.25
    SIGNAL_START    = time(10, 35)
    SIGNAL_END      = time(15, 49)

    out = pd.DataFrame(index=candles.index)

    # --- isolate RTH bars only ----------------------------------------------
    rth_mask = (
        (candles.index.time >= time(9, 30)) &
        (candles.index.time <  time(16, 0))
    )
    rth = candles[rth_mask].copy()

    if len(rth) < WINDOW:
        out["beta"]             = np.nan
        out["residual"]         = np.nan
        out["absorption_score"] = np.nan
        return out

    # --- returns in ticks (close to close) ----------------------------------
    returns = rth["close"].diff() / TICK_SIZE

    # --- delta --------------------------------------------------------------
    delta = rth["volume_delta"].astype(float)

    # --- rolling beta = cov(returns, delta) / var(delta) -------------------
    # min_periods=WINDOW ensures we get NaN during warmup, not noisy estimates
    roll_cov = returns.rolling(WINDOW, min_periods=WINDOW).cov(delta)
    roll_var = delta.rolling(WINDOW,   min_periods=WINDOW).var()

    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(roll_var > 0, roll_cov / roll_var, np.nan)
    beta = pd.Series(beta, index=rth.index)

    # --- expected move and residual -----------------------------------------
    expected  = beta * delta
    residual  = returns - expected

    # --- normalize residual -------------------------------------------------
    sigma_u = residual.rolling(WINDOW, min_periods=WINDOW).std()

    with np.errstate(divide='ignore', invalid='ignore'):
        score_raw = np.where(
            sigma_u > 0,
            -np.sign(delta) * residual / sigma_u,
            np.nan
        )
    score = pd.Series(score_raw, index=rth.index)

    # --- enforce valid signal window ----------------------------------------
    valid_mask = (
        (rth.index.time >= SIGNAL_START) &
        (rth.index.time <= SIGNAL_END)
    )
    score    = score.where(valid_mask,    other=np.nan)
    beta     = beta.where(valid_mask,     other=np.nan)
    residual = residual.where(valid_mask, other=np.nan)

    # --- align back to full session index (ETH bars get NaN) ----------------
    out["beta"]             = beta
    out["residual"]         = residual
    out["absorption_score"] = score

    return out

# ---------------------------------------------------------------------------
# SINGLE-FILE PROCESSOR
# ---------------------------------------------------------------------------

def _process_file(
    input_path:  Path,
    output_path: Path,
    skip_existing: bool = True,
    on_log: callable = None,
) -> None:
    """
    Read one candle Parquet, compute all 29 indicator columns,
    write indicators-only Parquet to output_path.

    on_log(msg) — same pattern as build_candles in candles_1m.py.
    """

    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    date_label = input_path.stem   # "2024-01-02"

    if skip_existing and output_path.exists():
        log(f"↷ Skipping {date_label} — already processed")
        return

    candles = pd.read_parquet(input_path)

    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex, got {type(candles.index)}")
    if candles.index.tz is None:
        raise ValueError("Index has no timezone — expected America/New_York")
    if candles.empty:
        raise ValueError("Empty candle file")

    bar_vwap  = _compute_bar_vwap(candles)        # 14 columns
    tick_vwap = _compute_tick_vwap(candles)        # 14 columns
    cvd       = _compute_cumulative_delta(candles) #  1 column
    absorption = _compute_absorption(candles)       # 3 columns

    indicators = pd.concat([bar_vwap, tick_vwap, cvd, absorption], axis=1)

    

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(output_path)
    log(f"✓ Saved {date_label}")


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE — called by the Data Formatter UI
# ---------------------------------------------------------------------------

def run_all(
    input_folder:  str,
    output_folder: str,
    skip_existing: bool = True,
    on_progress:   callable = None,
) -> None:
    """
    Process all daily candle Parquet files in input_folder.
    Writes one indicators Parquet per day to output_folder.

    Standard transform interface:
        on_progress(current, total, message)
    """
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.parquet"))
    total = len(files)

    if total == 0:
        if on_progress:
            on_progress(0, 0, "No .parquet files found in input folder.")
        return

    for i, file in enumerate(files, start=1):
        out_file = output_path / file.name   # YYYY-MM-DD.parquet

        # on_log forwards messages from _process_file into on_progress
        def on_log(msg: str, _i=i, _total=total):
            if on_progress:
                on_progress(_i, _total, msg)

        try:
            _process_file(
                input_path    = file,
                output_path   = out_file,
                skip_existing = skip_existing,
                on_log        = on_log,
            )
        except Exception as e:
            if on_progress:
                on_progress(i, total, f"ERROR {file.name}: {e}")
            continue