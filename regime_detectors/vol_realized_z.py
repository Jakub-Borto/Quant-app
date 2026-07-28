"""
Realized-volatility regime detector — labels each snapshot `low` / `normal` /
`high` (plus implicit `unknown` = "no answer", which downstream must EXCLUDE,
never bucket as a fourth regime).

The idea: measure how much the market actually moved so far today, compare it
against the SAME clock window on recent days (matched windows kill the
intraday U-shape), express the gap in standard deviations of log volatility.
Early in the session lean on a HAR forecast built from prior days (volatility
clusters); as the session fills in, shift weight onto observation. Label with
a Schmitt trigger (enter/exit dead zone) so the state doesn't chatter, and
chain the state through yesterday's output file so a regime can persist
across days.

Everything is stored as separate pure columns — the blend and the thresholds
are label-time decisions, so both can be revisited by relabelling existing
files without recomputing anything.

Contract rolls are a NON-ISSUE here, deliberately: each input file is one
globex session with one contract, so every return in §2 is within-contract,
and the HAR forecast compares volatilities (not close-to-close returns across
days), so a roll gap never enters any calculation.

All centers/scales/z-scores live in LOG volatility space — log vol is roughly
normal, raw vol is not. `_center` / `_scale` naming (not `_mean` / `_std`)
because `robust_scale` changes what they are.
"""

import numpy as np
import pandas as pd

from modules.regime_detector.backend.runner import (RegimeContext,
                                                    SHARED_PARAMS,
                                                    snapshot_grid)
from modules.regime_detector.backend.schema import UNKNOWN_STATE

SCRIPT_VERSION = "1.0"

PARAMS = {
    **SHARED_PARAMS,                  # rth_start/rth_end/snapshot_minutes/lookback_days
    "min_lookback_days": 60,          # below this -> unknown
    "har_horizons": "1,5,22",         # comma-separated day counts
    "robust_scale": True,             # median/MAD vs mean/std
    "overnight_eff_bars": 45,         # overnight's weight as RTH evidence
    "blend_k": 90,                    # forecast -> observation crossover
    # tuned ONCE against the day-to-day transition matrix on the full ES
    # history (plan §8) — plan-default ±0.75/±0.35 left `normal` at a 65%
    # diagonal; this dead zone puts all three diagonals in 71-79% with
    # balance ~44/29/26. Per the plan: now stop touching them.
    "enter_high": 0.90,
    "enter_low": -0.90,
    "exit_high": 0.30,                # dead zone (Schmitt trigger)
    "exit_low": -0.30,
    "drop_zero_volume_bars": False,
}

REGIME_STATES = {
    "vol_state": {"states": ["low", "normal", "high"],
                  "colors": ["#4d9de0", "#8d99ae", "#e63946"]},
}

COLUMN_TIERS = {
    "regime": ["vol_state"],
    "score": ["vol_z", "vol_z_obs", "vol_z_fcst",
              "vol_z_on", "vol_z_rth", "vol_z_gbx"],
    "diagnostic": ["vol_blend_w", "vol_har_1", "vol_har_5", "vol_har_22",
                   "diag_zero_vol_bar_frac", "diag_excluded_returns",
                   "diag_state_chain_reset"],
}

# Expectations the runner verifies by measurement (warns, never fails).
# vol_har_* names follow the horizons param; defaults declared here — the
# runner skips declared columns that don't exist in a given run.
CONSTANT_COLUMNS = [
    "vol_z_fcst", "vol_har_1", "vol_har_5", "vol_har_22",
    "rth_hist_daily_vol_center", "rth_hist_daily_vol_scale",
    "on_hist_vol_center", "on_hist_vol_scale", "on_hist_n",
    "diag_state_chain_reset",
]

_ONE_MIN_NS = 60_000_000_000
_REAL_STATES = ("low", "normal", "high")


# ── §2: returns with an explicit validity mask ───────────────────────────────

def _parse_horizons(raw) -> list[int]:
    """'1,5,22' -> [1, 5, 22]."""
    try:
        horizons = sorted({int(tok) for tok in str(raw).split(",") if tok.strip()})
    except ValueError:
        raise ValueError(f"har_horizons must be comma-separated ints, "
                         f"got {raw!r}") from None
    if not horizons or horizons[0] < 1:
        raise ValueError(f"har_horizons must be positive ints, got {raw!r}")
    return horizons


def _return_stats(bars: pd.DataFrame, drop_zero_volume: bool) -> dict:
    """Per-bar squared log returns with the validity mask, as prefix sums.

    A return is EXCLUDED (a price gap, not a market move) when:
    - it is the first bar of the session (no valid predecessor);
    - the previous bar is not the immediately preceding minute (data gap —
      this also covers any halt/break inside the file);
    - either close is non-positive;
    - optionally, the bar has zero volume (`drop_zero_volume_bars`).

    Returns prefix-sum arrays of length n_bars+1 so any window [a, b) costs
    two lookups: cum_r2 (sum of valid r²), cum_n (valid return count),
    cum_excl (excluded count), cum_zero (zero-volume bar count).
    """
    n = len(bars)
    closes = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float) if "volume" in bars \
        else np.ones(n)

    r2 = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    if n >= 2:
        ts = bars.index.as_unit("ns").asi8        # normalize us/ns units
        consecutive = (ts[1:] - ts[:-1]) == _ONE_MIN_NS
        price_ok = (closes[1:] > 0) & (closes[:-1] > 0)
        valid[1:] = consecutive & price_ok
        if drop_zero_volume:
            valid[1:] &= volume[1:] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(np.where(price_ok, closes[1:] / closes[:-1], np.nan))
        r2[1:] = np.where(valid[1:], np.square(np.where(np.isfinite(r), r, 0.0)),
                          0.0)

    def prefix(a):
        out = np.zeros(n + 1)
        np.cumsum(a, out=out[1:])
        return out

    return {
        "cum_r2": prefix(r2),
        "cum_n": prefix(valid.astype(float)),
        "cum_excl": prefix((~valid).astype(float)),
        "cum_zero": prefix((volume == 0).astype(float)),
    }


def _window_vol(stats: dict, a: int, b: int) -> tuple[float, int]:
    """(per-bar σ, n valid returns) over returns into bars [a, b).
    Scale-free in n — mean of r², not the sum (a sum makes half-days look
    artificially calm). σ=0 or n=0 -> (nan, n)."""
    if b <= a:
        return np.nan, 0
    ss = stats["cum_r2"][b] - stats["cum_r2"][a]
    n = int(stats["cum_n"][b] - stats["cum_n"][a])
    if n == 0 or ss <= 0:
        return np.nan, n
    return float(np.sqrt(ss / n)), n


def _ln(sigma: float) -> float:
    return float(np.log(sigma)) if np.isfinite(sigma) and sigma > 0 else np.nan


# ── §4: yardsticks ───────────────────────────────────────────────────────────

def _center_scale(values, robust: bool) -> tuple[float, float, int]:
    """(center, scale, n) of a sample of ln σ. Robust: median / MAD·1.4826.
    Degenerate scale (0, or n < 2) -> NaN scale, so downstream z is NaN and
    the label is `unknown` — never a silently wrong number."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(arr)
    if n < 2:
        return np.nan, np.nan, n
    if robust:
        center = float(np.median(arr))
        scale = float(np.median(np.abs(arr - center)) * 1.4826)
    else:
        center = float(np.mean(arr))
        scale = float(np.std(arr, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nan
    return center, scale, n


def _z(value: float, center: float, scale: float) -> float:
    if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale)):
        return np.nan
    return (value - center) / scale


# ── §6: Schmitt-trigger labelling ────────────────────────────────────────────

def _schmitt(z_values, seed_state: str, enter_high: float, enter_low: float,
             exit_high: float, exit_low: float) -> list[str]:
    """Three-state hysteresis over a z sequence. NaN emits `unknown` and
    leaves the chain state untouched. A single step may exit an extreme AND
    enter the opposite one (z jumping +2 -> -2 goes high -> low)."""
    state = seed_state if seed_state in _REAL_STATES else "normal"
    out = []
    for z in z_values:
        if not np.isfinite(z):
            out.append(UNKNOWN_STATE)
            continue
        if state == "high" and z < exit_high:
            state = "normal"
        if state == "low" and z > exit_low:
            state = "normal"
        if state == "normal":
            if z > enter_high:
                state = "high"
            elif z < enter_low:
                state = "low"
        out.append(state)
    return out


def _read_final_state(path) -> str | None:
    """Final vol_state of an existing output file, None when unreadable."""
    try:
        col = pd.read_parquet(path, columns=["vol_state"])["vol_state"]
        return str(col.iloc[-1]) if len(col) else None
    except Exception:  # noqa: BLE001 — any unreadable file means "no chain"
        return None


# ── the run ──────────────────────────────────────────────────────────────────

def run_all(input_folder, output_folder, skip_existing, on_progress, params):
    p = {**PARAMS, **(params or {})}
    horizons = _parse_horizons(p["har_horizons"])
    max_h = max(horizons)
    lookback = int(p["lookback_days"])
    if lookback <= max_h:
        raise ValueError(f"lookback_days ({lookback}) must exceed the longest "
                         f"HAR horizon ({max_h})")
    minutes = int(p["snapshot_minutes"])
    min_lookback = int(p["min_lookback_days"])
    robust = bool(p["robust_scale"])
    drop_zero = bool(p["drop_zero_volume_bars"])
    on_eff = float(p["overnight_eff_bars"])
    blend_k = float(p["blend_k"])
    rth_start, rth_end = str(p["rth_start"]), str(p["rth_end"])

    def summarize(bars: pd.DataFrame) -> dict:
        """One prior day boiled down for the walker's window: cumulative
        (sum_sq, n) at each snapshot CLOCK TIME for globex and RTH, the
        frozen overnight window, and the daily RTH window. Keyed by clock
        ("10:00"), not index, so half-days and DST line up."""
        stats = _return_stats(bars, drop_zero)
        if bars.empty:
            return {"gbx": {}, "rth": {}, "on": (0.0, 0), "daily": (0.0, 0)}
        day = bars.index[-1].strftime("%Y-%m-%d")     # files are RTH-date keyed
        idx = bars.index
        lo = idx.searchsorted(pd.Timestamp(f"{day} {rth_start}", tz=str(idx.tz)))
        hi = idx.searchsorted(pd.Timestamp(f"{day} {rth_end}", tz=str(idx.tz)))
        gbx, rth = {}, {}
        for ts in snapshot_grid(idx, minutes):
            pos = idx.searchsorted(ts)
            key = ts.strftime("%H:%M")
            gbx[key] = (stats["cum_r2"][pos], int(stats["cum_n"][pos]))
            if pos > lo:
                rth[key] = (stats["cum_r2"][pos] - stats["cum_r2"][lo],
                            int(stats["cum_n"][pos] - stats["cum_n"][lo]))
        return {
            "gbx": gbx,
            "rth": rth,
            "on": (stats["cum_r2"][lo], int(stats["cum_n"][lo])),
            "daily": (stats["cum_r2"][hi] - stats["cum_r2"][lo],
                      int(stats["cum_n"][hi] - stats["cum_n"][lo])),
        }

    ctx = RegimeContext(input_folder, output_folder, params=p,
                        script_name="vol_realized_z",
                        script_version=SCRIPT_VERSION,
                        script_file=__file__,
                        states=REGIME_STATES, tiers=COLUMN_TIERS,
                        summarize=summarize, skip_existing=skip_existing,
                        on_progress=on_progress,
                        constant_columns=CONSTANT_COLUMNS)

    chain_resets: list[str] = []
    ctx.meta_extra["chain_resets"] = chain_resets     # live reference

    prev_day: str | None = None
    prev_state: str | None = None                     # None -> read lazily

    def ln_from_pair(pair) -> float:
        ss, n = pair
        return _ln(np.sqrt(ss / n)) if n > 0 and ss > 0 else np.nan

    for day in ctx.days():
        window = ctx.lookback(day)
        if ctx.should_skip(day):
            prev_day, prev_state = day, None          # chain reads the file
            continue

        # ── seed the state chain from yesterday's OUTPUT file (§6.2) ─────
        if prev_day is None:
            seed, reset = "normal", True
        else:
            state = prev_state if prev_state is not None \
                else _read_final_state(ctx.out_path(prev_day))
            if state in _REAL_STATES:
                seed, reset = state, False
            else:                                     # absent file OR unknown
                seed, reset = "normal", True
        if reset:
            chain_resets.append(day)

        bars = ctx.bars(day)
        stats = _return_stats(bars, drop_zero)
        times = ctx.snapshot_times(day)
        if not len(times):
            ctx.write(day, [])
            prev_day, prev_state = day, None
            continue

        idx = bars.index
        tz = str(idx.tz)
        pos_lo = idx.searchsorted(pd.Timestamp(f"{day} {rth_start}", tz=tz))

        # ── day-level pieces (constant within the file) ──────────────────
        summaries = list(window.values())             # date order
        daily_lns = [ln_from_pair(s["daily"]) for s in summaries]
        on_lns = [ln_from_pair(s["on"]) for s in summaries]

        daily_center, daily_scale, _daily_n = _center_scale(daily_lns, robust)
        on_center, on_scale, on_n = _center_scale(on_lns, robust)

        har_means = {h: (float(np.nanmean(daily_lns[-h:]))
                         if daily_lns and not np.isnan(daily_lns[-h:]).all()
                         else np.nan)
                     for h in horizons}
        ln_hat = (float(np.mean(list(har_means.values())))
                  if all(np.isfinite(v) for v in har_means.values())
                  else np.nan)
        z_fcst = _z(ln_hat, daily_center, daily_scale)

        # warm-up: an explicit "no answer", never a silently wrong number
        day_known = (len(window) >= min_lookback
                     and len(window) >= max_h + 1)

        on_vol, on_vol_n = _window_vol(stats, 0, pos_lo)
        z_on_frozen = _z(_ln(on_vol), on_center, on_scale)

        rows, z_seq = [], []
        for ts in times:
            pos = int(idx.searchsorted(ts))
            key = ts.strftime("%H:%M")
            post_open = pos >= pos_lo and ts >= pd.Timestamp(
                f"{day} {rth_start}", tz=tz)

            gbx_vol, gbx_n = _window_vol(stats, 0, pos)
            rth_vol, rth_n = (_window_vol(stats, pos_lo, pos) if post_open
                              else (np.nan, 0))

            # matched-window yardsticks: same clock key on each prior day
            gbx_sample = [ln_from_pair(s["gbx"][key]) for s in summaries
                          if key in s["gbx"]]
            rth_sample = [ln_from_pair(s["rth"][key]) for s in summaries
                          if key in s["rth"]]
            g_c, g_s, g_n = _center_scale(gbx_sample, robust)
            r_c, r_s, r_n = _center_scale(rth_sample, robust)

            z_gbx = _z(_ln(gbx_vol), g_c, g_s)
            z_rth = _z(_ln(rth_vol), r_c, r_s)
            z_on = z_on_frozen if post_open else np.nan

            # ── observation + blend (§5) ─────────────────────────────────
            if post_open:
                parts = [(on_eff, z_on), (float(rth_n), z_rth)]
                used = [(wgt, z) for wgt, z in parts
                        if wgt > 0 and np.isfinite(z)]
                z_obs = (sum(w_ * z_ for w_, z_ in used)
                         / sum(w_ for w_, z_ in used)) if used else np.nan
                n_eff = sum(w_ for w_, z_ in used)
            else:
                z_obs = z_gbx
                n_eff = float(gbx_n) if np.isfinite(z_gbx) else 0.0

            w = n_eff / (n_eff + blend_k) if n_eff > 0 else 0.0
            if np.isfinite(z_obs) and np.isfinite(z_fcst):
                z = w * z_obs + (1.0 - w) * z_fcst
            elif np.isfinite(z_obs):
                z, w = z_obs, 1.0
            elif np.isfinite(z_fcst):
                z, w = z_fcst, 0.0
            else:
                z = np.nan

            z_seq.append(z if day_known else np.nan)
            row = {
                "ts": ts,
                "vol_z": z, "vol_z_obs": z_obs, "vol_z_fcst": z_fcst,
                "vol_z_on": z_on, "vol_z_rth": z_rth, "vol_z_gbx": z_gbx,
                "gbx_vol": gbx_vol, "gbx_vol_n": gbx_n,
                "rth_vol": rth_vol, "rth_vol_n": rth_n,
                "on_vol": on_vol if post_open else _window_vol(stats, 0, pos)[0],
                "on_vol_n": on_vol_n if post_open else gbx_n,
                "gbx_hist_vol_center": g_c, "gbx_hist_vol_scale": g_s,
                "gbx_hist_n": g_n,
                "rth_hist_vol_center": r_c, "rth_hist_vol_scale": r_s,
                "rth_hist_n": r_n,
                "on_hist_vol_center": on_center, "on_hist_vol_scale": on_scale,
                "on_hist_n": on_n,
                "rth_hist_daily_vol_center": daily_center,
                "rth_hist_daily_vol_scale": daily_scale,
                "vol_blend_w": w,
                "diag_zero_vol_bar_frac": (stats["cum_zero"][pos] / pos
                                           if pos else np.nan),
                "diag_excluded_returns": int(stats["cum_excl"][pos]),
                "diag_state_chain_reset": bool(reset),
            }
            for h, v in har_means.items():
                row[f"vol_har_{h}"] = v
            rows.append(row)

        states = _schmitt(z_seq, seed,
                          float(p["enter_high"]), float(p["enter_low"]),
                          float(p["exit_high"]), float(p["exit_low"]))
        for row, state in zip(rows, states):
            row["vol_state"] = state

        ctx.write(day, rows)
        prev_day, prev_state = day, states[-1]
