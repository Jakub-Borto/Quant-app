"""Per-day precomputed context shared by all entry finders and risk scripts.

This is the core of the vectorized rewrite: every per-day quantity is materialized ONCE as
plain numpy arrays / pre-parsed JSON, and the windows the finders and risk scripts operate on
(`post_retest`, `post_entry`) become positional views into those arrays instead of DataFrame
slices. All the semantics of the original per-bar code are preserved exactly:

  - tick_volume / passive_orders are parsed once per day into (prices, a, b) arrays kept in
    JSON DOCUMENT ORDER, so "first qualifying level" scans pick the same level the original
    dict-iteration loops did. A missing / "{}" / unparseable value becomes None, reproducing
    the original per-access failure semantics.
  - baselines, CVD and VWAP bands are stored positionally aligned to the RTH index, so the
    original `Series.get(ts)` lookups become integer indexing.
"""

import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON parsing (once per day)
# ---------------------------------------------------------------------------

def _to_arrays(raw: dict) -> tuple:
    """One parsed bar dict -> (prices, first_vals, second_vals) in document order.
    numpy converts the price-key strings to float64 at C level; value dtypes are inferred
    (ints stay integer) so `.item()` returns the same python type the original json values
    had — keeps note fields like trigger_volume byte-identical."""
    prices = np.array(list(raw), dtype=np.float64)
    vals   = np.array(list(raw.values()))
    return prices, vals[:, 0], vals[:, 1]


def parse_json_column(values) -> list:
    """Parse a tick_volume ({price: [buy, sell]}) or passive_orders ({price: [size, count]})
    column. Returns per bar (prices, first_vals, second_vals) numpy arrays in document order,
    or None (missing / empty / unparseable — matches the original "no data" semantics).

    Fast path: the whole column is parsed as ONE json array and all keys/values are converted
    in a handful of whole-day numpy C calls (per-bar tuples are views into them). If any bar
    is malformed the per-bar fallback isolates it; if any value is non-integer the per-bar
    array build preserves that bar's exact dtype (so `.item()` keeps the original json types).
    """
    strings = [s if (isinstance(s, str) and s and s != "{}") else None for s in values]

    try:
        docs = json.loads("[" + ",".join(s if s is not None else "{}" for s in strings) + "]")
    except Exception:
        docs = None

    out = []
    if docs is not None:
        counts = [len(raw) for raw in docs]
        vals_flat = np.array([x for raw in docs for v in raw.values() for x in v])
        # whole-day batching folds every bar's values into one dtype; that is only lossless
        # when everything is integer (the real schema) — otherwise keep per-bar dtypes
        if vals_flat.dtype.kind in "iu":
            keys_flat = np.array([k for raw in docs for k in raw], dtype=np.float64)
            vals2 = vals_flat.reshape(-1, 2)
            a, b  = vals2[:, 0], vals2[:, 1]
            offs  = np.concatenate(([0], np.cumsum(counts)))
            for i, cnt in enumerate(counts):
                if cnt == 0:
                    out.append(None)
                else:
                    s0, s1 = offs[i], offs[i + 1]
                    out.append((keys_flat[s0:s1], a[s0:s1], b[s0:s1]))
        else:
            for raw in docs:
                out.append(_to_arrays(raw) if raw else None)
        return out

    # fallback: per-bar parse, so a single malformed bar only disables itself
    loads = json.loads
    for s in strings:
        if s is None:
            out.append(None)
            continue
        try:
            raw = loads(s)
        except Exception:
            out.append(None)
            continue
        out.append(_to_arrays(raw) if raw else None)
    return out


# ---------------------------------------------------------------------------
# Day-level context
# ---------------------------------------------------------------------------

class DayData:
    """All per-day state as positional arrays over the RTH session.

    Built from the FULL session + an RTH boolean mask: numeric columns are converted once at
    C level and masked as numpy arrays; the JSON string columns are masked at the Series
    level first so only RTH strings are ever materialized. No sliced DataFrame is created.

    A DayData is param-independent — except for `session_start`, which shapes the RTH slice
    and is therefore part of the day-core cache key — and immutable-after-parse, so the
    package caches it across backtest runs (see the day-core cache in __init__). The two expensive pieces — JSON
    parsing and the per-bar passive-max pass — are lazy, memoized properties: a day (or a
    param set) that never touches them never pays for them, and once computed they live on
    the cached object for every later run.

    Per-RUN derived state (baselines, cvd/cvd_std, vwap_bands) is param-dependent: it is
    reset and rebuilt by core.process_day on every call. `cvd_raw` / `bands_raw` hold the
    param-independent indicator arrays the derived fields are cut from.
    """

    def __init__(self, session: pd.DataFrame, rth_mask: np.ndarray, rth_minutes: np.ndarray,
                 warmup_min: int):
        self.index = session.index[rth_mask]
        self.n     = len(self.index)
        self.open  = session["open"].to_numpy(dtype=np.float64)[rth_mask]
        self.high  = session["high"].to_numpy(dtype=np.float64)[rth_mask]
        self.low   = session["low"].to_numpy(dtype=np.float64)[rth_mask]
        self.close = session["close"].to_numpy(dtype=np.float64)[rth_mask]
        self.vdp   = session["volume_delta_pct"].to_numpy(dtype=np.float64)[rth_mask]
        self.buy_vol  = session["buy_volume"].to_numpy(dtype=np.float64)[rth_mask]
        self.sell_vol = session["sell_volume"].to_numpy(dtype=np.float64)[rth_mask]

        # bars at/after the baseline warm-up start (positions into the RTH arrays);
        # warmup_min = session_start + BASELINE_WARMUP_MINUTES, resolved by core
        self.warmup_pos = np.flatnonzero(rth_minutes >= warmup_min)

        # raw JSON strings, parsed lazily (see properties below)
        self._tv_raw = (session["tick_volume"][rth_mask].to_numpy()
                        if "tick_volume" in session.columns else None)
        self._po_raw = (session["passive_orders"][rth_mask].to_numpy()
                        if "passive_orders" in session.columns else None)
        self._tick_volume    = None
        self._passive_orders = None
        self._best_long  = None
        self._best_short = None

        # param-independent indicator arrays (set by core.build_day_core when available)
        self.cvd_raw   = None       # float array aligned to index, or None
        self.bands_raw = None       # dict col -> float array, or None

        # per-run derived state — reset + rebuilt by core.process_day for every run
        self.sell_baseline = None
        self.buy_baseline  = None
        self.passive_baseline_long  = None
        self.passive_baseline_short = None
        self.cvd        = None      # float array aligned to index, or None
        self.cvd_std    = None      # float array aligned to index, or None
        self.vwap_bands = None      # dict col -> float array, or None

    def reset_run_state(self):
        """Clear everything param-dependent before a (re-)run over a cached day."""
        self.sell_baseline = None
        self.buy_baseline  = None
        self.passive_baseline_long  = None
        self.passive_baseline_short = None
        self.cvd        = None
        self.cvd_std    = None
        self.vwap_bands = None

    @property
    def tick_volume(self) -> list:
        tv = self._tick_volume
        if tv is None:
            raw = self._tv_raw
            tv = parse_json_column(raw) if raw is not None else [None] * self.n
            self._tick_volume = tv
            self._tv_raw = None
        return tv

    @property
    def passive_orders(self) -> list:
        po = self._passive_orders
        if po is None:
            raw = self._po_raw
            po = parse_json_column(raw) if raw is not None else [None] * self.n
            self._passive_orders = po
            self._po_raw = None
        return po

    def _ensure_best_passive(self):
        """Per-bar max defended-side resting size (count > 0), both directions, fully
        vectorized over the day's concatenated levels — feeds build_passive_baseline AND
        the passive finders' candidate prefilter. Computed once per day, ever."""
        if self._best_long is not None:
            return
        best_long  = np.full(self.n, np.nan)
        best_short = np.full(self.n, np.nan)
        po_list = self.passive_orders
        lens = np.array([0 if po is None else po[0].size for po in po_list])
        if lens.any():
            bar_ids    = np.repeat(np.arange(self.n), lens)
            all_prices = np.concatenate([po[0] for po in po_list if po is not None])
            all_sizes  = np.concatenate([po[1] for po in po_list if po is not None]).astype(np.float64)
            all_counts = np.concatenate([po[2] for po in po_list if po is not None])
            opens_rep  = self.open[bar_ids]
            ok    = all_counts > 0
            below = ok & (all_prices < opens_rep)
            above = ok & (all_prices > opens_rep)
            bl = np.full(self.n, -np.inf)
            bs = np.full(self.n, -np.inf)
            np.maximum.at(bl, bar_ids[below], all_sizes[below])
            np.maximum.at(bs, bar_ids[above], all_sizes[above])
            best_long[bl > -np.inf]  = bl[bl > -np.inf]
            best_short[bs > -np.inf] = bs[bs > -np.inf]
        self._best_long  = best_long
        self._best_short = best_short

    @property
    def best_passive_long(self) -> np.ndarray:
        self._ensure_best_passive()
        return self._best_long

    @property
    def best_passive_short(self) -> np.ndarray:
        self._ensure_best_passive()
        return self._best_short


# ---------------------------------------------------------------------------
# Shared window masks
# ---------------------------------------------------------------------------

def _build_masks(o, h, l, c, vdp, direction, params):
    """(wick_frac, confirm_any, confirm_dir) for a bar window.

    wick_frac   — defended-side wick as a fraction of bar range (-inf where range <= 0, so
                  any `>= threshold` test fails exactly like the original early return).
    confirm_any — body_threshold + delta_threshold only (absorption_delta's confirm candle).
    confirm_dir — confirm_any + correct candle direction (every other confirm/entry candle).
    """
    rng  = h - l
    body = np.abs(c - o)
    pos_rng = rng > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        body_frac = np.where(pos_rng, body / rng, 0.0)
        if direction == "long":
            wick = np.minimum(o, c) - l
        else:
            wick = h - np.maximum(o, c)
        wick_frac = np.where(pos_rng, wick / rng, -np.inf)

    body_ok = pos_rng & (body_frac >= params["body_threshold"])
    dt = params["delta_threshold"]
    if direction == "long":
        confirm_any = body_ok & (vdp >= dt)
        confirm_dir = confirm_any & (c > o)
    else:
        confirm_any = body_ok & (vdp <= -dt)
        confirm_dir = confirm_any & (c < o)
    return wick_frac, confirm_any, confirm_dir


class EntryWindow:
    """The post_retest window ([breakout bar] + retest..retest+entry_window) as positions
    into a DayData, with everything the seven finders share precomputed once."""

    def __init__(self, day: DayData, pos: np.ndarray, direction: str,
                 vah: float, val: float, params: dict):
        self.day = day
        self.pos = pos
        self.n   = len(pos)
        self.direction = direction

        self.o = o = day.open[pos]
        self.h = h = day.high[pos]
        self.l = l = day.low[pos]
        self.c = c = day.close[pos]
        self.vdp = vdp = day.vdp[pos]

        self.wick_frac, self.confirm_any, self.confirm_dir = \
            _build_masks(o, h, l, c, vdp, direction, params)

        # whole-trade invalidation: close back through VAL (long) / VAH (short)
        invalid = c < val if direction == "long" else c > vah
        self.invalid   = invalid
        self.first_inv = int(invalid.argmax()) if invalid.any() else self.n

        # baselines are gated by which finders/detectors are enabled (see process_day);
        # a None baseline means every consumer of it is switched off this run
        if direction == "long":
            ab, pb = day.sell_baseline, day.passive_baseline_long
            self.abs_base     = ab[pos] if ab is not None else None
            self.p_base       = pb[pos] if pb is not None else None
            self.best_passive = day.best_passive_long[pos] if pb is not None else None
        else:
            ab, pb = day.buy_baseline, day.passive_baseline_short
            self.abs_base     = ab[pos] if ab is not None else None
            self.p_base       = pb[pos] if pb is not None else None
            self.best_passive = day.best_passive_short[pos] if pb is not None else None

        tv_day = day.tick_volume
        self.tv = [tv_day[p] for p in pos]
        if self.p_base is not None:
            po_day = day.passive_orders
            self.po = [po_day[p] for p in pos]
        else:
            self.po = None

    def ts(self, i: int):
        """Timestamp of window bar i."""
        return self.day.index[self.pos[i]]


class TradeWindow:
    """The post_entry window (entry bar .. end of day) as a contiguous positional slice
    of a DayData — same precomputed masks as EntryWindow, minus invalidation."""

    def __init__(self, day: DayData, start: int, direction: str, params: dict):
        self.day   = day
        self.start = start
        self.n     = day.n - start
        self.direction = direction
        self.index = day.index[start:]

        self.o = o = day.open[start:]
        self.h = h = day.high[start:]
        self.l = l = day.low[start:]
        self.c = c = day.close[start:]
        self.vdp = vdp = day.vdp[start:]

        self.wick_frac, self.confirm_any, self.confirm_dir = \
            _build_masks(o, h, l, c, vdp, direction, params)

        # same gating as EntryWindow: None baseline => every consumer is off this run
        if direction == "long":
            ab, pb = day.sell_baseline, day.passive_baseline_long
            self.abs_base     = ab[start:] if ab is not None else None
            self.p_base       = pb[start:] if pb is not None else None
            self.best_passive = day.best_passive_long[start:] if pb is not None else None
        else:
            ab, pb = day.buy_baseline, day.passive_baseline_short
            self.abs_base     = ab[start:] if ab is not None else None
            self.p_base       = pb[start:] if pb is not None else None
            self.best_passive = day.best_passive_short[start:] if pb is not None else None

        self.tv = day.tick_volume[start:]
        self.po = day.passive_orders[start:] if self.p_base is not None else None

    def ts(self, i: int):
        """Timestamp of window bar i."""
        return self.index[i]


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

def prev_rolling_max(a: np.ndarray, k: int) -> np.ndarray:
    """rolling(k).max().shift(1) — prev[i] = max(a[i-k .. i-1]), NaN while fewer than k."""
    n   = len(a)
    out = np.full(n, np.nan)
    if n > k:
        from numpy.lib.stride_tricks import sliding_window_view
        out[k:] = sliding_window_view(a, k).max(axis=1)[: n - k]
    return out


def prev_rolling_min(a: np.ndarray, k: int) -> np.ndarray:
    """rolling(k).min().shift(1) — prev[i] = min(a[i-k .. i-1]), NaN while fewer than k."""
    n   = len(a)
    out = np.full(n, np.nan)
    if n > k:
        from numpy.lib.stride_tricks import sliding_window_view
        out[k:] = sliding_window_view(a, k).min(axis=1)[: n - k]
    return out
