"""
Mean-reversion / trend regime detector — labels each snapshot `reverting` /
`neutral` / `trending` (plus implicit `unknown` = "no answer", which downstream
must EXCLUDE, never bucket as a fourth regime).

Same spine as vol_realized_z: expanding today's-data snapshots, matched-window
z-scores against the SAME clock window on prior days, Schmitt-trigger
hysteresis, cross-day state chain. What differs is the measure in the middle —
and there are five of them, deliberately, so they can be watched disagreeing.

HONEST FRAMING (the plan's §0, kept here because it governs how to read the
output): this axis is weaker and shakier than volatility. Vol clusters, so its
label predicts; trending/reverting character persists much less reliably. At
the 10:00 snapshot there are 30 RTH bars — a small sample for any of these
measures. Whether the label is TRUSTWORTHY at 10:00 is an empirical question
answered by scripts/mr_regime/mr_regime_validation.py, not by this file. A weak
result there is a real finding (the axis describes rather than filters), not a
bug to fix. There is deliberately NO forecast/HAR component: prior days supply
only the yardstick, never the label.

THE MEASURES
  variance ratio (Lo–MacKinlay, overlapping blocks, hetero-robust z) — primary,
      anchor-free, most robust, matches IVB's holding horizon;
  Hurst via variance-of-differences — anchor-free. NOTE, because the plan's
      §9 prediction depends on it: this estimator is ALGEBRAICALLY the variance
      ratio family (Var of k-differences vs k·Var of 1-differences IS VR(k);
      the regression just fits a slope through the VR curve). It cannot dissent
      from VR by construction, so it is not an independent opinion;
  Hurst via rescaled range (R/S) — anchor-free, genuinely independent of VR,
      genuinely noisy on short windows. This is the "watch it be noisy"
      comparator the agreement matrix needs. Score column only, no state;
  ADF t-statistic — anchor-based, the Chan classic, data-hungry and marginal on
      intraday windows by nature;
  efficiency ratio (Kaufman) — anchor-based, no distributional assumptions,
      works where ADF and Hurst fail.

SIGN CONVENTION — positive z means MORE TRENDING THAN NORMAL, for every measure,
so the chart colours mean the same thing on every axis. ln VR > 0, H > 0.5,
ER → 1 all mean trending; and the ADF t-statistic is ALREADY monotone
increasing in trendiness (strong reversion ⇒ β ≪ 0 ⇒ t ≈ −4; random walk ⇒
t ≈ 0; momentum ⇒ t ≥ 0). It is therefore stored and z-scored AS IS — do NOT
"flip the sign for ADF because reversion is negative". Flipping is what would
make reversion read as trend. This is the single easiest column in the module
to get backwards; test_adf_sign_convention pins it.

ANCHORS ("revert to what" is part of the question, not a tweak — the same path
is trend against the session open and reversion against a rising EMA). Anchor
times are their OWN params (`anchor_open_time` / `anchor_close_time`), not
derived from rth_start/rth_end, so the anchors can be moved independently of
the snapshot session. Only ADF and ER consume an anchor; VR and Hurst don't.

Each anchor is defined over its own DOMAIN of bars, and an anchored measure's
window is the intersection of the scope window with that domain — the spread
only exists where the anchor exists. Consequence worth knowing: with the
default anchor_open_time == rth_start, gbx_*_open and rth_*_open are identical
by construction (both windows start at the anchor). They diverge as soon as
anchor_open_time is moved earlier (e.g. "18:00" to anchor at the globex open).

VWAP is computed FRESH from the candles — never the indicator dataset's VWAP,
which is pinned to 09:30 by the transform regardless of session params and
would silently anchor this detector to a fixed time.

All z-scores are matched-window (same clock time on prior days) because the raw
measures have a strong intraday pattern — the open trends, midday chops, so a
raw VR means different things at 10:00 and 14:00. VR is ratio-like and
right-skewed so its z lives in ln space; ER, ADF-t and both Hurst estimates are
roughly symmetric and z-score raw. The per-measure transform is recorded in
meta.json under script_extras.z_transforms.

Contract rolls are a non-issue: each input file is one globex session with one
contract, so every return and every spread is within-contract.
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
    "q_values": "2,5,10",             # VR horizons (q=1 rejected: VR(1) == 1)
    "q_label": 10,                    # which q drives the primary mr_state
    "anchors": "open,vwap_gbx,vwap_rth,ema,rmean",   # ADF + ER only
    "anchor_open_time": "09:30",      # `open` / `vwap_rth` anchor reset
    "anchor_close_time": "16:00",     # `vwap_rth` stops accumulating here
    "ema_n": 20,                      # `ema` anchor length
    "rmean_n": 20,                    # `rmean` anchor length
    "adf_maxlag": 1,                  # fixed-low, NOT auto-selected
    "robust_scale": True,             # median/MAD vs mean/std
    # below this many bars OR this many MOVING bars, that measure says nothing
    # (see EFFECTIVE SAMPLE in DayMeasures._one_bar_moments)
    "min_bars_for_measure": 20,
    # mirrors the vol detector's TUNED dead zone (its plan-default ±0.75/±0.35
    # left the middle state at a 65% diagonal). Retune once against the
    # transition matrix in the validation script, then stop.
    "enter_trend": 0.90,
    "enter_revert": -0.90,
    "exit_trend": 0.30,               # dead zone (Schmitt trigger)
    "exit_revert": -0.30,
    "drop_zero_volume_bars": False,
}

# Anchor vocabulary. `anchors` selects a subset; an unknown name is a hard
# error at parse time (a typo must never silently drop a whole column family).
ANCHOR_NAMES = ("open", "vwap_gbx", "vwap_rth", "ema", "rmean")

# Internal, deliberately not params — two more knobs on an already-noisy axis
# buys nothing. Hurst(VoD) regresses ln Var(k-bar returns) on ln k over these
# lags; R/S regresses ln(mean R/S) on ln(block size) over these block sizes.
HURST_LAGS = (1, 2, 4, 8)
RS_BLOCKS = (8, 16, 32, 64, 128, 256)

STATES = ["reverting", "neutral", "trending"]
# blue = reverting, grey = neutral, red = trending — same palette as the vol
# detector's low/normal/high so the two modules read the same way.
STATE_COLORS = ["#4d9de0", "#8d99ae", "#e63946"]

SCOPES = ("gbx", "rth")


def _state_columns(anchors) -> list[str]:
    """Every regime column this run emits, in display order."""
    return (["mr_state", "mr_state_hurst"]
            + [f"mr_state_adf_{a}" for a in anchors]
            + [f"mr_state_er_{a}" for a in anchors])


def _regime_states(anchors) -> dict:
    return {col: {"states": list(STATES), "colors": list(STATE_COLORS)}
            for col in _state_columns(anchors)}


# Declared for the UI (colours, validation) from the DEFAULT params; run_all
# passes the run's actual, param-derived dicts to the runner, so narrowing
# `anchors` really does drop those columns instead of failing validation.
REGIME_STATES = _regime_states(ANCHOR_NAMES)


def _measure_table(q_values, anchors) -> list[tuple[str, str, str]]:
    """(cache key, raw column base, z column base) for every measure that gets
    a matched-window z. One table drives the cache, the columns and the labels,
    so a measure can never exist in one and not the others."""
    table = [(f"vr_q{q}", f"vr_q{q}", f"vr_z_q{q}") for q in q_values]
    table += [("hurst", "hurst", "hurst_z"),
              ("hurst_rs", "hurst_rs", "hurst_rs_z")]
    table += [(f"adf_{a}", f"adf_{a}", f"adf_z_{a}") for a in anchors]
    table += [(f"er_{a}", f"er_{a}", f"er_z_{a}") for a in anchors]
    return table


def _column_tiers(q_values, anchors) -> dict:
    # the z mr_state was actually triggered from, scope handoff already
    # applied — the one score column downstream and the validation need
    score, diagnostic = ["mr_z"], []
    for _key, raw, zed in _measure_table(q_values, anchors):
        for scope in SCOPES:
            score += [f"{scope}_mr_{raw}", f"{scope}_mr_{zed}"]
    for q in q_values:
        for scope in SCOPES:
            score.append(f"{scope}_mr_vr_lm_z_q{q}")
            diagnostic.append(f"{scope}_mr_vr_blocks_q{q}")
    for _key, raw, _z in _measure_table(q_values, anchors):
        for scope in SCOPES:
            diagnostic += [f"{scope}_hist_{raw}_center",
                           f"{scope}_hist_{raw}_scale",
                           f"{scope}_hist_{raw}_n"]
    for scope in SCOPES:
        diagnostic.append(f"{scope}_mr_n_moves")
    for a in anchors:
        diagnostic.append(f"mr_anchor_price_{a}")
        for scope in SCOPES:
            diagnostic.append(f"{scope}_mr_adf_nobs_{a}")
    diagnostic += ["diag_excluded_returns", "diag_state_chain_reset"]
    return {"regime": _state_columns(anchors), "score": score,
            "diagnostic": diagnostic}


COLUMN_TIERS = _column_tiers([2, 5, 10], ANCHOR_NAMES)

# Expectations the runner verifies by measurement (warns, never fails).
CONSTANT_COLUMNS = ["diag_state_chain_reset"]

_ONE_MIN_NS = 60_000_000_000
_NEUTRAL = "neutral"


# ── param parsing ────────────────────────────────────────────────────────────

def _parse_q_values(raw) -> list[int]:
    """'2,5,10' -> [2, 5, 10]. q=1 is REJECTED, not silently dropped: VR(1) is
    exactly 1.0 by construction (variance ÷ 1×the same variance), so asking for
    it means the caller misunderstands the measure."""
    try:
        qs = sorted({int(tok) for tok in str(raw).split(",") if tok.strip()})
    except ValueError:
        raise ValueError(f"q_values must be comma-separated ints, "
                         f"got {raw!r}") from None
    if not qs:
        raise ValueError("q_values is empty")
    if 1 in qs:
        raise ValueError("q_values may not contain 1 - VR(1) is exactly 1.0 by "
                         "construction (zero information). Use 2 or more.")
    if qs[0] < 2:
        raise ValueError(f"q_values must be >= 2, got {raw!r}")
    return qs


def _parse_anchors(raw) -> list[str]:
    """'open,ema' -> ['open', 'ema'], in ANCHOR_NAMES order. Unknown names are
    an error — a typo must not silently drop an entire column family."""
    names = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    unknown = [n for n in names if n not in ANCHOR_NAMES]
    if unknown:
        raise ValueError(f"unknown anchor(s): {', '.join(unknown)} - "
                         f"valid anchors are {', '.join(ANCHOR_NAMES)}")
    if not names:
        raise ValueError("anchors is empty - ADF and ER need at least one")
    return [a for a in ANCHOR_NAMES if a in set(names)]


# ── returns with an explicit validity mask (same rules as vol_realized_z) ─────

def _return_stats(bars: pd.DataFrame, drop_zero_volume: bool) -> dict:
    """Per-bar log returns + validity mask, as prefix sums.

    A return is EXCLUDED (a price gap, not a market move) when it is the first
    bar of the session, the previous bar is not the immediately preceding
    minute (any data gap, halt or break), either close is non-positive, or —
    optionally — the bar has zero volume. Excluded returns are stored as 0.0
    and left out of every count, so any window [a, b) costs two lookups.
    """
    n = len(bars)
    closes = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float) if "volume" in bars \
        else np.ones(n)

    r = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    if n >= 2:
        ts = bars.index.as_unit("ns").asi8          # normalize us/ns units
        consecutive = (ts[1:] - ts[:-1]) == _ONE_MIN_NS
        price_ok = (closes[1:] > 0) & (closes[:-1] > 0)
        valid[1:] = consecutive & price_ok
        if drop_zero_volume:
            valid[1:] &= volume[1:] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(np.where(price_ok, closes[1:] / closes[:-1], np.nan))
        r[1:] = np.where(valid[1:] & np.isfinite(raw), raw, 0.0)

    return {"r": r, "valid": valid,
            "cum_r": _prefix(r), "cum_r2": _prefix(np.square(r)),
            "cum_v": _prefix(valid.astype(float)),
            "cum_excl": _prefix((~valid).astype(float)),
            # bars where the price actually MOVED. A frozen tape (limit-locked
            # session, holiday-thin book) has hundreds of valid returns that
            # are all exactly zero; they are arithmetic, not evidence, and
            # every measure needs a floor on them - see EFFECTIVE SAMPLE below.
            "cum_move": _prefix((valid & (r != 0.0)).astype(float))}


def _prefix(a) -> np.ndarray:
    """Prefix sums with a leading zero: out[i] = sum(a[:i])."""
    out = np.zeros(len(a) + 1)
    np.cumsum(np.asarray(a, dtype=float), out=out[1:])
    return out


def _seg(cum: np.ndarray, a, b) -> np.ndarray:
    """cum[b] - cum[a] with a clipped into [0, b] — an empty or inverted
    window sums to zero instead of borrowing from before its own start."""
    a = np.clip(np.asarray(a), 0, None)
    b = np.asarray(b)
    a = np.minimum(a, b)
    return cum[b] - cum[a]


# ── anchors ──────────────────────────────────────────────────────────────────

def _anchor_series(bars: pd.DataFrame, name: str, pos_open: int,
                   pos_close: int, ema_n: int, rmean_n: int) -> tuple:
    """(values, domain_start) for one anchor.

    `values` is a full-length float array (NaN outside the domain) and
    `domain_start` is the first bar index at which the anchor is defined —
    which is what makes every anchored measure strictly causal: a window may
    never start before its anchor exists.

    VWAPs are computed FRESH here (see the module docstring). `vwap_rth` stops
    accumulating at anchor_close_time and holds its last value, so the
    post-close tape can't drag the RTH anchor around.
    """
    n = len(bars)
    closes = bars["close"].to_numpy(dtype=float)
    out = np.full(n, np.nan)

    if name == "open":
        if pos_open >= n:
            return out, n
        level = float(bars["open"].to_numpy(dtype=float)[pos_open])
        out[pos_open:] = level
        return out, pos_open

    if name in ("vwap_gbx", "vwap_rth"):
        typ = (bars["high"].to_numpy(dtype=float)
               + bars["low"].to_numpy(dtype=float) + closes) / 3.0
        vol = bars["volume"].to_numpy(dtype=float) if "volume" in bars \
            else np.ones(n)
        start = 0 if name == "vwap_gbx" else pos_open
        if start >= n:
            return out, n
        stop = n if name == "vwap_gbx" else min(pos_close, n)
        stop = max(stop, start + 1)
        num = np.cumsum(typ[start:stop] * vol[start:stop])
        den = np.cumsum(vol[start:stop])
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(den > 0, num / den, np.nan)
        out[start:stop] = vwap
        if stop < n:                                # frozen after the close
            out[stop:] = vwap[-1]
        return out, start

    if name == "ema":
        typ = (bars["high"].to_numpy(dtype=float)
               + bars["low"].to_numpy(dtype=float) + closes) / 3.0
        ema = pd.Series(typ).ewm(span=max(int(ema_n), 1),
                                 adjust=False).mean().to_numpy()
        start = min(max(int(ema_n) - 1, 0), n)      # skip the warm-up
        out[start:] = ema[start:]
        return out, start

    if name == "rmean":
        k = max(int(rmean_n), 1)
        rm = pd.Series(closes).rolling(k).mean().to_numpy()
        start = min(k - 1, n)
        out[start:] = rm[start:]
        return out, start

    raise ValueError(f"unknown anchor {name!r}")


# ── reference implementations (single explicit window, no prefix sums) ────────
# The run path computes these same numbers vectorized over every snapshot at
# once. Hand-rolled batched linear algebra is exactly where silent errors live,
# so the readable version stays in the file and the tests assert the two agree.

def _vr_reference(r, valid, q: int) -> tuple[float, float, int]:
    """(VR(q), Lo–MacKinlay heteroskedasticity-robust z, overlapping blocks).

    Lo–MacKinlay (1988) with OVERLAPPING q-blocks: sliding the q-window one bar
    at a time over n bars gives n−q+1 blocks where disjoint chunks would give
    n//q — on a 30-bar sample at q=10 that is 21 blocks instead of 3, which is
    the whole reason to use this estimator (its variance carries the
    overlap correction). Excluded returns are dropped, and a q-block spanning
    any excluded return is dropped with them.
    """
    r = np.asarray(r, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n = int(valid.sum())
    if n <= q or q < 2:
        return np.nan, np.nan, 0
    mu = float(r[valid].sum()) / n
    e = np.where(valid, r - mu, 0.0)
    ss1 = float(np.sum(e[valid] ** 2))
    if ss1 <= 0:
        return np.nan, np.nan, 0

    # overlapping q-blocks, only where all q constituent returns are valid
    csum = np.concatenate([[0.0], np.cumsum(r)])
    cval = np.concatenate([[0.0], np.cumsum(valid.astype(float))])
    idx = np.arange(q - 1, len(r))
    blocks = csum[idx + 1] - csum[idx + 1 - q]
    ok = (cval[idx + 1] - cval[idx + 1 - q]) == q
    nq = int(ok.sum())
    if nq < 2:
        return np.nan, np.nan, nq
    ssq = float(np.sum((blocks[ok] - q * mu) ** 2))

    m = q * nq * (1.0 - q / n)                      # LM's overlap correction
    if m <= 0:
        return np.nan, np.nan, nq
    sigma_a2 = ss1 / (n - 1)
    vr = (ssq / m) / sigma_a2 if sigma_a2 > 0 else np.nan

    # hetero-robust variance of VR−1: θ = Σ [2(q−k)/q]² δ(k)
    theta = 0.0
    for k in range(1, q):
        pair = valid[k:] & valid[:-k]
        if not pair.any():
            continue
        delta = float(np.sum(e[k:][pair] ** 2 * e[:-k][pair] ** 2)) / (ss1 ** 2)
        theta += (2.0 * (q - k) / q) ** 2 * delta * n
    lm_z = (np.sqrt(n) * (vr - 1.0) / np.sqrt(theta)
            if theta > 0 and np.isfinite(vr) else np.nan)
    return float(vr), float(lm_z), nq


def _adf_reference(s, step_valid, maxlag: int = 1) -> tuple[float, int]:
    """(ADF t-statistic of β, nobs) for Δs_t = α + β s_{t−1} + Σγ_i Δs_{t−i}.

    A reliably negative β means the spread is pulled back toward zero — mean
    reversion. The t-statistic is returned RAW: it is already monotone
    increasing in trendiness (see the module docstring's sign convention).
    `maxlag` is fixed-low by design; auto-selecting lag order on 30 bars is
    another noisy decision stacked on a noisy estimate.
    """
    s = np.asarray(s, dtype=float)
    step_valid = np.asarray(step_valid, dtype=bool)
    L = max(int(maxlag), 0)
    ds = np.full(len(s), np.nan)
    ds[1:] = np.where(step_valid[1:], s[1:] - s[:-1], np.nan)

    rows = []
    for i in range(1 + L, len(s)):
        y = ds[i]
        x = [1.0, s[i - 1]] + [ds[i - j] for j in range(1, L + 1)]
        if np.isfinite(y) and all(np.isfinite(v) for v in x):
            rows.append((y, x))
    k = 2 + L
    if len(rows) <= k:
        return np.nan, len(rows)
    y = np.array([row[0] for row in rows])
    X = np.array([row[1] for row in rows])
    xtx = X.T @ X
    try:
        beta = np.linalg.solve(xtx, X.T @ y)
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.nan, len(rows)
    rss = float(np.sum((y - X @ beta) ** 2))
    dof = len(rows) - k
    sigma2 = rss / dof if dof > 0 else np.nan
    var_b = sigma2 * inv[1, 1]
    t = beta[1] / np.sqrt(var_b) if var_b > 0 else np.nan
    return float(t), len(rows)


def _er_reference(s, step_valid) -> float:
    """Kaufman efficiency ratio of the (price − anchor) spread: net spread
    displacement over the sum of absolute spread changes. Near 1 → the spread
    went straight (trend); near 0 → it wandered and came back (chop). No
    distributional assumptions, which is why it works on tiny windows where
    ADF and Hurst don't. With a fixed `open` anchor and a window starting at
    that anchor (the default case) this is exactly |P_end − P_open| / Σ|ΔP|."""
    s = np.asarray(s, dtype=float)
    step_valid = np.asarray(step_valid, dtype=bool)
    finite = np.isfinite(s)
    if finite.sum() < 2:
        return np.nan
    steps = np.abs(np.diff(s))
    use = step_valid[1:] & np.isfinite(steps)
    denom = float(steps[use].sum())
    if denom <= 0:
        return np.nan
    first, last = np.argmax(finite), len(s) - 1 - np.argmax(finite[::-1])
    return float(abs(s[last] - s[first]) / denom)


def _hurst_rs(r, valid) -> float:
    """Hurst from rescaled range: for each block size L, split the window into
    non-overlapping blocks, take mean R/S, regress ln(mean R/S) on ln L.

    Genuinely independent of the variance-ratio family (it uses the range of
    cumulative deviations, not variances) and genuinely biased/noisy on short
    windows — its error bars swallow the third decimal at 30 bars. It is here
    to be WATCHED disagreeing, and to calibrate trust in the robust measures;
    it drives no state column.
    """
    r = np.where(np.asarray(valid, dtype=bool), np.asarray(r, dtype=float), 0.0)
    n = len(r)
    xs, ys = [], []
    for L in RS_BLOCKS:
        if L > n:
            break
        blocks = r[:(n // L) * L].reshape(-1, L)
        mean = blocks.mean(axis=1, keepdims=True)
        dev = np.cumsum(blocks - mean, axis=1)
        rng = dev.max(axis=1) - dev.min(axis=1)
        sd = blocks.std(axis=1, ddof=1)
        ok = (sd > 0) & (rng > 0)
        if not ok.any():
            continue
        xs.append(np.log(L))
        ys.append(np.log(float(np.mean(rng[ok] / sd[ok]))))
    if len(xs) < 2:
        return np.nan
    x, y = np.asarray(xs), np.asarray(ys)
    den = len(x) * float(x @ x) - float(x.sum()) ** 2
    if den <= 0:
        return np.nan
    return float((len(x) * float(x @ y) - float(x.sum()) * float(y.sum()))
                 / den)


# ── yardsticks ───────────────────────────────────────────────────────────────

def _center_scale_matrix(mat, robust: bool) -> tuple:
    """Column-wise (centers, scales, counts) for a (days × measures) matched-
    window sample. Robust: median / MAD·1.4826.

    Vectorized across measures on purpose: this is called once per snapshot
    clock per scope, and a per-measure Python loop here costs minutes over a
    4000-day history. A degenerate scale (0, or fewer than 2 observations)
    yields a NaN scale, so the z is NaN and the label is `unknown` — never a
    silently wrong number.
    """
    mat = np.atleast_2d(np.asarray(mat, dtype=float))
    counts = np.isfinite(mat).sum(axis=0)
    width = mat.shape[1]
    centers = np.full(width, np.nan)
    scales = np.full(width, np.nan)
    if mat.shape[0] == 0:
        return centers, scales, counts
    # only columns with observations — nanmedian of an all-NaN column is a
    # RuntimeWarning per call, and this runs ~90 times per day
    good = np.flatnonzero(counts > 0)
    if len(good):
        sub = mat[:, good]
        with np.errstate(invalid="ignore", divide="ignore"):
            if robust:
                centers[good] = np.nanmedian(sub, axis=0)
                scales[good] = np.nanmedian(
                    np.abs(sub - centers[good]), axis=0) * 1.4826
            else:
                centers[good] = np.nanmean(sub, axis=0)
                scales[good] = np.nanstd(sub, axis=0, ddof=1)
    enough = counts >= 2
    centers = np.where(enough, centers, np.nan)
    scales = np.where(enough & np.isfinite(scales) & (scales > 0), scales,
                      np.nan)
    return centers, scales, counts


def _center_scale(values, robust: bool) -> tuple[float, float, int]:
    """Single-sample form of _center_scale_matrix (one measure)."""
    col = np.asarray(list(values), dtype=float).reshape(-1, 1)
    centers, scales, counts = _center_scale_matrix(col, robust)
    return float(centers[0]), float(scales[0]), int(counts[0])


def _z(value: float, center: float, scale: float) -> float:
    if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale)):
        return np.nan
    return (value - center) / scale


_ER_EPS = 1e-6


def _to_z_space(key: str, value: float) -> float:
    """The per-measure transform the z lives in (recorded in meta.json).

    VR is ratio-like and right-skewed -> ln.

    ER is bounded [0, 1] and, against a fast moving anchor, PILES AT ZERO: the
    spread against a 20-bar mean barely displaces net while wandering a lot, so
    the median ER is ~0.01 with a long right tail. Z-scoring that raw produced
    a mean z of +0.4 that grew through the session (0.29 at 10:00 -> 0.84 at
    16:30) with |z| running past 100 — i.e. a broken yardstick, not a market
    fact. Logit fixes the support; the plan's §5 authorizes exactly this once
    the histogram has been looked at, and it has been.

    Hurst (both estimators) and the ADF t-statistic are roughly symmetric
    already and z-score raw. The ADF t is NOT sign-flipped — see the module
    docstring's sign convention.
    """
    if not np.isfinite(value):
        return np.nan
    if key.startswith("vr_q"):
        return float(np.log(value)) if value > 0 else np.nan
    if key.startswith("er_"):
        p = min(max(float(value), _ER_EPS), 1.0 - _ER_EPS)
        return float(np.log(p / (1.0 - p)))
    return float(value)


Z_TRANSFORMS = {"vr_q*": "ln", "hurst": "raw", "hurst_rs": "raw",
                "adf_*": "raw (t-stat, NOT sign-flipped)",
                "er_*": "logit (ER piles at 0 against moving anchors)"}


# ── Schmitt-trigger labelling ────────────────────────────────────────────────

def _schmitt(z_values, seed_state: str, enter_trend: float,
             enter_revert: float, exit_trend: float,
             exit_revert: float) -> list[str]:
    """Three-state hysteresis over a z sequence. NaN emits `unknown` and leaves
    the chain state untouched. A single step may exit one extreme AND enter the
    opposite one (z jumping +2 -> -2 goes trending -> reverting)."""
    state = seed_state if seed_state in STATES else _NEUTRAL
    out = []
    for z in z_values:
        if not np.isfinite(z):
            out.append(UNKNOWN_STATE)
            continue
        if state == "trending" and z < exit_trend:
            state = _NEUTRAL
        if state == "reverting" and z > exit_revert:
            state = _NEUTRAL
        if state == _NEUTRAL:
            if z > enter_trend:
                state = "trending"
            elif z < enter_revert:
                state = "reverting"
        out.append(state)
    return out


def _read_final_state(path) -> str | None:
    """Final mr_state of an existing output file, None when unreadable."""
    try:
        col = pd.read_parquet(path, columns=["mr_state"])["mr_state"]
        return str(col.iloc[-1]) if len(col) else None
    except Exception:  # noqa: BLE001 — any unreadable file means "no chain"
        return None


# ── the day's measures, vectorized over snapshots ────────────────────────────

class DayMeasures:
    """Every measure for one day, at every snapshot, for one scope at a time.

    All five measures are built from prefix sums, so each (scope, snapshot,
    anchor) triple costs a constant number of array lookups instead of a fresh
    fit — the difference between minutes and hours over a 4000-day history. The
    only per-snapshot loops left are R/S (needs the range of a cumulative sum)
    and the Lo–MacKinlay θ (needed once per snapshot in the output pass, never
    inside the lookback cache).
    """

    def __init__(self, bars: pd.DataFrame, cfg: dict):
        self.bars = bars
        self.cfg = cfg
        self.n = len(bars)
        self.stats = _return_stats(bars, cfg["drop_zero"])
        self.r = self.stats["r"]
        self.valid = self.stats["valid"]

        # k-bar overlapping return prefix sums, for every k any measure needs
        lags = sorted(set(cfg["q_values"]) | set(HURST_LAGS))
        self.lag = {}
        for k in lags:
            self.lag[k] = self._lag_prefixes(k)

        # anchors + their ADF/ER prefix sums
        self.anchor = {}
        for name in cfg["anchors"]:
            values, start = _anchor_series(
                bars, name, cfg["pos_anchor_open"], cfg["pos_anchor_close"],
                cfg["ema_n"], cfg["rmean_n"])
            self.anchor[name] = {"values": values, "start": start,
                                 **self._spread_prefixes(values)}

    # -- prefix builders ----------------------------------------------------
    def _lag_prefixes(self, k: int) -> dict:
        """Prefix sums of the k-bar overlapping returns R_k[i] = ln(C_i/C_{i−k}),
        valid only where all k constituent 1-bar returns are valid."""
        n, cum_r, cum_v = self.n, self.stats["cum_r"], self.stats["cum_v"]
        rk = np.zeros(n)
        vk = np.zeros(n, dtype=bool)
        if n >= k:
            i = np.arange(k - 1, n)
            rk[i] = cum_r[i + 1] - cum_r[i + 1 - k]
            vk[i] = (cum_v[i + 1] - cum_v[i + 1 - k]) == k
        rk = np.where(vk, rk, 0.0)
        return {"k": k, "sum": _prefix(rk), "sumsq": _prefix(np.square(rk)),
                "n": _prefix(vk.astype(float))}

    def _spread_prefixes(self, anchor_values: np.ndarray) -> dict:
        """Prefix sums for the ADF normal equations and the ER path length of
        the spread s = close − anchor.

        ADF row i regresses y = Δs_i on [1, s_{i−1}, Δs_{i−1}]; the row is only
        usable when every piece of it comes from a real consecutive-minute
        transition with a defined anchor. Because every quantity in X'X and X'y
        is a plain sum, one 3×3 system per snapshot falls straight out of the
        prefix arrays.
        """
        n = self.n
        s = self.bars["close"].to_numpy(dtype=float) - anchor_values
        step_ok = np.zeros(n, dtype=bool)
        step_ok[1:] = self.valid[1:] & np.isfinite(s[1:]) & np.isfinite(s[:-1])
        ds = np.zeros(n)
        ds[1:] = np.where(step_ok[1:], s[1:] - s[:-1], 0.0)

        # regression rows: need Δs_i, s_{i−1}, Δs_{i−1} all real
        row_ok = np.zeros(n, dtype=bool)
        row_ok[2:] = step_ok[2:] & step_ok[1:-1] & np.isfinite(s[1:-1])
        y = np.where(row_ok, ds, 0.0)
        # effective sample for the spread: steps where it actually moved
        step_move = step_ok & (ds != 0.0)
        row_move = row_ok & (y != 0.0)
        x2 = np.zeros(n)
        x2[2:] = np.where(row_ok[2:], s[1:-1], 0.0)
        x3 = np.zeros(n)
        x3[2:] = np.where(row_ok[2:], ds[1:-1], 0.0)

        return {
            "s": s, "step_ok": step_ok,
            "p_abs_ds": _prefix(np.where(step_ok, np.abs(ds), 0.0)),
            "p_step_move": _prefix(step_move.astype(float)),
            "p_row_move": _prefix(row_move.astype(float)),
            "p_n": _prefix(row_ok.astype(float)),
            "p_y": _prefix(y), "p_yy": _prefix(y * y),
            "p_x2": _prefix(x2), "p_x3": _prefix(x3),
            "p_x2x2": _prefix(x2 * x2), "p_x3x3": _prefix(x3 * x3),
            "p_x2x3": _prefix(x2 * x3),
            "p_yx2": _prefix(y * x2), "p_yx3": _prefix(y * x3),
        }

    # -- measures -----------------------------------------------------------
    def _one_bar_moments(self, a, ends):
        """(valid returns, mean, centred sum of squares, MOVING bars).

        EFFECTIVE SAMPLE: n1 counts valid returns, n_move counts the ones that
        weren't exactly zero. On 2020-03-16 (ES limit-locked overnight) a
        930-bar window held 14 moving bars, and the ADF read t = -84 - a number
        produced by arithmetic on a frozen tape, not evidence of reversion.
        Every measure floors on n_move as well as on bars, so a frozen window
        answers `unknown` instead of answering confidently and wrongly.
        """
        n1 = _seg(self.stats["cum_v"], a, ends)
        s1 = _seg(self.stats["cum_r"], a, ends)
        q1 = _seg(self.stats["cum_r2"], a, ends)
        n_move = _seg(self.stats["cum_move"], a, ends)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(n1 > 0, s1 / n1, np.nan)
            ss1 = q1 - n1 * mu ** 2
        return n1, mu, ss1, n_move

    def _block_variance(self, k: int, a, ends, mu, n1):
        """(variance of k-bar returns, block count) — the σ̂²_c(k) of the
        Lo–MacKinlay estimator, with the overlap correction m."""
        lag = self.lag[k]
        lo = np.asarray(a) + k - 1
        nk = _seg(lag["n"], lo, ends)
        sk = _seg(lag["sum"], lo, ends)
        qk = _seg(lag["sumsq"], lo, ends)
        with np.errstate(divide="ignore", invalid="ignore"):
            ssk = qk - 2.0 * (k * mu) * sk + (k * mu) ** 2 * nk
            m = k * nk * (1.0 - k / np.where(n1 > 0, n1, np.nan))
            var = np.where((m > 0) & (nk >= 2) & (ssk > 0), ssk / m, np.nan)
        return var, nk

    def variance_ratios(self, a, ends) -> dict:
        """{q: (VR, block count)} for every requested q, vectorized."""
        n1, mu, ss1, n_move = self._one_bar_moments(a, ends)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_a2 = np.where(n1 >= 2, ss1 / np.maximum(n1 - 1, 1), np.nan)
            sigma_a2 = np.where(sigma_a2 > 0, sigma_a2, np.nan)
        out = {}
        for q in self.cfg["q_values"]:
            var_q, nq = self._block_variance(q, a, ends, mu, n1)
            # q-aware sample floor: VR(q=10) needs 30 bars, which is exactly
            # the 10:00 RTH sample — so the snapshot IVB reads is the first one
            # that can label off it, by design rather than by accident.
            # The MOVE floor stays flat at min_bars rather than q-aware: at
            # 10:00 the median ES morning has 25-29 moving bars out of 30, so a
            # q-aware move floor would blank the label on 86% of days. Bars
            # resolve the horizon; moves supply the evidence.
            floor = max(self.cfg["min_bars"], 3 * q)
            enough = (n1 >= floor) & (n_move >= self.cfg["min_bars"])
            with np.errstate(divide="ignore", invalid="ignore"):
                # var_q is already σ̂²_c(q) (the overlap correction m divides by
                # q), so VR = σ̂²_c(q) / σ̂²_a with no further q factor.
                vr = np.where(enough, var_q / sigma_a2, np.nan)
            out[q] = (vr, nq)
        return out

    def hurst_vod(self, a, ends) -> np.ndarray:
        """Hurst from variance-of-differences: ln Var(k-bar returns) = 2H ln k
        + c, fitted over HURST_LAGS. Structurally the same family as the
        variance ratio (module docstring) — kept because the plan asks for it
        and because it summarizes the whole VR curve in one number."""
        n1, mu, _ss1, n_move = self._one_bar_moments(a, ends)
        lnk, lnv, use = [], [], []
        for k in HURST_LAGS:
            var, nk = self._block_variance(k, a, ends, mu, n1)
            with np.errstate(divide="ignore", invalid="ignore"):
                # _block_variance returns σ̂²_c(k), which is Var(R_k)/k (the
                # overlap correction m carries the k). The Hurst regression
                # needs Var(R_k) ITSELF — under a random walk that grows as k,
                # giving slope 1 and H = 0.5. Forgetting the k makes every day
                # look violently mean-reverting (H ≈ 0).
                lv = np.where(var > 0, np.log(var * k), np.nan)
            lnk.append(np.full(len(np.atleast_1d(ends)), float(np.log(k))))
            lnv.append(lv)
            use.append((nk >= 4) & np.isfinite(lv))
        x = np.vstack(lnk)
        y = np.vstack(lnv)
        m = np.vstack(use)
        x = np.where(m, x, 0.0)
        y = np.where(m, np.where(np.isfinite(y), y, 0.0), 0.0)
        cnt = m.sum(axis=0)
        sx, sy = x.sum(axis=0), y.sum(axis=0)
        sxx, sxy = (x * x).sum(axis=0), (x * y).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = cnt * sxx - sx ** 2
            slope = np.where((cnt >= 3) & (den > 0),
                             (cnt * sxy - sx * sy) / den, np.nan)
        enough = (n1 >= self.cfg["min_bars"]) & (n_move >= self.cfg["min_bars"])
        return np.where(enough, slope / 2.0, np.nan)

    def hurst_rs_series(self, a, ends) -> np.ndarray:
        """R/S Hurst per snapshot (the one measure with no O(1) form — the
        range of a cumulative sum isn't a prefix sum)."""
        ends = np.atleast_1d(ends)
        out = np.full(len(ends), np.nan)
        n1, _mu, _ss, n_move = self._one_bar_moments(a, ends)
        a0 = int(np.asarray(a).item()) if np.ndim(a) == 0 else None
        floor = self.cfg["min_bars"]
        for j, b in enumerate(ends):
            start = a0 if a0 is not None else int(np.asarray(a)[j])
            if n1[j] < floor or n_move[j] < floor or b <= start:
                continue
            out[j] = _hurst_rs(self.r[start:b], self.valid[start:b])
        return out

    def lm_z_series(self, a, ends, q: int) -> np.ndarray:
        """Lo–MacKinlay heteroskedasticity-robust significance z for VR(q).
        Only the output pass needs it (it is stored raw, never z-scored against
        history), so the O(n·q) reference path is used and the lookback cache
        stays cheap."""
        ends = np.atleast_1d(ends)
        out = np.full(len(ends), np.nan)
        a0 = int(np.asarray(a).item()) if np.ndim(a) == 0 else None
        for j, b in enumerate(ends):
            start = a0 if a0 is not None else int(np.asarray(a)[j])
            if b - start <= q:
                continue
            out[j] = _vr_reference(self.r[start:b], self.valid[start:b], q)[1]
        return out

    def adf(self, name: str, a, ends) -> tuple[np.ndarray, np.ndarray]:
        """(t-statistic, nobs) per snapshot for one anchor, from the batched
        3×3 normal equations. Returned RAW — see the sign convention."""
        p = self.anchor[name]
        ends = np.atleast_1d(ends)
        lo = np.maximum(np.asarray(a), p["start"]) + 2      # first usable row
        n = _seg(p["p_n"], lo, ends)
        sy = _seg(p["p_y"], lo, ends)
        syy = _seg(p["p_yy"], lo, ends)
        s2 = _seg(p["p_x2"], lo, ends)
        s3 = _seg(p["p_x3"], lo, ends)
        s22 = _seg(p["p_x2x2"], lo, ends)
        s33 = _seg(p["p_x3x3"], lo, ends)
        s23 = _seg(p["p_x2x3"], lo, ends)
        sy2 = _seg(p["p_yx2"], lo, ends)
        sy3 = _seg(p["p_yx3"], lo, ends)

        k = 3                                   # [1, s_{t−1}, Δs_{t−1}]
        xtx = np.empty((len(ends), k, k))
        xtx[:, 0, 0] = n
        xtx[:, 0, 1] = xtx[:, 1, 0] = s2
        xtx[:, 0, 2] = xtx[:, 2, 0] = s3
        xtx[:, 1, 1] = s22
        xtx[:, 1, 2] = xtx[:, 2, 1] = s23
        xtx[:, 2, 2] = s33
        xty = np.stack([sy, sy2, sy3], axis=1)

        t = np.full(len(ends), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            det = np.linalg.det(xtx)
        # one singular window must not take the whole day's column with it
        idx = np.flatnonzero((n > k) & np.isfinite(det) & (det != 0.0))
        if len(idx):
            with np.errstate(divide="ignore", invalid="ignore"):
                # (k,3,1) rhs: numpy>=2 reads a 2-D b as a stack of matrices
                beta = np.linalg.solve(xtx[idx], xty[idx][..., None])[..., 0]
                inv = np.linalg.inv(xtx[idx])
                rss = syy[idx] - np.einsum("ij,ij->i", beta, xty[idx])
                dof = n[idx] - k
                var_b = np.where(dof > 0, rss / dof, np.nan) * inv[:, 1, 1]
                t[idx] = np.where(var_b > 0, beta[:, 1] / np.sqrt(var_b),
                                  np.nan)
        moves = _seg(p["p_row_move"], lo, ends)
        t = np.where((n >= self.cfg["min_bars"])
                     & (moves >= self.cfg["min_bars"]), t, np.nan)
        return t, n.astype(int)

    def er(self, name: str, a, ends) -> np.ndarray:
        """Efficiency ratio per snapshot for one anchor."""
        p = self.anchor[name]
        ends = np.atleast_1d(ends)
        start = np.maximum(np.asarray(a), p["start"])
        start = np.broadcast_to(start, ends.shape)
        denom = _seg(p["p_abs_ds"], start + 1, ends)
        s = p["s"]
        last = np.clip(ends - 1, 0, self.n - 1)
        first = np.clip(start, 0, max(self.n - 1, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            disp = np.abs(s[last] - s[first])
            out = np.where((denom > 0) & (ends - start >= 2),
                           disp / np.where(denom > 0, denom, np.nan), np.nan)
        n_bars = ends - start
        moves = _seg(p["p_step_move"], start + 1, ends)
        return np.where((n_bars >= self.cfg["min_bars"])
                        & (moves >= self.cfg["min_bars"]), out, np.nan)

    # -- the whole set ------------------------------------------------------
    def all_measures(self, scope_start: int, ends) -> dict:
        """{cache key: array} for every z-scored measure, plus the diagnostics
        the output pass wants. One call serves both the lookback cache and the
        output rows, so a prior day and today are measured by identical code."""
        ends = np.atleast_1d(np.asarray(ends, dtype=int))
        out, extra = {}, {}
        for q, (vr, nq) in self.variance_ratios(scope_start, ends).items():
            out[f"vr_q{q}"] = vr
            extra[f"vr_blocks_q{q}"] = nq.astype(int)
        out["hurst"] = self.hurst_vod(scope_start, ends)
        out["hurst_rs"] = self.hurst_rs_series(scope_start, ends)
        extra["n_moves"] = _seg(self.stats["cum_move"], scope_start,
                                ends).astype(int)
        for name in self.cfg["anchors"]:
            t, nobs = self.adf(name, scope_start, ends)
            out[f"adf_{name}"] = t
            extra[f"adf_nobs_{name}"] = nobs
            out[f"er_{name}"] = self.er(name, scope_start, ends)
        return {"measures": out, "extra": extra}


# ── the run ──────────────────────────────────────────────────────────────────

def run_all(input_folder, output_folder, skip_existing, on_progress, params):
    p = {**PARAMS, **(params or {})}
    q_values = _parse_q_values(p["q_values"])
    anchors = _parse_anchors(p["anchors"])
    lookback = int(p["lookback_days"])
    min_lookback = int(p["min_lookback_days"])
    if lookback < min_lookback:
        raise ValueError(f"lookback_days ({lookback}) must be >= "
                         f"min_lookback_days ({min_lookback})")
    if int(p["adf_maxlag"]) != 1:
        # The O(1) prefix-sum ADF is derived for the fixed 3-regressor form
        # [1, s_{t−1}, Δs_{t−1}]. Refusing loudly beats silently ignoring the
        # param — and fixed-low is the honest choice intraday anyway (§2.3).
        raise ValueError(f"adf_maxlag must be 1 (got {p['adf_maxlag']}) - the "
                         f"prefix-sum ADF is derived for one augmented lag")
    minutes = int(p["snapshot_minutes"])
    robust = bool(p["robust_scale"])
    rth_start, rth_end = str(p["rth_start"]), str(p["rth_end"])
    anchor_open = str(p["anchor_open_time"])
    anchor_close = str(p["anchor_close_time"])

    # q_label fallback: never silently label off a q that wasn't asked for
    q_label = int(p["q_label"])
    q_label_fallback = None
    if q_label not in q_values:
        q_label_fallback = {"requested": q_label, "used": q_values[-1],
                            "reason": "q_label not in q_values"}
        q_label = q_values[-1]

    table = _measure_table(q_values, anchors)
    states_decl = _regime_states(anchors)
    tiers = _column_tiers(q_values, anchors)

    base_cfg = {
        "q_values": q_values, "anchors": anchors,
        "ema_n": int(p["ema_n"]), "rmean_n": int(p["rmean_n"]),
        "adf_maxlag": int(p["adf_maxlag"]),
        "min_bars": int(p["min_bars_for_measure"]),
        "drop_zero": bool(p["drop_zero_volume_bars"]),
    }

    def day_cfg(bars: pd.DataFrame, day: str) -> dict:
        """Per-day config: the clock times resolved to bar positions."""
        idx = bars.index
        tz = str(idx.tz)
        return {**base_cfg,
                "pos_anchor_open": int(idx.searchsorted(
                    pd.Timestamp(f"{day} {anchor_open}", tz=tz))),
                "pos_anchor_close": int(idx.searchsorted(
                    pd.Timestamp(f"{day} {anchor_close}", tz=tz)))}

    def scope_starts(bars: pd.DataFrame, day: str) -> dict:
        idx = bars.index
        tz = str(idx.tz)
        return {"gbx": 0,
                "rth": int(idx.searchsorted(
                    pd.Timestamp(f"{day} {rth_start}", tz=tz)))}

    measure_keys = [key for key, _raw, _z in table]

    def summarize(bars: pd.DataFrame) -> dict:
        """One prior day boiled down to the yardstick material: every measure
        at every snapshot CLOCK TIME, for both scopes.

        Keyed by clock ("10:00"), never by index, so half-days and DST line up
        — a 10:00 yardstick is built from prior 10:00s, which is what kills the
        intraday pattern (the open trends, midday chops, so a raw VR means
        different things at 10:00 and 14:00).

        Values are cached ALREADY IN Z-SPACE (ln for VR, raw for the rest) as
        one small array per clock in `table` order, so the output pass can
        stack a whole matched window and centre/scale every measure at once.
        """
        if bars.empty:
            return {scope: {} for scope in SCOPES}
        day = bars.index[-1].strftime("%Y-%m-%d")   # files are RTH-date keyed
        idx = bars.index
        grid = snapshot_grid(idx, minutes)
        if not len(grid):
            return {scope: {} for scope in SCOPES}
        ends = idx.searchsorted(grid).astype(int)
        clocks = [ts.strftime("%H:%M") for ts in grid]
        dm = DayMeasures(bars, day_cfg(bars, day))
        starts = scope_starts(bars, day)

        out = {}
        for scope in SCOPES:
            a = starts[scope]
            res = dm.all_measures(a, ends)["measures"]
            stack = np.column_stack([
                [_to_z_space(key, v) for v in np.asarray(res[key], dtype=float)]
                for key in measure_keys])
            out[scope] = {clock: stack[j] for j, clock in enumerate(clocks)
                          if ends[j] > a}           # window must be open
        return out

    ctx = RegimeContext(input_folder, output_folder, params=p,
                        script_name="mr_multi",
                        script_version=SCRIPT_VERSION,
                        script_file=__file__,
                        states=states_decl, tiers=tiers,
                        summarize=summarize, skip_existing=skip_existing,
                        on_progress=on_progress,
                        constant_columns=CONSTANT_COLUMNS)

    chain_resets: list[str] = []
    ctx.meta_extra.update({
        "primary_regime_column": "mr_state",
        "primary_measure": f"variance ratio at q={q_label} (anchor-free), "
                           f"rth_ scope after the open, gbx_ before it",
        "q_label_used": q_label,
        "q_label_fallback": q_label_fallback,
        "z_transforms": dict(Z_TRANSFORMS),
        "secondary_states": "reset to neutral each day (research columns); "
                            "only mr_state chains across days",
        "hurst_note": "hurst (variance-of-differences) is algebraically the "
                      "variance-ratio family and cannot dissent from it; "
                      "hurst_rs is the independent comparator",
        "chain_resets": chain_resets,               # live reference
    })

    prev_day: str | None = None
    prev_state: str | None = None                   # None -> read lazily

    for day in ctx.days():
        window = ctx.lookback(day)
        if ctx.should_skip(day):
            prev_day, prev_state = day, None        # chain reads the file
            continue

        # ── seed the primary state chain from yesterday's OUTPUT file ────
        if prev_day is None:
            seed, reset = _NEUTRAL, True
        else:
            state = prev_state if prev_state is not None \
                else _read_final_state(ctx.out_path(prev_day))
            if state in STATES:
                seed, reset = state, False
            else:                                   # absent file OR unknown
                seed, reset = _NEUTRAL, True
        if reset:
            chain_resets.append(day)

        bars = ctx.bars(day)
        times = ctx.snapshot_times(day)
        if not len(times) or bars.empty:
            ctx.write(day, [])
            prev_day, prev_state = day, None
            continue

        idx = bars.index
        ends = idx.searchsorted(times).astype(int)
        keys = [ts.strftime("%H:%M") for ts in times]
        dm = DayMeasures(bars, day_cfg(bars, day))
        starts = scope_starts(bars, day)
        post_open = ends > starts["rth"]

        # warm-up: an explicit "no answer", never a silently wrong number
        day_known = len(window) >= min_lookback
        summaries = list(window.values())           # date order

        # ── per-scope measures + matched-window z ───────────────────────
        cols: dict[str, np.ndarray] = {}
        z_by_key: dict[str, np.ndarray] = {}        # the label-driving z
        for scope in SCOPES:
            a = starts[scope]
            block = dm.all_measures(a, ends)
            raw, extra = block["measures"], block["extra"]

            for name, values in extra.items():
                cols[f"{scope}_mr_{name}"] = np.asarray(values)
            for q in q_values:
                cols[f"{scope}_mr_vr_lm_z_q{q}"] = dm.lm_z_series(a, ends, q)

            # matched-window yardsticks: for each snapshot clock, stack the
            # same clock on every prior day and centre/scale all measures at
            # once (see _center_scale_matrix on why this isn't a per-measure
            # loop). Everything here is in the measure's own z-space.
            width = len(measure_keys)
            centers = np.full((len(ends), width), np.nan)
            scales = np.full((len(ends), width), np.nan)
            counts = np.zeros((len(ends), width), dtype=int)
            for j, clock in enumerate(keys):
                sample = [s[scope][clock] for s in summaries
                          if clock in s.get(scope, {})]
                if not sample:
                    continue
                c, sc, n = _center_scale_matrix(np.vstack(sample), robust)
                centers[j], scales[j], counts[j] = c, sc, n

            today = np.column_stack([
                [_to_z_space(key, v)
                 for v in np.asarray(raw[key], dtype=float)]
                for key in measure_keys])
            with np.errstate(invalid="ignore", divide="ignore"):
                zed = (today - centers) / scales

            for i, (key, raw_base, z_base) in enumerate(table):
                cols[f"{scope}_mr_{raw_base}"] = np.asarray(raw[key],
                                                            dtype=float)
                cols[f"{scope}_hist_{raw_base}_center"] = centers[:, i]
                cols[f"{scope}_hist_{raw_base}_scale"] = scales[:, i]
                cols[f"{scope}_hist_{raw_base}_n"] = counts[:, i]
                cols[f"{scope}_mr_{z_base}"] = zed[:, i]
                z_by_key[f"{scope}:{key}"] = zed[:, i]

        # anchor levels at each snapshot (the last bar strictly before it)
        for name in anchors:
            values = dm.anchor[name]["values"]
            prev = np.clip(ends - 1, -1, dm.n - 1)
            cols[f"mr_anchor_price_{name}"] = np.where(
                prev >= 0, values[prev], np.nan)

        cols["diag_excluded_returns"] = _seg(
            dm.stats["cum_excl"], np.zeros_like(ends), ends).astype(int)
        cols["diag_state_chain_reset"] = np.full(len(ends), bool(reset))

        # ── labels: rth_ after the open, gbx_ before it (the same pre/post-
        #    open handoff the vol detector uses) ─────────────────────────
        def handoff(key: str) -> np.ndarray:
            z = np.where(post_open, z_by_key[f"rth:{key}"],
                         z_by_key[f"gbx:{key}"])
            return np.where(day_known, z, np.nan)

        thresholds = (float(p["enter_trend"]), float(p["enter_revert"]),
                      float(p["exit_trend"]), float(p["exit_revert"]))
        state_cols = {"mr_state": (f"vr_q{q_label}", seed),
                      "mr_state_hurst": ("hurst", _NEUTRAL)}
        for name in anchors:
            state_cols[f"mr_state_adf_{name}"] = (f"adf_{name}", _NEUTRAL)
            state_cols[f"mr_state_er_{name}"] = (f"er_{name}", _NEUTRAL)
        primary_z = handoff(f"vr_q{q_label}")
        cols["mr_z"] = primary_z
        labels = {col: _schmitt(handoff(key), seed_state, *thresholds)
                  for col, (key, seed_state) in state_cols.items()}

        rows = []
        for j, ts in enumerate(times):
            row = {"ts": ts}
            for col, values in cols.items():
                value = values[j]
                row[col] = (bool(value) if isinstance(value, np.bool_)
                            else value)
            for col, seq in labels.items():
                row[col] = seq[j]
            rows.append(row)

        ctx.write(day, rows)
        prev_day, prev_state = day, labels["mr_state"][-1]
