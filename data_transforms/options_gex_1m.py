"""Per-minute dealer-exposure (GEX/DEX/VEX/CHEX) snapshots from options-on-futures data.

Input: ANY of the three option DBN dataset folders under
``{root}/raw_dbn/{type}/{ASSET}/`` — DEFINITION, STATISTICS or TBBO. The sibling
definitions/statistics folders are located by case-insensitive schema substring
in the folder name; TBBO data itself is not read. Daily DBN files only
(``glbx-mdp3-YYYYMMDD.<schema>.dbn.zst``) — monthly-batched files are rejected
with a clear error.

Output: one parquet per trading day, rows = (minute x strike), minute-major.
Per-strike signed exposures (gex/dex/vex/chex + call/put splits) plus per-minute
scalars (spot, zero_gamma, walls, totals, oi_regime) duplicated across the
minute's strike rows (parquet RLE makes this nearly free).

Semantics (all lookahead-free — a reconstruction of what a live GEX terminal
would have shown during the session):

- IV basis     : frozen from the PREVIOUS trading day's settlements for the
                 whole session (``iv_basis='prev_settle'`` in metadata).
                 Day-D settlements publish ~16:00 ET and are NOT switched in —
                 doing so would inject a discontinuity into every greek for the
                 final hour.
- OI           : as-of publication time. CME publishes OI twice per trade date
                 (prelim ~01:00 UTC, final ~14:00 UTC, both keyed by ts_ref to
                 the PREVIOUS trade date). VERIFIED on real data: OI rows carry
                 stat_flags = 0 — prelim vs final is distinguishable ONLY by
                 publication time, so publications are clustered into bursts by
                 ts_event gaps. Within session D the OI regimes are typically:
                 final(D-2) -> prelim(D-1) at ~21:00 ET -> final(D-1) at
                 10:00 ET. Minutes map to regimes via searchsorted on the
                 regime start times.
- F            : per-minute front-month price from the existing 1m candle
                 parquet; each option is priced against ITS OWN underlying
                 month via a basis offset frozen from the previous settlements
                 (F_month(t) = F_front(t) + [settle(month) - settle(front)]).
- T            : decays per minute toward each contract's own ``expiration``
                 timestamp (which encodes the true settlement moment — 09:30 NY
                 for AM-settled quarterlies, 16:00 NY for PM weeklies), floored
                 at ``t_floor_minutes`` (0DTE guard; post-expiry minutes stay at
                 the floor rather than vanishing mid-session).
- ts_ref keying: ALL statistics rows are keyed by ts_ref (the trade date they
                 refer to), never by dissemination time — Friday finals can
                 arrive Sat/Sun (see settlement_reference.py). Per day the
                 loader scans every available stats file between adjacent
                 trading days, weekend files included.
- Charm units  : annualized (dDelta/dt per year); divide by 365 for per-day.
- Zero gamma   : per minute, total signed gamma-dollars swept over a
                 hypothetical parallel spot move grid (coarse grid, crossing
                 linearly interpolated — the profile is smooth, so 41 points
                 already give sub-tick accuracy); crossing nearest spot wins.
- Sign models  : OI is unsigned; the dealer side is an ASSUMPTION, exposed as
                 the ``sign_model`` parameter, never a constant.
- American style: quarterly/serial options (parent symbol == underlying root)
                 are American; Black-76 is European, so their greeks are an
                 approximation (flagged via exercise_style).

Asset-agnostic by construction: multiplier, strikes, expiries, underlying
months all come from the definitions file; the asset is read off the path.
Runs unchanged on any asset once its daily DBN files exist.
"""

# NOTE: no `from __future__ import annotations` here — string annotations on
# @dataclass fields crash Python 3.13's dataclasses._is_type when the module
# is loaded via the plugin loader (spec_from_file_location without sys.modules
# registration): sys.modules.get(cls.__module__) is None.
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.special import ndtr

TRANSFORM_VERSION = "1.0"

NY = "America/New_York"
PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = np.iinfo(np.int64).max
INT32_NULL = np.iinfo(np.int32).max
UNDEF_TS = np.iinfo(np.uint64).max
NS_PER_DAY = 86_400_000_000_000
NS_PER_YEAR = 365.0 * NS_PER_DAY
INV_SQRT_2PI = 0.3989422804014327
SQRT2 = math.sqrt(2.0)

# stat_type values (Databento GLBX statistics schema)
STAT_OPEN, STAT_SETTLE, STAT_LOW, STAT_HIGH = 1, 3, 4, 5
STAT_VOLUME, STAT_LOW_OFFER, STAT_HIGH_BID, STAT_OI = 6, 7, 8, 9
# stat_flags bits (CME tag 731-SettlPriceType; see settlement_reference.py)
FLAG_FINAL = 1 << 0     # final (vs preliminary)
FLAG_ACTUAL = 1 << 1    # actual (vs theoretical/CME-model)
FLAG_INTRADAY = 1 << 3  # intraday settlement, disseminated before official EOD
UPDATE_ACTION_ADDED = 1

PARAMS = {
    "futures_type": "Futures",       # folder under raw_dbn/ holding the futures data
    "futures_dataset": "",           # '' = unique 'statistics' substring match
    "futures_asset": "",             # '' = auto from definitions' underlying roots
    "candles_dataset": "",           # '' = prefer '1m_ohlcv_globex', fallback 'ohlcv'
    "risk_free_rate_pct": 4.5,       # percent (the params form is 2-decimal)
    "sign_model": "type",            # type | inverse | absolute | moneyness
    "t_floor_minutes": 1.0,          # 0DTE gamma-blowup guard
    "min_open_interest": 0,          # extra OI filter; OI==0 always dropped
    "strike_range_pct": 25.0,        # keep strikes within +-X% of prev front settle; 0 = off
    "max_dte_days": 0,               # keep contracts expiring within X days; 0 = off
    "drop_theoretical": False,       # drop settlements CME computed with their model
    "validate_parity": True,         # put-call parity cross-check of F (V1)
    "zero_gamma_range_pct": 8.0,     # +- sweep range as % of spot
    "zero_gamma_steps": 41,          # coarse grid; crossing is interpolated (sub-tick already)
    "snapshot_every_minutes": 1,
}

PARAM_SECTIONS = {
    "Data sources": ["futures_type", "futures_dataset", "futures_asset", "candles_dataset"],
    "Model": ["risk_free_rate_pct", "sign_model", "t_floor_minutes"],
    "Filters": ["min_open_interest", "strike_range_pct", "max_dte_days", "drop_theoretical"],
    "Zero gamma": ["zero_gamma_range_pct", "zero_gamma_steps"],
    "Output & validation": ["snapshot_every_minutes", "validate_parity"],
}


# ---------------------------------------------------------------------------
# Black-76
# ---------------------------------------------------------------------------

def _norm_pdf(x):
    return np.exp(-0.5 * x * x) * INV_SQRT_2PI


def d1_d2(F, K, sigma, T):
    st = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / st
    return d1, d1 - st


def black76_price(F, K, sigma, T, r, is_call):
    d1, d2 = d1_d2(F, K, sigma, T)
    disc = np.exp(-r * T)
    call = disc * (F * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def black76_greeks(F, K, sigma, T, r, is_call):
    """delta/gamma/vanna/charm sharing d1/d2/n(d1)/e^{-rT}. Charm is annualized
    dDelta/dt (calendar-time derivative; identity charm_c - charm_p = r*e^{-rT})."""
    d1, d2 = d1_d2(F, K, sigma, T)
    disc = np.exp(-r * T)
    nd1 = _norm_pdf(d1)
    Nd1 = ndtr(d1)
    delta = np.where(is_call, disc * Nd1, -disc * (1.0 - Nd1))
    gamma = disc * nd1 / (F * sigma * np.sqrt(T))
    vanna = -disc * nd1 * d2 / sigma
    charm_common = disc * nd1 * d2 / (2.0 * T)
    charm = np.where(is_call,
                     disc * r * Nd1 + charm_common,
                     -disc * r * (1.0 - Nd1) + charm_common)
    return {"delta": delta, "gamma": gamma, "vanna": vanna, "charm": charm}


def _ncdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _price_scalar(F, K, sigma, T, r, is_call) -> float:
    st = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / st
    d2 = d1 - st
    disc = math.exp(-r * T)
    if is_call:
        return disc * (F * _ncdf_scalar(d1) - K * _ncdf_scalar(d2))
    return disc * (K * _ncdf_scalar(-d2) - F * _ncdf_scalar(-d1))


def implied_vol(price_obs, F, K, T, r, is_call, lo=1e-4, hi=5.0, tol=1e-8):
    """Vectorized pre-screen + bracketed brentq per solvable row.

    NaN where T <= 0, the price has no time value (<= discounted intrinsic),
    or the price exceeds the sigma=hi bound (unbracketable). Never Newton —
    it blows up on low-vega deep-OTM contracts.
    """
    price_obs = np.asarray(price_obs, dtype="float64")
    F = np.asarray(F, dtype="float64")
    K = np.asarray(K, dtype="float64")
    T = np.asarray(T, dtype="float64")
    is_call = np.asarray(is_call, dtype=bool)

    out = np.full(price_obs.shape, np.nan)
    with np.errstate(all="ignore"):
        disc = np.exp(-r * T)
        intrinsic = disc * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
        ok = (
            np.isfinite(price_obs) & np.isfinite(F) & (F > 0) & (K > 0) & (T > 0)
            & (price_obs > intrinsic + 1e-12)
        )
        p_hi = np.where(ok, black76_price(F, K, hi, T, r, is_call), np.nan)
        ok &= price_obs < p_hi

    for i in np.nonzero(ok)[0]:
        try:
            out[i] = brentq(
                lambda s: _price_scalar(F[i], K[i], s, T[i], r, bool(is_call[i])) - price_obs[i],
                lo, hi, xtol=tol,
            )
        except ValueError:
            pass  # not bracketed after all (numerical edge) -> stays NaN
    return out


# ---------------------------------------------------------------------------
# Dealer sign models — OI is unsigned; the dealer side is an assumption.
# ---------------------------------------------------------------------------

def _sign_type(is_call, K, F):
    """Industry default: dealers long calls (+gamma), short puts (-gamma)."""
    return np.where(is_call, 1.0, -1.0)


def _sign_inverse(is_call, K, F):
    """Retail-call-speculation regimes: dealers short calls, long puts."""
    return np.where(is_call, -1.0, 1.0)


def _sign_absolute(is_call, K, F):
    """No signing — magnitude only ('where is gamma densest?')."""
    return np.ones(np.broadcast(is_call, K, F).shape, dtype="float64")


def _sign_moneyness(is_call, K, F):
    """Only the high-conviction bucket: customers buy OTM puts (protection),
    so dealers are short them (-1); everything else stays agnostic (0)."""
    otm_put = (~np.asarray(is_call, dtype=bool)) & (np.asarray(K) < np.asarray(F))
    return np.where(otm_put, -1.0, 0.0)


SIGN_MODELS = {
    "type": _sign_type,
    "inverse": _sign_inverse,
    "absolute": _sign_absolute,
    "moneyness": _sign_moneyness,
}


def get_sign_model(name: str):
    try:
        return SIGN_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown sign_model '{name}'. Available: {', '.join(sorted(SIGN_MODELS))}"
        ) from None


# ---------------------------------------------------------------------------
# Expiry / symbols
# ---------------------------------------------------------------------------

_UNDERLYING_RE = re.compile(r"^(.+?)[FGHJKMNQUVXZ]\d{1,2}$")


def underlying_root(symbol: str) -> str:
    """ESU6 -> ES, ESZ26 -> ES, CLM7 -> CL. No match -> input unchanged."""
    m = _UNDERLYING_RE.match(str(symbol).strip())
    return m.group(1) if m else str(symbol).strip()


def classify_settle_type(expiration_utc: pd.Series) -> np.ndarray:
    """'AM' (SOQ, morning) vs 'PM' (close) from the expiration's NY wall clock.
    tz conversion makes this DST-proof (verified 09:30 vs 16:00 on real data)."""
    hours = pd.DatetimeIndex(expiration_utc).tz_convert(NY).hour
    return np.where(np.asarray(hours) < 12, "AM", "PM")


def classify_exercise_style(parent: np.ndarray, uroot: np.ndarray) -> np.ndarray:
    """American iff the parent symbol equals the underlying root (the ES/NQ
    quarterly+serial series). Heuristic — documented approximation."""
    return np.where(np.asarray(parent) == np.asarray(uroot), "american", "european")


def settlement_reference_ns(ts_event_ns: np.ndarray, trade_date: str) -> int:
    """Asset-agnostic 'as-of' moment for settlement-based IV: the median
    dissemination time of the day's futures-settlement wave; falls back to
    16:00 NY on the trade date when the wave is empty."""
    ts = np.asarray(ts_event_ns, dtype="int64")
    if ts.size:
        return int(np.median(ts))
    return int(pd.Timestamp(f"{trade_date} 16:00", tz=NY).value)


def time_to_expiry_years(expiration_ns, now_ns, floor_minutes: float):
    floor_ns = float(floor_minutes) * 60e9
    dt = np.maximum(np.asarray(expiration_ns, dtype="float64") - np.asarray(now_ns, dtype="float64"),
                    floor_ns)
    return dt / NS_PER_YEAR


# ---------------------------------------------------------------------------
# Folder resolution + DBN decode
# ---------------------------------------------------------------------------

def _has_daily_files(folder: Path) -> bool:
    return any(_DAILY_STEM_RE.search(f.name) and not _MONTHLY_STEM_RE.search(f.name)
               for f in folder.glob("*.dbn*"))


def _unique_dir_match(parent: Path, needle: str) -> Path:
    hits = [d for d in sorted(parent.iterdir()) if d.is_dir() and needle in d.name.lower()]
    if len(hits) > 1:
        # daily files are the contract — a monthly-batched leftover next to the
        # daily re-download must not make the match ambiguous
        daily = [d for d in hits if _has_daily_files(d)]
        if len(daily) == 1:
            return daily[0]
    if len(hits) != 1:
        names = [d.name for d in sorted(parent.iterdir()) if d.is_dir()]
        raise ValueError(
            f"Expected exactly one folder containing '{needle}' with daily DBN files "
            f"under {parent} (found {len(hits)} matches). Folders present: {names}"
        )
    return hits[0]


def resolve_option_datasets(input_folder: str) -> dict:
    """Input may be the DEFINITION, STATISTICS or TBBO dataset folder. Walks up
    the fixed hierarchy dataset -> asset -> type -> raw_dbn -> root (never
    string-matches the root name) and finds the defs/stats siblings by schema
    substring."""
    p = Path(input_folder).resolve()
    asset_dir = p.parent
    type_dir = asset_dir.parent
    raw_dir = type_dir.parent
    if raw_dir.name.lower() != "raw_dbn":
        raise ValueError(
            f"Input folder is not at the expected raw_dbn/{{type}}/{{asset}}/{{dataset}} "
            f"depth: {p} (got '{raw_dir.name}' where 'raw_dbn' was expected)"
        )
    name = p.name.lower()
    defs_dir = p if "definition" in name else _unique_dir_match(asset_dir, "definition")
    stats_dir = p if "statistic" in name else _unique_dir_match(asset_dir, "statistic")
    return {
        "root": raw_dir.parent,
        "asset_type": type_dir.name,
        "asset": asset_dir.name,
        "defs_dir": defs_dir,
        "stats_dir": stats_dir,
    }


def resolve_futures_dirs(root: Path, futures_type: str, futures_asset: str,
                         futures_dataset: str, candles_dataset: str) -> tuple[Path, Path]:
    fut_asset_dir = root / "raw_dbn" / futures_type / futures_asset
    if not fut_asset_dir.is_dir():
        raise ValueError(f"Futures asset folder not found: {fut_asset_dir}")
    if futures_dataset:
        fut_stats = fut_asset_dir / futures_dataset
        if not fut_stats.is_dir():
            raise ValueError(f"futures_dataset folder not found: {fut_stats}")
    else:
        fut_stats = _unique_dir_match(fut_asset_dir, "statistic")

    candle_parent = root / "parquet" / futures_type / futures_asset
    if candles_dataset:
        candles = candle_parent / candles_dataset
        if not candles.is_dir():
            raise ValueError(f"candles_dataset folder not found: {candles}")
    else:
        if not candle_parent.is_dir():
            raise ValueError(f"Futures candle parent folder not found: {candle_parent}")
        try:
            candles = _unique_dir_match(candle_parent, "1m_ohlcv_globex")
        except ValueError:
            candles = _unique_dir_match(candle_parent, "ohlcv")
    return fut_stats, candles


_DAILY_STEM_RE = re.compile(r"-(\d{8})\.[a-z0-9-]+\.dbn(\.zst)?$", re.IGNORECASE)
_MONTHLY_STEM_RE = re.compile(r"-(\d{8})-(\d{8})\.", re.IGNORECASE)


def index_dbn_files(folder: Path) -> dict[str, Path]:
    """'YYYY-MM-DD' -> file. Daily files only; monthly-batched folders are
    rejected with a clear error (the user re-downloads those as daily)."""
    out: dict[str, Path] = {}
    n_monthly = 0
    for f in sorted(folder.glob("*.dbn*")):
        if _MONTHLY_STEM_RE.search(f.name):
            n_monthly += 1
            continue
        m = _DAILY_STEM_RE.search(f.name)
        if m:
            d = m.group(1)
            out[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = f
    if not out:
        extra = (f" The folder holds {n_monthly} monthly-batched files "
                 f"(...YYYYMMDD-YYYYMMDD...); daily files are required." if n_monthly else "")
        raise FileNotFoundError(f"No daily DBN files (glbx-...-YYYYMMDD.*.dbn.zst) in {folder}.{extra}")
    return out


def index_parquet_days(folder: Path) -> dict[str, Path]:
    return {f.stem: f for f in sorted(folder.glob("*.parquet"))
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.stem)}


def _ts_ref_dates(ts_ref_u64: np.ndarray) -> np.ndarray:
    """uint64 ns -> 'YYYY-MM-DD' strings; undefined sentinel -> ''."""
    valid = ts_ref_u64 != UNDEF_TS
    days = (ts_ref_u64.astype("int64") // NS_PER_DAY).astype("datetime64[D]")
    out = np.where(valid, np.datetime_as_string(days), "")
    return out


def load_stats(path: Path, stat_types: tuple[int, ...]) -> pd.DataFrame:
    """Decode a statistics DBN file via to_ndarray and mask stat_type BEFORE
    building any frame (stat 7/8 are ~92% of rows). Prices arrive int64
    fixed-point 1e-9; sentinels mapped to NaN."""
    store = db.DBNStore.from_file(str(path))
    arr = store.to_ndarray()
    arr = arr[np.isin(arr["stat_type"], stat_types)]
    price = np.where(arr["price"] == UNDEF_PRICE, np.nan, arr["price"].astype("float64") / PRICE_SCALE)
    qty = arr["quantity"].astype("float64")
    qty[arr["quantity"] == INT32_NULL] = np.nan
    return pd.DataFrame({
        "instrument_id": arr["instrument_id"].astype("int64"),
        "stat_type": arr["stat_type"].astype("int16"),
        "price": price,
        "quantity": qty,
        "stat_flags": arr["stat_flags"].astype("int16"),
        "update_action": arr["update_action"].astype("int16"),
        "ts_event": arr["ts_event"].astype("int64"),
        "ts_recv": arr["ts_recv"].astype("int64"),
        "ts_ref_date": _ts_ref_dates(arr["ts_ref"]),
    })


def load_futures_settlements(path: Path) -> pd.DataFrame:
    """Settlement rows of a futures statistics file with symbols resolved only
    for the handful of settlement instrument_ids. Spreads and intraday
    settlements excluded; per (symbol, ts_ref_date) the final wins, then last
    ts_recv."""
    store = db.DBNStore.from_file(str(path))
    arr = store.to_ndarray()
    arr = arr[arr["stat_type"] == STAT_SETTLE]
    flags = arr["stat_flags"].astype("int16")
    price = np.where(arr["price"] == UNDEF_PRICE, np.nan, arr["price"].astype("float64") / PRICE_SCALE)
    keep = (
        (arr["update_action"] == UPDATE_ACTION_ADDED)
        & np.isfinite(price)
        & ((flags & FLAG_INTRADAY) == 0)
    )
    arr, flags, price = arr[keep], flags[keep], price[keep]

    imap = db.common.symbology.InstrumentMap()
    imap.insert_metadata(store.metadata)
    date = pd.Timestamp(store.metadata.start, unit="ns").date()
    id2sym = {}
    for iid in np.unique(arr["instrument_id"]):
        try:
            sym = imap.resolve(int(iid), date)
        except Exception:
            sym = None
        id2sym[int(iid)] = sym if sym is not None else str(int(iid))

    df = pd.DataFrame({
        "symbol": [id2sym[int(i)] for i in arr["instrument_id"]],
        "price": price,
        "stat_flags": flags,
        "is_final": (flags & FLAG_FINAL) != 0,
        "is_theoretical": (flags & FLAG_ACTUAL) == 0,
        "ts_event": arr["ts_event"].astype("int64"),
        "ts_recv": arr["ts_recv"].astype("int64"),
        "ts_ref_date": _ts_ref_dates(arr["ts_ref"]),
    })
    df = df[~df["symbol"].str.contains("-", na=False)]
    df = df.sort_values(["is_final", "ts_recv"]).groupby(
        ["symbol", "ts_ref_date"], as_index=False).last()
    return df


def load_definitions(path: Path) -> pd.DataFrame:
    """Outright options only (instrument_class C/P, security_type OOF — the T/M
    spread classes double-count gamma). Dedup: last definition per instrument."""
    df = db.DBNStore.from_file(str(path)).to_df().reset_index()
    df = df[df["instrument_class"].isin(["C", "P"]) & (df["security_type"] == "OOF")]
    df = df[(df["strike_price"] > 0) & df["expiration"].notna() & (df["underlying"] != "")]
    df = df.sort_values("ts_recv").groupby("instrument_id", as_index=False).last()

    out = pd.DataFrame({
        "instrument_id": df["instrument_id"].astype("int64"),
        "raw_symbol": df["raw_symbol"].astype(str),
        "parent": df["asset"].astype(str),
        "underlying": df["underlying"].astype(str),
        "right": df["instrument_class"].astype(str),
        "strike": df["strike_price"].astype("float64"),
        "expiration_ns": pd.DatetimeIndex(df["expiration"]).as_unit("ns").asi8,
        "multiplier": df["unit_of_measure_qty"].astype("float64"),
        "min_price_increment": df["min_price_increment"].astype("float64"),
    })
    out["underlying_root"] = [underlying_root(s) for s in out["underlying"]]
    out["settle_type"] = classify_settle_type(df["expiration"])
    out["exercise_style"] = classify_exercise_style(out["parent"].to_numpy(),
                                                    out["underlying_root"].to_numpy())
    return out.reset_index(drop=True)


def dedup_settlements(stats: pd.DataFrame, ts_ref_date: str) -> pd.DataFrame:
    """One settlement per instrument for the given trade date: exclude intraday
    rows, prefer final over preliminary, then last ts_recv."""
    s = stats[(stats["stat_type"] == STAT_SETTLE)
              & (stats["ts_ref_date"] == ts_ref_date)
              & (stats["update_action"] == UPDATE_ACTION_ADDED)
              & stats["price"].notna()
              & ((stats["stat_flags"] & FLAG_INTRADAY) == 0)].copy()
    if s.empty:
        return s.assign(is_final=pd.Series(dtype=bool), is_theoretical=pd.Series(dtype=bool))
    s["is_final"] = (s["stat_flags"] & FLAG_FINAL) != 0
    s["is_theoretical"] = (s["stat_flags"] & FLAG_ACTUAL) == 0
    return s.sort_values(["is_final", "ts_recv"]).groupby("instrument_id", as_index=False).last()


# ---------------------------------------------------------------------------
# Exposures / aggregation / zero gamma
# ---------------------------------------------------------------------------

def compute_exposures(greeks: dict, oi, mult, F_own, sign) -> dict:
    """GEX = sign*gamma*OI*mult*F^2*0.01 ($ per 1% move); DEX/VEX/CHEX are
    dollar-delta-equivalents. S := F_own, the option's OWN underlying-month
    price — the only self-consistent choice since gamma = d2V/dF_month^2.
    Note DEX applies `sign` to the already-signed Black-76 delta; 'absolute'
    recovers the raw net delta."""
    w = oi * mult
    return {
        "gex": sign * greeks["gamma"] * w * F_own * F_own * 0.01,
        "dex": sign * greeks["delta"] * w * F_own,
        "vex": sign * greeks["vanna"] * w * F_own,
        "chex": sign * greeks["charm"] * w * F_own,
    }


def aggregate_by_strike(values: np.ndarray, strike_codes: np.ndarray, n_strikes: int) -> np.ndarray:
    """(n_contracts, n_minutes) -> (n_minutes, n_strikes) via bincount."""
    n, m = values.shape
    idx = strike_codes[:, None] + np.arange(m, dtype="int64")[None, :] * n_strikes
    flat = np.bincount(idx.ravel(), weights=values.ravel(), minlength=m * n_strikes)
    return flat.reshape(m, n_strikes)


def zero_gamma_curve(u_grid: np.ndarray, b: np.ndarray, c: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Aggregate signed gamma-dollars at hypothetical spot ratios u.
    Factorized: d1_i(u) = b_i + c_i*ln(u); G(u) = u * sum_i w0_i*exp(-d1^2/2)
    — one transcendental per (contract, grid) cell."""
    d1 = b[:, None] + c[:, None] * np.log(u_grid)[None, :]
    return u_grid * (w0[:, None] * np.exp(-0.5 * d1 * d1)).sum(axis=0)


def zero_gamma_level(spot: float, u_grid: np.ndarray, G: np.ndarray) -> float:
    """Linearly interpolated sign-change of G(u) nearest u=1; NaN if none.
    Handles crossings landing exactly on a grid point (sign()==0)."""
    s = np.sign(G)
    crossings = list(u_grid[s == 0.0])
    flips = np.nonzero(s[:-1] * s[1:] < 0)[0]
    if flips.size:
        u0, u1 = u_grid[flips], u_grid[flips + 1]
        g0, g1 = G[flips], G[flips + 1]
        crossings.extend(u0 - g0 * (u1 - u0) / (g1 - g0))
    if not crossings:
        return float("nan")
    u_cross = np.asarray(crossings)
    return float(spot * u_cross[np.argmin(np.abs(u_cross - 1.0))])


# ---------------------------------------------------------------------------
# Validations (V1/V2/V5/V6)
# ---------------------------------------------------------------------------

def check_parity(joined: pd.DataFrame, fut_settle_by_month: dict, r: float,
                 t_ref_ns: int, t_floor_minutes: float, n_pairs: int = 5) -> dict:
    """V1: F_implied = K + e^{rT}(C - P) on the n nearest-ATM strike pairs per
    (expiration, underlying); disagreement with the futures settle is a free
    data-quality signal (probably theoretical settlements), not a bug."""
    max_err, worst, checked = 0.0, "", 0
    for (exp_ns, month), g in joined.groupby(["expiration_ns", "underlying"]):
        F = fut_settle_by_month.get(month)
        if F is None or not np.isfinite(F):
            continue
        piv = g.pivot_table(index="strike", columns="right", values="settle_price", aggfunc="last")
        if "C" not in piv.columns or "P" not in piv.columns:
            continue
        piv = piv.dropna()
        if piv.empty:
            continue
        piv = piv.iloc[np.argsort(np.abs(piv.index.to_numpy() - F))[:n_pairs]]
        T = time_to_expiry_years(exp_ns, t_ref_ns, t_floor_minutes)
        f_implied = piv.index.to_numpy() + math.exp(r * float(T)) * (
            piv["C"].to_numpy() - piv["P"].to_numpy())
        err = float(np.median(np.abs(f_implied - F)))
        checked += 1
        if err > max_err:
            max_err, worst = err, f"{month}/{pd.Timestamp(exp_ns).date()}"
    return {"parity_max_error": max_err, "parity_worst_expiry": worst,
            "parity_expiries_checked": checked}


def pct_oi_theoretical(oi: np.ndarray, is_theoretical: np.ndarray) -> float:
    """V2: the single most important data-quality metric — how much of the OI
    sits on CME-model-derived (not market-derived) settlements."""
    total = float(np.nansum(oi))
    if total <= 0:
        return 0.0
    return float(np.nansum(np.where(is_theoretical, oi, 0.0)) / total * 100.0)


def gex_magnitude_bound(oi, mult, F, sigma, T, r) -> float:
    """V5: analytic upper bound on total |GEX| — n(d1) <= n(0), so
    sum(oi*mult*F*0.01*n(0)*e^{-rT}/(sigma*sqrt(T))) bounds any total. An
    observed total above this (x slack for intraday F drift) means a
    unit/scaling bug."""
    with np.errstate(all="ignore"):
        per = oi * mult * F * 0.01 * INV_SQRT_2PI * np.exp(-r * T) / (sigma * np.sqrt(T))
    return float(np.nansum(per))


def oi_prelim_final_consistency(prelim: pd.Series, final: pd.Series) -> dict:
    """V6: prelim vs final OI publications for the same ts_ref."""
    both = pd.concat([prelim.rename("p"), final.rename("f")], axis=1).dropna()
    if both.empty:
        return {"oi_pf_changed_pct": float("nan"), "oi_pf_abs_diff_total": 0}
    diff = (both["p"] - both["f"]).abs()
    return {
        "oi_pf_changed_pct": float((diff > 0).mean() * 100.0),
        "oi_pf_abs_diff_total": int(diff.sum()),
    }


# ---------------------------------------------------------------------------
# Day pipeline
# ---------------------------------------------------------------------------

@dataclass
class DayInputs:
    """Plain frames only — the synthetic-test seam (no filesystem access
    inside process_day)."""
    trade_date: str
    prev_date: str
    defs: pd.DataFrame           # load_definitions(defs file D)
    opt_stats: pd.DataFrame      # concat load_stats(stat 3+9) over files [prev_date..D]
    fut_settles: pd.DataFrame    # concat load_futures_settlements over the same window
    candles: pd.DataFrame        # candle parquet D (needs 'close'; tz-aware index)


OI_BURST_GAP_NS = 2 * 3600 * 10**9  # publications are tight bursts ~13h apart


def split_bursts(rows: pd.DataFrame, gap_ns: int = OI_BURST_GAP_NS) -> list[pd.DataFrame]:
    """Cluster publication rows into bursts by ts_event gaps. Real OI rows have
    stat_flags = 0, so prelim (~01:00 UTC) vs final (~14:00 UTC) publications
    of the same ts_ref are distinguishable ONLY by publication time."""
    rows = rows.sort_values("ts_event")
    ev = rows["ts_event"].to_numpy()
    if ev.size == 0:
        return []
    burst_id = np.concatenate([[0], np.cumsum(np.diff(ev) > gap_ns)])
    return [rows.iloc[burst_id == k] for k in range(int(burst_id[-1]) + 1)]


def _build_oi_waves(oi_rows: pd.DataFrame) -> list[dict]:
    """OI publications grouped into per-ts_ref bursts; wave effective time =
    max ts_event of the burst (fully published). Chronological order."""
    waves = []
    oi_rows = oi_rows[oi_rows["ts_ref_date"] != ""]
    for ref, g in oi_rows.groupby("ts_ref_date"):
        bursts = split_bursts(g)
        for k, b in enumerate(bursts):
            per_inst = b.sort_values("ts_recv").groupby("instrument_id")["quantity"].last()
            waves.append({
                "ts_ref": ref,
                "label": f"{ref}/pub{k + 1}of{len(bursts)}",
                "start_ns": int(b["ts_event"].max()),
                "oi": per_inst,
            })
    waves.sort(key=lambda w: w["start_ns"])
    return waves


def _build_oi_regimes(waves: list[dict], instrument_ids: np.ndarray,
                      session_start_ns: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fold waves published before the session start into the initial state;
    each in-session wave opens a new regime (partial updates carry the previous
    state forward). Returns (oi_matrix (n_regimes, n), regime_start_ns, labels)."""
    ids = pd.Index(instrument_ids)
    state = pd.Series(0.0, index=ids)
    regimes, starts, labels = [], [], []
    label0 = "none"
    for w in waves:
        upd = w["oi"].reindex(ids)
        if w["start_ns"] <= session_start_ns:
            state = upd.fillna(state)
            label0 = w["label"]
        else:
            if not regimes:
                regimes.append(state.to_numpy(copy=True))
                starts.append(0)
                labels.append(label0)
            state = upd.fillna(state)
            regimes.append(state.to_numpy(copy=True))
            starts.append(w["start_ns"])
            labels.append(w["label"])
    if not regimes:
        regimes.append(state.to_numpy(copy=True))
        starts.append(0)
        labels.append(label0)
    meta = [{"label": lab, "start_utc": ("" if s == 0 else str(pd.Timestamp(s, tz="UTC")))}
            for lab, s in zip(labels, starts)]
    return np.vstack(regimes), np.asarray(starts, dtype="int64"), meta


def _minute_grid(candles: pd.DataFrame, every_k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(minute_ns int64 UTC, front close ffilled, n_leading_nan_dropped).
    Candle indexes can be datetime64[us] — normalize to ns FIRST."""
    idx = pd.DatetimeIndex(candles.index)
    if idx.tz is None:
        raise ValueError("Candle parquet index must be tz-aware")
    t_ns = idx.tz_convert("UTC").as_unit("ns").asi8
    close = candles["close"].to_numpy(dtype="float64")
    close = pd.Series(close).ffill().to_numpy()
    valid_from = int(np.argmax(np.isfinite(close))) if np.isfinite(close).any() else len(close)
    n_dropped = valid_from
    t_ns, close = t_ns[valid_from:], close[valid_from:]
    if every_k > 1:
        t_ns, close = t_ns[::every_k], close[::every_k]
    return t_ns, close, n_dropped


def process_day(inputs: DayInputs, p: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Pure per-day computation. Returns (output_df, metadata, warnings)."""
    warnings: list[str] = []
    r = float(p["risk_free_rate_pct"]) / 100.0
    t_floor = float(p["t_floor_minutes"])
    sign_fn = get_sign_model(p["sign_model"])

    # --- minute grid + front month ------------------------------------------------
    t_ns, f_front, n_lead_dropped = _minute_grid(inputs.candles, int(p["snapshot_every_minutes"]))
    if t_ns.size == 0:
        raise ValueError("candle file has no usable minutes")
    session_start_ns = int(t_ns[0])

    fut_prev = inputs.fut_settles[inputs.fut_settles["ts_ref_date"] == inputs.prev_date]
    if fut_prev.empty:
        raise ValueError(f"no futures settlements with ts_ref={inputs.prev_date}")
    fut_prev = fut_prev.sort_values(["is_final", "ts_recv"]).groupby("symbol", as_index=False).last()
    settle_by_month = dict(zip(fut_prev["symbol"], fut_prev["price"]))

    # front month = month whose prev settle is nearest the session's first close
    # (data-driven; robust through roll weeks where OI lags the volume roll)
    months = list(settle_by_month)
    front_month = min(months, key=lambda mth: abs(settle_by_month[mth] - f_front[0]))
    front_settle = settle_by_month[front_month]

    # --- IV table (frozen on prev settlements) -------------------------------------
    opt_settle = dedup_settlements(inputs.opt_stats, inputs.prev_date)
    if opt_settle.empty:
        raise ValueError(f"no option settlements with ts_ref={inputs.prev_date}")
    joined = inputs.defs.merge(
        opt_settle[["instrument_id", "price", "is_final", "is_theoretical"]].rename(
            columns={"price": "settle_price", "is_final": "settle_is_final"}),
        on="instrument_id", how="inner")
    n_stats_unmatched = int(len(opt_settle) - len(joined))

    fF = joined["underlying"].map(settle_by_month)
    n_months_no_settle = int(joined.loc[fF.isna(), "underlying"].nunique())
    if n_months_no_settle:
        warnings.append(f"{n_months_no_settle} underlying month(s) without a futures settle "
                        f"(basis=0 fallback): "
                        f"{sorted(joined.loc[fF.isna(), 'underlying'].unique())[:6]}")
    joined["F_settle"] = fF.fillna(front_settle)
    joined["basis"] = joined["F_settle"] - front_settle

    t_ref_ns = settlement_reference_ns(
        fut_prev.loc[fut_prev["is_final"], "ts_event"].to_numpy()
        if fut_prev["is_final"].any() else fut_prev["ts_event"].to_numpy(),
        inputs.prev_date)

    # --- OI regimes (as-of publication) --------------------------------------------
    oi_rows = inputs.opt_stats[inputs.opt_stats["stat_type"] == STAT_OI]
    waves = _build_oi_waves(oi_rows)
    oi_matrix, regime_start_ns, regime_meta = _build_oi_regimes(
        waves, joined["instrument_id"].to_numpy(), session_start_ns)

    # V6 on the first vs last publication burst of the prev trade date
    # (prelim ~01:00 UTC vs final ~14:00 UTC — flags carry no distinction)
    v6 = {"oi_pf_changed_pct": float("nan"), "oi_pf_abs_diff_total": 0}
    prev_waves = [w for w in waves if w["ts_ref"] == inputs.prev_date]
    if len(prev_waves) >= 2:
        v6 = oi_prelim_final_consistency(prev_waves[0]["oi"], prev_waves[-1]["oi"])

    # --- contract filters (before the IV solve — no wasted solves) ------------------
    oi_max = oi_matrix.max(axis=0)
    keep = oi_max >= max(1.0, float(p["min_open_interest"]))
    n_no_oi = int((~keep).sum())
    if float(p["strike_range_pct"]) > 0:
        lo = front_settle * (1 - p["strike_range_pct"] / 100.0)
        hi = front_settle * (1 + p["strike_range_pct"] / 100.0)
        keep &= joined["strike"].to_numpy() >= lo
        keep &= joined["strike"].to_numpy() <= hi
    if float(p["max_dte_days"]) > 0:
        dte = (joined["expiration_ns"].to_numpy() - t_ref_ns) / NS_PER_DAY
        keep &= dte <= float(p["max_dte_days"])
    if p["drop_theoretical"]:
        keep &= ~joined["is_theoretical"].to_numpy()

    pct_theo = pct_oi_theoretical(oi_max, joined["is_theoretical"].to_numpy())

    parity = {"parity_max_error": float("nan"), "parity_expiries_checked": 0,
              "parity_worst_expiry": ""}
    if p["validate_parity"]:
        parity = check_parity(joined, settle_by_month, r, t_ref_ns, t_floor)
        rel = parity["parity_max_error"] / front_settle if front_settle else 0.0
        if parity["parity_expiries_checked"] and rel > 0.002:
            warnings.append(f"put-call parity max error {parity['parity_max_error']:.2f} "
                            f"({rel * 100:.2f}% of F) at {parity['parity_worst_expiry']} — "
                            f"settlements there are probably theoretical")

    joined = joined.loc[keep].reset_index(drop=True)
    oi_matrix = oi_matrix[:, keep]
    if joined.empty:
        raise ValueError(f"no contracts survived filters (dropped {n_no_oi} with no OI)")

    # --- IV solve --------------------------------------------------------------------
    is_call = (joined["right"] == "C").to_numpy()
    T0 = time_to_expiry_years(joined["expiration_ns"].to_numpy(), t_ref_ns, t_floor)
    iv = implied_vol(joined["settle_price"].to_numpy(), joined["F_settle"].to_numpy(),
                     joined["strike"].to_numpy(), T0, r, is_call)
    iv_ok = np.isfinite(iv)
    n_iv_failed = int((~iv_ok).sum())
    oi_iv_failed = float(oi_matrix.max(axis=0)[~iv_ok].sum())
    oi_total = float(oi_matrix.max(axis=0).sum())
    if oi_total > 0 and oi_iv_failed / oi_total > 0.10:
        warnings.append(f"IV solve failed on {oi_iv_failed / oi_total * 100:.1f}% of OI "
                        f"({n_iv_failed} contracts) — aggregate GEX may be biased")

    joined = joined.loc[iv_ok].reset_index(drop=True)
    oi_matrix = oi_matrix[:, iv_ok]
    iv = iv[iv_ok]
    is_call = is_call[iv_ok]
    T0 = T0[iv_ok]
    if joined.empty:
        raise ValueError("no contracts with a solvable IV")

    # --- per-minute engine -------------------------------------------------------------
    n = len(joined)
    m = t_ns.size
    K = joined["strike"].to_numpy()
    mult = joined["multiplier"].to_numpy()
    basis = joined["basis"].to_numpy()
    exp_ns = joined["expiration_ns"].to_numpy()
    strikes_unique = np.unique(K)
    n_strikes = strikes_unique.size
    strike_codes = np.searchsorted(strikes_unique, K)
    call_mask = is_call.astype("float64")
    put_mask = 1.0 - call_mask

    regime_idx = np.clip(np.searchsorted(regime_start_ns, t_ns, side="right") - 1,
                         0, oi_matrix.shape[0] - 1)

    u_grid = np.linspace(1 - p["zero_gamma_range_pct"] / 100.0,
                         1 + p["zero_gamma_range_pct"] / 100.0,
                         int(p["zero_gamma_steps"]))

    agg = {k: np.empty((m, n_strikes)) for k in
           ("gex", "dex", "vex", "chex", "call_gex", "put_gex", "call_oi", "put_oi")}
    zg = np.empty(m)
    n_zg_missing = 0

    chunk = 256
    floor_ns = t_floor * 60e9
    for a in range(0, m, chunk):
        bnd = slice(a, min(a + chunk, m))
        tc = t_ns[bnd]
        mc = tc.size
        F = f_front[bnd][None, :] + basis[:, None]                     # (n, mc)
        T = np.maximum(exp_ns[:, None].astype("float64") - tc[None, :].astype("float64"),
                       floor_ns) / NS_PER_YEAR
        sig = iv[:, None]
        st = sig * np.sqrt(T)
        d1 = (np.log(F / K[:, None]) + 0.5 * sig * sig * T) / st
        d2 = d1 - st
        disc = np.exp(-r * T)
        nd1 = _norm_pdf(d1)
        Nd1 = ndtr(d1)
        icm = is_call[:, None]
        delta = np.where(icm, disc * Nd1, -disc * (1.0 - Nd1))
        gamma = disc * nd1 / (F * st)
        vanna = -disc * nd1 * d2 / sig
        charm = np.where(icm, disc * r * Nd1, -disc * r * (1.0 - Nd1)) + disc * nd1 * d2 / (2.0 * T)

        oi_t = oi_matrix[regime_idx[bnd], :].T                          # (n, mc)
        sign = sign_fn(icm, K[:, None], F)
        expo = compute_exposures({"delta": delta, "gamma": gamma, "vanna": vanna, "charm": charm},
                                 oi_t, mult[:, None], F, sign)
        gex_mag = gamma * oi_t * mult[:, None] * F * F * 0.01

        per_strike_inputs = {
            "gex": expo["gex"], "dex": expo["dex"], "vex": expo["vex"], "chex": expo["chex"],
            "call_gex": gex_mag * call_mask[:, None],
            "put_gex": gex_mag * put_mask[:, None],
            "call_oi": oi_t * call_mask[:, None],
            "put_oi": oi_t * put_mask[:, None],
        }
        for key, vals in per_strike_inputs.items():
            agg[key][bnd] = aggregate_by_strike(vals, strike_codes, n_strikes)

        # zero gamma per minute (factorized profile; see zero_gamma_curve)
        w0 = sign * oi_t * mult[:, None] * 0.01 * disc * F * INV_SQRT_2PI / st
        c_fac = 1.0 / st
        for j in range(mc):
            G = zero_gamma_curve(u_grid, d1[:, j], c_fac[:, j], w0[:, j])
            z = zero_gamma_level(float(f_front[bnd][j]), u_grid, G)
            zg[a + j] = z
            if not np.isfinite(z):
                n_zg_missing += 1

    totals = {k: agg[k].sum(axis=1) for k in ("gex", "dex", "vex", "chex")}
    call_wall = strikes_unique[np.argmax(agg["call_gex"], axis=1)]
    put_wall = strikes_unique[np.argmax(agg["put_gex"], axis=1)]

    # V5 magnitude anchor (x1.5 slack for intraday F drift vs the settle-based bound)
    bound = gex_magnitude_bound(oi_matrix.max(axis=0), mult, joined["F_settle"].to_numpy(),
                                iv, T0, r)
    max_abs_gex = float(np.abs(totals["gex"]).max())
    if not np.isfinite(max_abs_gex) or (bound > 0 and max_abs_gex > bound * 1.5):
        warnings.append(f"V5 magnitude check FAILED: max |total_gex| {max_abs_gex:.3e} vs "
                        f"analytic bound {bound:.3e} — suspect multiplier/scaling bug")

    # --- assemble output ---------------------------------------------------------------
    ts_index = pd.DatetimeIndex(np.repeat(t_ns, n_strikes), tz="UTC").tz_convert(NY)
    ts_index.name = "timestamp"
    df = pd.DataFrame({
        "strike": np.tile(strikes_unique, m),
        "gex": agg["gex"].ravel().astype("float32"),
        "dex": agg["dex"].ravel().astype("float32"),
        "vex": agg["vex"].ravel().astype("float32"),
        "chex": agg["chex"].ravel().astype("float32"),
        "call_gex": agg["call_gex"].ravel().astype("float32"),
        "put_gex": agg["put_gex"].ravel().astype("float32"),
        "call_oi": np.round(agg["call_oi"]).ravel().astype("int32"),
        "put_oi": np.round(agg["put_oi"]).ravel().astype("int32"),
        "spot": np.repeat(f_front, n_strikes),
        "zero_gamma": np.repeat(zg, n_strikes),
        "call_wall": np.repeat(call_wall, n_strikes),
        "put_wall": np.repeat(put_wall, n_strikes),
        "total_gex": np.repeat(totals["gex"], n_strikes),
        "total_dex": np.repeat(totals["dex"], n_strikes),
        "total_vex": np.repeat(totals["vex"], n_strikes),
        "total_chex": np.repeat(totals["chex"], n_strikes),
        "oi_regime": np.repeat(regime_idx.astype("int8"), n_strikes),
    }, index=ts_index)

    meta = {
        "trade_date": inputs.trade_date,
        "iv_basis": "prev_settle",
        "iv_basis_date": inputs.prev_date,
        "front_month": front_month,
        "sign_model": p["sign_model"],
        "risk_free_rate_pct": p["risk_free_rate_pct"],
        "spot_open": float(f_front[0]),
        "spot_close": float(f_front[-1]),
        "zero_gamma_close": float(zg[-1]),
        "call_wall_close": float(call_wall[-1]),
        "put_wall_close": float(put_wall[-1]),
        "total_gex_close": float(totals["gex"][-1]),
        "n_contracts": n,
        "n_strikes": int(n_strikes),
        "n_minutes": int(m),
        "n_leading_candles_dropped": int(n_lead_dropped),
        "n_iv_failed": n_iv_failed,
        "oi_iv_failed": oi_iv_failed,
        "n_no_oi": n_no_oi,
        "n_stats_unmatched": n_stats_unmatched,
        "n_months_no_settle": n_months_no_settle,
        "n_minutes_no_zero_gamma": n_zg_missing,
        "pct_oi_theoretical": pct_theo,
        "basis_months": json.dumps({k: round(v - front_settle, 4)
                                    for k, v in settle_by_month.items()}),
        "oi_regimes": json.dumps(regime_meta),
        "oi_pf_changed_pct": v6["oi_pf_changed_pct"],
        "oi_pf_abs_diff_total": v6["oi_pf_abs_diff_total"],
        "parity_max_error": parity["parity_max_error"],
        "parity_worst_expiry": parity["parity_worst_expiry"],
        "parity_expiries_checked": parity["parity_expiries_checked"],
        "gex_magnitude_bound": bound,
        "transform_version": TRANSFORM_VERSION,
        "params_json": json.dumps(p, sort_keys=True),
    }
    return df, meta, warnings


def write_day(df: pd.DataFrame, meta: dict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    md = dict(table.schema.metadata or {})
    for k, v in meta.items():
        md[str(k).encode()] = str(v).encode()
    pq.write_table(table.replace_schema_metadata(md), out_file, compression="zstd")


# ---------------------------------------------------------------------------
# run_all — the Data Formatter contract
# ---------------------------------------------------------------------------

class _StatsCache:
    """Tiny LRU for decoded per-file frames — consecutive days share files."""

    def __init__(self, loader, maxlen: int = 8):
        self._loader = loader
        self._maxlen = maxlen
        self._store: dict = {}

    def get(self, key, path):
        if key not in self._store:
            if len(self._store) >= self._maxlen:
                self._store.pop(next(iter(self._store)))
            self._store[key] = self._loader(path)
        return self._store[key]


def auto_futures_asset(defs_dir: Path, defs_index: dict[str, Path]) -> str:
    """Mode of the underlying roots in the latest definitions file (handles
    LO-options-on-CL-futures style folder-name mismatches)."""
    latest = defs_index[max(defs_index)]
    defs = load_definitions(latest)
    if defs.empty:
        raise ValueError(f"no outright options in {latest}")
    return defs["underlying_root"].mode().iloc[0]


def run_all(input_folder, output_folder, skip_existing=True, on_progress=None, params=None):
    p = {**PARAMS, **(params or {})}
    on_progress = on_progress or (lambda *a: None)

    get_sign_model(p["sign_model"])  # fail fast with the available-model list

    ds = resolve_option_datasets(input_folder)
    defs_index = index_dbn_files(ds["defs_dir"])
    stats_index = index_dbn_files(ds["stats_dir"])

    futures_asset = p["futures_asset"] or auto_futures_asset(ds["defs_dir"], defs_index)
    fut_stats_dir, candles_dir = resolve_futures_dirs(
        ds["root"], p["futures_type"], futures_asset, p["futures_dataset"], p["candles_dataset"])
    fut_index = index_dbn_files(fut_stats_dir)
    candle_index = index_parquet_days(candles_dir)

    trading_days = sorted(set(stats_index) & set(candle_index) & set(defs_index))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    total = len(trading_days)
    if not total:
        on_progress(1, 1, "ERROR: no dates covered by options statistics, definitions "
                          "AND futures candles simultaneously")
        return

    on_progress(0, total,
                f"[SETUP] asset={ds['asset']} futures_asset={futures_asset} "
                f"defs={ds['defs_dir'].name} stats={ds['stats_dir'].name} "
                f"fut_stats={fut_stats_dir.name} candles={candles_dir.name} "
                f"days={total} ({trading_days[0]} .. {trading_days[-1]})")

    opt_cache = _StatsCache(lambda path: load_stats(path, (STAT_SETTLE, STAT_OI)))
    fut_cache = _StatsCache(load_futures_settlements)

    stats_dates = sorted(stats_index)
    for i, date in enumerate(trading_days):
        out_file = output_path / f"{date}.parquet"
        if skip_existing and out_file.exists():
            on_progress(i + 1, total, f"[SKIP] {date} (exists)")
            continue

        prev_candidates = [d for d in trading_days if d < date]
        if not prev_candidates:
            on_progress(i + 1, total, f"[SKIP] {date}: no previous trading day (IV basis)")
            continue
        prev_date = prev_candidates[-1]

        window = [d for d in stats_dates if prev_date <= d <= date]
        fut_window = [d for d in sorted(fut_index) if prev_date <= d <= date]
        if not fut_window:
            on_progress(i + 1, total,
                        f"[SKIP] {date}: no futures statistics file in [{prev_date}..{date}] "
                        f"under {fut_stats_dir}")
            continue

        try:
            inputs = DayInputs(
                trade_date=date,
                prev_date=prev_date,
                defs=load_definitions(defs_index[date]),
                opt_stats=pd.concat([opt_cache.get(d, stats_index[d]) for d in window],
                                    ignore_index=True),
                fut_settles=pd.concat([fut_cache.get(d, fut_index[d]) for d in fut_window],
                                      ignore_index=True),
                candles=pd.read_parquet(candle_index[date], columns=["close"]),
            )
            df, meta, warns = process_day(inputs, p)
            meta["asset"] = ds["asset"]
            meta["futures_asset"] = futures_asset
            for w in warns:
                on_progress(i + 1, total, f"[WARN] {date}: {w}")
            write_day(df, meta, out_file)
            on_progress(i + 1, total,
                        f"[DONE] {date}: {meta['n_contracts']} contracts, "
                        f"{meta['n_strikes']} strikes, spot={meta['spot_close']:.2f}, "
                        f"zg={meta['zero_gamma_close']:.2f}, "
                        f"total_gex={meta['total_gex_close']:.3e}")
        except Exception as e:  # noqa: BLE001 — per-day isolation, cancellation re-raises below
            msg = f"[ERROR] {date}: {e}"
            on_progress(i + 1, total, msg)  # a cancel raised in here propagates out
