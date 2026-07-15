"""Per-contract end-of-day options greeks/exposures table.

One row per outright option per trading day: IV (backed out from the day's own
settlement via Black-76), delta/gamma/vanna/charm, signed GEX/DEX/VEX/CHEX,
open interest, cleared volume, settlement flags, session OHL and quote
extremes. Enables skew, term structure and the strike x expiry heatmap.

Shares ALL machinery (Black-76, IV solver, sign models, expiry, folder
resolution, DBN decode, validations) with data_transforms/options_gex_1m.py —
see that module's docstring for the semantics and data-layout contracts.

EOD-specific semantics:
- Settlement, session O/H/L, quote extremes: rows with ts_ref == D (the day's
  own settlement — this is a hindsight research table, unlike the
  lookahead-free 1m snapshots).
- OI + cleared volume for trade date D publish the NEXT morning, so they are
  taken from the following file(s) with ts_ref == D (final preferred, prelim
  flagged via oi_is_final=False). If no later file exists yet (most recent
  day), the previous date's OI is used and flagged oi_is_stale=True.
- Session extremes (lowest offer / highest bid, stat 7/8) are informational
  columns ONLY — they are session extremes, can be crossed, and are never used
  as a price input. The only EOD price source is the settlement.
- Input folder: any of DEFINITION / STATISTICS / TBBO (siblings resolved by
  schema substring). Daily DBN files only.
"""

# NOTE: no `from __future__ import annotations` — see options_gex_1m.py (the
# plugin loader does not register modules in sys.modules; string dataclass
# annotations crash dataclasses._is_type on Python 3.13).
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_transforms.options_gex_1m import (  # noqa: E402
    FLAG_FINAL,
    NS_PER_DAY,
    STAT_HIGH,
    STAT_HIGH_BID,
    STAT_LOW,
    STAT_LOW_OFFER,
    STAT_OI,
    STAT_OPEN,
    STAT_SETTLE,
    STAT_VOLUME,
    TRANSFORM_VERSION,
    _StatsCache,
    auto_futures_asset,
    black76_greeks,
    check_parity,
    compute_exposures,
    dedup_settlements,
    get_sign_model,
    implied_vol,
    index_dbn_files,
    load_definitions,
    load_futures_settlements,
    load_stats,
    pct_oi_theoretical,
    resolve_futures_dirs,
    resolve_option_datasets,
    settlement_reference_ns,
    split_bursts,
    time_to_expiry_years,
    write_day,
)

NY = "America/New_York"

PARAMS = {
    "futures_type": "Futures",
    "futures_dataset": "",
    "futures_asset": "",
    "risk_free_rate_pct": 4.5,
    "sign_model": "type",
    "t_floor_minutes": 1.0,
    "min_open_interest": 0,
    "drop_theoretical": False,
    "validate_parity": True,
}

PARAM_SECTIONS = {
    "Data sources": ["futures_type", "futures_dataset", "futures_asset"],
    "Model": ["risk_free_rate_pct", "sign_model", "t_floor_minutes"],
    "Filters & validation": ["min_open_interest", "drop_theoretical", "validate_parity"],
}

_EOD_STAT_TYPES = (STAT_OPEN, STAT_SETTLE, STAT_LOW, STAT_HIGH,
                   STAT_VOLUME, STAT_LOW_OFFER, STAT_HIGH_BID, STAT_OI)
_NEXT_STAT_TYPES = (STAT_SETTLE, STAT_VOLUME, STAT_OI)


@dataclass
class EodInputs:
    """Plain frames only — the synthetic-test seam."""
    trade_date: str
    defs: pd.DataFrame            # load_definitions(defs file D)
    opt_stats: pd.DataFrame       # concat load_stats over files [D .. D_next window]
    fut_settles: pd.DataFrame     # concat load_futures_settlements over the same window


def _last_per_instrument(stats: pd.DataFrame, stat_type: int, ts_ref_date: str) -> pd.Series:
    s = stats[(stats["stat_type"] == stat_type) & (stats["ts_ref_date"] == ts_ref_date)]
    if s.empty:
        return pd.Series(dtype="float64")
    return s.sort_values("ts_recv").groupby("instrument_id")["price"].last()


def _select_oi(stats: pd.DataFrame, trade_date: str) -> tuple[pd.Series, bool, str]:
    """OI for trade date D: last publication burst with ts_ref==D. Real OI rows
    carry stat_flags = 0 — prelim (~01:00 UTC) vs final (~14:00 UTC) is
    distinguishable only by publication time, so finality is inferred: a final
    flag if present, else >=2 bursts means the last one is CME's final figure.
    If ts_ref==D is entirely absent (no next-day file yet), fall back to the
    latest earlier ts_ref present, flagged stale."""
    oi_rows = stats[(stats["stat_type"] == STAT_OI) & (stats["ts_ref_date"] != "")]
    on_date = oi_rows[oi_rows["ts_ref_date"] == trade_date]
    if not on_date.empty:
        bursts = split_bursts(on_date)
        last = bursts[-1]
        is_final = bool(((last["stat_flags"] & FLAG_FINAL) != 0).any()) or len(bursts) >= 2
        return (last.sort_values("ts_recv").groupby("instrument_id")["quantity"].last(),
                is_final, "next_final" if is_final else "next_prelim")
    earlier = oi_rows[oi_rows["ts_ref_date"] < trade_date]
    if earlier.empty:
        return pd.Series(dtype="float64"), False, "none"
    sub = earlier[earlier["ts_ref_date"] == earlier["ts_ref_date"].max()]
    return (split_bursts(sub)[-1].sort_values("ts_recv").groupby("instrument_id")["quantity"].last(),
            False, "stale_prev")


def process_day_eod(inputs: EodInputs, p: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Pure per-day computation. Returns (output_df, metadata, warnings)."""
    warnings: list[str] = []
    r = float(p["risk_free_rate_pct"]) / 100.0
    t_floor = float(p["t_floor_minutes"])
    sign_fn = get_sign_model(p["sign_model"])
    D = inputs.trade_date

    fut = inputs.fut_settles[inputs.fut_settles["ts_ref_date"] == D]
    if fut.empty:
        raise ValueError(f"no futures settlements with ts_ref={D}")
    fut = fut.sort_values(["is_final", "ts_recv"]).groupby("symbol", as_index=False).last()
    settle_by_month = dict(zip(fut["symbol"], fut["price"]))

    opt_settle = dedup_settlements(inputs.opt_stats, D)
    if opt_settle.empty:
        raise ValueError(f"no option settlements with ts_ref={D}")
    joined = inputs.defs.merge(
        opt_settle[["instrument_id", "price", "is_final", "is_theoretical"]].rename(
            columns={"price": "settle_price", "is_final": "settle_is_final",
                     "is_theoretical": "settle_is_theoretical"}),
        on="instrument_id", how="inner")
    n_stats_unmatched = int(len(opt_settle) - len(joined))

    fF = joined["underlying"].map(settle_by_month)
    n_months_no_settle = int(joined.loc[fF.isna(), "underlying"].nunique())
    if n_months_no_settle:
        warnings.append(f"{n_months_no_settle} underlying month(s) without a futures settle "
                        f"— their contracts are dropped")
    joined["F_month"] = fF
    joined = joined[joined["F_month"].notna()].reset_index(drop=True)
    if joined.empty:
        raise ValueError("no contracts with a priced underlying month")

    t_ref_ns = settlement_reference_ns(
        fut.loc[fut["is_final"], "ts_event"].to_numpy()
        if fut["is_final"].any() else fut["ts_event"].to_numpy(), D)

    # OI / cleared volume (ts_ref == D, published next day) + session stats (ts_ref == D)
    oi, oi_is_final, oi_source = _select_oi(inputs.opt_stats, D)
    if oi_source == "stale_prev":
        warnings.append("no OI with ts_ref=trade_date yet (next-day file missing) — "
                        "using previous date's OI, flagged oi_is_stale")
    elif oi_source == "none":
        warnings.append("no OI publications found at all — oi column is 0")
    vol_rows = inputs.opt_stats[(inputs.opt_stats["stat_type"] == STAT_VOLUME)
                                & (inputs.opt_stats["ts_ref_date"] == D)]
    volume = (vol_rows.sort_values("ts_recv").groupby("instrument_id")["quantity"].last()
              if not vol_rows.empty else pd.Series(dtype="float64"))

    ids = joined["instrument_id"]
    joined["oi"] = ids.map(oi).fillna(0.0)
    joined["oi_is_final"] = bool(oi_is_final)
    joined["oi_is_stale"] = oi_source == "stale_prev"
    joined["cleared_volume"] = ids.map(volume)
    joined["session_open"] = ids.map(_last_per_instrument(inputs.opt_stats, STAT_OPEN, D))
    joined["session_high"] = ids.map(_last_per_instrument(inputs.opt_stats, STAT_HIGH, D))
    joined["session_low"] = ids.map(_last_per_instrument(inputs.opt_stats, STAT_LOW, D))
    joined["lowest_offer"] = ids.map(_last_per_instrument(inputs.opt_stats, STAT_LOW_OFFER, D))
    joined["highest_bid"] = ids.map(_last_per_instrument(inputs.opt_stats, STAT_HIGH_BID, D))

    pct_theo = pct_oi_theoretical(joined["oi"].to_numpy(),
                                  joined["settle_is_theoretical"].to_numpy())

    parity = {"parity_max_error": float("nan"), "parity_expiries_checked": 0,
              "parity_worst_expiry": ""}
    if p["validate_parity"]:
        parity = check_parity(joined, settle_by_month, r, t_ref_ns, t_floor)

    # filters
    keep = joined["oi"].to_numpy() >= float(p["min_open_interest"])
    if p["drop_theoretical"]:
        keep &= ~joined["settle_is_theoretical"].to_numpy()
    joined = joined.loc[keep].reset_index(drop=True)
    if joined.empty:
        raise ValueError("no contracts survived filters")

    # IV + greeks at the day's settlement snapshot
    is_call = (joined["right"] == "C").to_numpy()
    F = joined["F_month"].to_numpy()
    K = joined["strike"].to_numpy()
    T = time_to_expiry_years(joined["expiration_ns"].to_numpy(), t_ref_ns, t_floor)
    iv = implied_vol(joined["settle_price"].to_numpy(), F, K, T, r, is_call)
    n_iv_failed = int((~np.isfinite(iv)).sum())

    greeks = {k: np.where(np.isfinite(iv), v, np.nan) for k, v in
              black76_greeks(F, K, np.where(np.isfinite(iv), iv, 1.0), T, r, is_call).items()}
    sign = sign_fn(is_call, K, F)
    expo = compute_exposures(greeks, joined["oi"].to_numpy(), joined["multiplier"].to_numpy(),
                             F, sign)

    exp_idx = pd.DatetimeIndex(joined["expiration_ns"], tz="UTC").tz_convert(NY)
    exp_idx.name = "expiration"
    df = pd.DataFrame({
        "instrument_id": joined["instrument_id"].to_numpy(),
        "symbol": joined["raw_symbol"].to_numpy(),
        "parent": joined["parent"].to_numpy(),
        "underlying": joined["underlying"].to_numpy(),
        "underlying_root": joined["underlying_root"].to_numpy(),
        "right": joined["right"].to_numpy(),
        "strike": joined["strike"].to_numpy(),
        "settle_type": joined["settle_type"].to_numpy(),
        "exercise_style": joined["exercise_style"].to_numpy(),
        "dte_days": (joined["expiration_ns"].to_numpy() - t_ref_ns) / NS_PER_DAY,
        "multiplier": joined["multiplier"].to_numpy(),
        "F_month": F,
        "settle_price": joined["settle_price"].to_numpy(),
        "settle_is_final": joined["settle_is_final"].to_numpy(),
        "settle_is_theoretical": joined["settle_is_theoretical"].to_numpy(),
        "iv": iv,
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "vanna": greeks["vanna"],
        "charm": greeks["charm"],
        "oi": joined["oi"].to_numpy().astype("int64"),
        "oi_is_final": joined["oi_is_final"].to_numpy(),
        "oi_is_stale": joined["oi_is_stale"].to_numpy(),
        "cleared_volume": joined["cleared_volume"].to_numpy(),
        "session_open": joined["session_open"].to_numpy(),
        "session_high": joined["session_high"].to_numpy(),
        "session_low": joined["session_low"].to_numpy(),
        "lowest_offer": joined["lowest_offer"].to_numpy(),
        "highest_bid": joined["highest_bid"].to_numpy(),
        "gex": expo["gex"],
        "dex": expo["dex"],
        "vex": expo["vex"],
        "chex": expo["chex"],
    }, index=exp_idx).sort_values(["strike", "right"]).sort_index(kind="stable")

    # spot = settle of the month carrying the most OI (no candles dependency)
    oi_by_month = joined.groupby("underlying")["oi"].sum()
    spot_month = oi_by_month.idxmax() if not oi_by_month.empty else ""
    spot = settle_by_month.get(spot_month, float("nan"))

    meta = {
        "trade_date": D,
        "sign_model": p["sign_model"],
        "risk_free_rate_pct": p["risk_free_rate_pct"],
        "spot": spot,
        "spot_month": spot_month,
        "oi_source": oi_source,
        "n_contracts": int(len(df)),
        "n_iv_failed": n_iv_failed,
        "n_stats_unmatched": n_stats_unmatched,
        "n_months_no_settle": n_months_no_settle,
        "pct_oi_theoretical": pct_theo,
        "parity_max_error": parity["parity_max_error"],
        "parity_worst_expiry": parity["parity_worst_expiry"],
        "parity_expiries_checked": parity["parity_expiries_checked"],
        "total_gex": float(np.nansum(expo["gex"])),
        "total_dex": float(np.nansum(expo["dex"])),
        "total_vex": float(np.nansum(expo["vex"])),
        "total_chex": float(np.nansum(expo["chex"])),
        "transform_version": TRANSFORM_VERSION,
        "params_json": json.dumps(p, sort_keys=True),
    }
    return df, meta, warnings


def run_all(input_folder, output_folder, skip_existing=True, on_progress=None, params=None):
    p = {**PARAMS, **(params or {})}
    on_progress = on_progress or (lambda *a: None)

    get_sign_model(p["sign_model"])  # fail fast with the available-model list

    ds = resolve_option_datasets(input_folder)
    defs_index = index_dbn_files(ds["defs_dir"])
    stats_index = index_dbn_files(ds["stats_dir"])

    futures_asset = p["futures_asset"] or auto_futures_asset(ds["defs_dir"], defs_index)
    fut_stats_dir, _ = resolve_futures_dirs(
        ds["root"], p["futures_type"], futures_asset, p["futures_dataset"], "")
    fut_index = index_dbn_files(fut_stats_dir)

    trading_days = sorted(set(stats_index) & set(defs_index) & set(fut_index))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    total = len(trading_days)
    if not total:
        on_progress(1, 1, "ERROR: no dates covered by options statistics, definitions "
                          "AND futures statistics simultaneously")
        return

    on_progress(0, total,
                f"[SETUP] asset={ds['asset']} futures_asset={futures_asset} "
                f"defs={ds['defs_dir'].name} stats={ds['stats_dir'].name} "
                f"fut_stats={fut_stats_dir.name} days={total} "
                f"({trading_days[0]} .. {trading_days[-1]})")

    day_cache = _StatsCache(lambda path: load_stats(path, _EOD_STAT_TYPES))
    next_cache = _StatsCache(lambda path: load_stats(path, _NEXT_STAT_TYPES))
    fut_cache = _StatsCache(load_futures_settlements)

    stats_dates = sorted(stats_index)
    for i, date in enumerate(trading_days):
        out_file = output_path / f"{date}.parquet"
        if skip_existing and out_file.exists():
            on_progress(i + 1, total, f"[SKIP] {date} (exists)")
            continue

        # window: file D (full stat set) + up to the next 3 available files
        # (light stat set) — covers weekend files and next-morning OI/volume
        later = [d for d in stats_dates if d > date][:3]
        fut_window = [d for d in sorted(fut_index) if date <= d <= (later[-1] if later else date)]

        try:
            frames = [day_cache.get((date, "full"), stats_index[date])]
            frames += [next_cache.get((d, "light"), stats_index[d]) for d in later]
            inputs = EodInputs(
                trade_date=date,
                defs=load_definitions(defs_index[date]),
                opt_stats=pd.concat(frames, ignore_index=True),
                fut_settles=pd.concat([fut_cache.get(d, fut_index[d]) for d in fut_window],
                                      ignore_index=True),
            )
            df, meta, warns = process_day_eod(inputs, p)
            meta["asset"] = ds["asset"]
            meta["futures_asset"] = futures_asset
            for w in warns:
                on_progress(i + 1, total, f"[WARN] {date}: {w}")
            write_day(df, meta, out_file)
            on_progress(i + 1, total,
                        f"[DONE] {date}: {meta['n_contracts']} contracts, "
                        f"spot={meta['spot']:.2f} ({meta['spot_month']}), "
                        f"oi={meta['oi_source']}, total_gex={meta['total_gex']:.3e}")
        except Exception as e:  # noqa: BLE001 — per-day isolation, cancellation re-raises below
            on_progress(i + 1, total, f"[ERROR] {date}: {e}")  # cancel raised here propagates
