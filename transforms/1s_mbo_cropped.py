# transforms/1s_mbo_cropped.py
#
# Convert raw Databento MBO (L3) .dbn.zst files into enriched 1-second candle
# Parquet files for a DOM heatmap renderer — CROPPED variant.
#
# Identical schema to 1s_mbo_full_book.py, but bid_depth / ask_depth keep only:
#   1. levels within ±N_TICKS of the second's trade high/low (the touch), and
#   2. far-away "big" resting orders — size >= BIG_ORDER_MULT × a rolling
#      near-book size baseline (median over the last BASELINE_WINDOW_MIN minutes,
#      so the threshold adapts across ETH vs RTH liquidity regimes).
# Empty levels are never written.
#
# The sequential L3 replay + cropping + JSON emission run in Rust
# (heatmap_rs.replay_cropped). This transform REQUIRES the heatmap_rs extension
# (build with: maturin develop --release -m heatmap_rs/Cargo.toml).

from __future__ import annotations

import gc
import re
from pathlib import Path
from time import perf_counter

import databento as db
import numpy as np
import orjson
import pandas as pd

try:
    import heatmap_rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

PRICE_SCALE = 1_000_000_000
NS_PER_SEC  = 1_000_000_000

# ── TUNABLE PARAMETERS ──────────────────────────────────────────────────────
N_TICKS            = 100   # keep levels within ±N_TICKS of the touch each second
BIG_ORDER_MULT     = 2.0   # a far level is "big" if size >= MULT × baseline
BASELINE_WINDOW_MIN = 30   # rolling window (minutes) for the near-book baseline
DEFAULT_TICK_SIZE  = 0.25  # fallback if the asset's tick isn't in TICK_SIZES

PARAMS = {
    "n_ticks":            N_TICKS,
    "big_order_mult":     BIG_ORDER_MULT,
    "baseline_window_min": BASELINE_WINDOW_MIN,
}

# Tick size per contract root (price increment). Extend as needed.
TICK_SIZES = {
    "ES": 0.25, "NQ": 0.25, "RTY": 0.10, "YM": 1.00,
    "MES": 0.25, "MNQ": 0.25, "M2K": 0.10, "MYM": 1.00,
    "CL": 0.01, "GC": 0.10, "SI": 0.005, "HG": 0.0005,
    "ZN": 0.015625, "ZB": 0.03125, "ZF": 0.0078125, "ZT": 0.00390625,
    "6E": 0.00005, "6J": 0.0000005, "6B": 0.0001,
}

# Event encodings passed to the Rust kernel.
_ACODE = {"A": 0, "C": 1, "M": 2, "F": 3, "R": 4, "T": 5}
_SCODE = {"B": 0, "A": 1}

# Per-section timing to stdout. Flip to False to silence once profiling is done.
TIMING = True


def _tlog(msg: str) -> None:
    if TIMING:
        print(f"[TIMING] {msg}", flush=True)


def _tick_size(symbol: str) -> float:
    """Infer the price increment from a futures symbol, e.g. ESM6 -> ES -> 0.25."""
    m = re.match(r"[A-Z]+", str(symbol))
    if not m:
        return DEFAULT_TICK_SIZE
    letters = m.group(0)
    root = letters[:-1] if len(letters) > 1 else letters  # drop the month code
    return TICK_SIZES.get(root, DEFAULT_TICK_SIZE)


def run_all(
    input_folder: str,
    output_folder: str,
    skip_existing: bool,
    on_progress,
) -> None:
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not _HAS_RUST:
        on_progress(1, 1, "ERROR: heatmap_rs extension not built. Run: "
                          "maturin develop --release -m heatmap_rs/Cargo.toml")
        return

    files = sorted(input_path.glob("*.dbn.zst"))

    if len(files) < 2:
        on_progress(1, 1, "ERROR: Need at least 2 files to build a session.")
        return

    total   = len(files) - 1
    prev_df = _load_and_clean(files[0])

    for i in range(total):
        curr_file = files[i + 1]
        # Filename stem like "glbx-mdp3-20260423.mbo" -> ISO date "2026-04-23".
        ymd       = curr_file.stem.split(".")[0].split("-")[-1]
        date_str  = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        out_file  = output_path / f"{date_str}.parquet"

        def log(msg: str, _i=i):
            on_progress(_i + 1, total, msg)

        if skip_existing and out_file.exists():
            log(f"[SKIP] {date_str}")
            del prev_df
            gc.collect()
            prev_df = _load_and_clean(curr_file)
            continue

        curr_df = _load_and_clean(curr_file)

        try:
            _process_day(
                prev_df  = prev_df,
                curr_df  = curr_df,
                out_file = out_file,
                date_str = date_str,
                log      = log,
            )
        except Exception as e:
            log(f"[ERROR] {date_str}: {e}")
            del prev_df
            gc.collect()
            prev_df = curr_df
            continue

        del prev_df
        gc.collect()
        prev_df = curr_df

    del prev_df
    gc.collect()


def _load_and_clean(path: Path) -> pd.DataFrame:
    """Load a DBN MBO file and apply minimal filtering.

    Prices are left untouched — to_df() already returns float64 (e.g. 7025.0).
    side=N rows are kept (action=R book-clears carry side=N / NaN price).
    """
    t0 = perf_counter()
    df = db.DBNStore.from_file(str(path)).to_df()

    # to_df() indexes by ts_recv. Ensure tz-aware UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["action", "side", "price", "size", "order_id", "symbol"]]

    # Drop spreads / combos (e.g. ESM6-ESU6).
    df = df[~df["symbol"].str.contains("-", na=False)]

    # Keep all sides — N rows are needed for action=R. Tighten dtypes only.
    df = df.astype({"size": np.int32, "order_id": np.int64})

    _tlog(f"load {path.name}: {perf_counter() - t0:6.2f}s  ({len(df):,} rows)")
    return df


def _build_session(
    prev_df: pd.DataFrame, curr_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Splice prev + curr into one 23-hour Globex session (UTC index kept).

    Window: prev_date 22:00 UTC -> prev_date 22:00 UTC + 23h (= curr 21:00 UTC).
    prev_df.index[-1] is used (not [0]) because [0] is the 00:00 UTC synthetic
    book snapshot, which would mis-date the session on edge files.
    """
    prev_date     = prev_df.index[-1].normalize()
    session_start = prev_date + pd.Timedelta(hours=22)
    session_end   = session_start + pd.Timedelta(hours=23)

    session_df = pd.concat([
        prev_df[prev_df.index >= session_start],
        curr_df[curr_df.index  <  session_end],
    ]).sort_index()

    return session_df, session_start, session_end


def _get_front_month(df: pd.DataFrame) -> tuple[str | None, bool]:
    """Front month = symbol with the most traded volume. Roll day = back month
    volume > 20% of front month volume."""
    trades = df[df["action"] == "T"]
    vol_by_symbol = trades.groupby("symbol")["size"].sum()

    if vol_by_symbol.empty:
        return None, False

    sorted_vol  = vol_by_symbol.sort_values(ascending=False)
    front_month = sorted_vol.index[0]
    is_roll_day = len(sorted_vol) > 1 and sorted_vol.iloc[1] > 0.2 * sorted_vol.iloc[0]

    return front_month, is_roll_day


def _process_day(
    prev_df:  pd.DataFrame,
    curr_df:  pd.DataFrame,
    out_file: Path,
    date_str: str,
    log,
) -> None:
    times: dict[str, float] = {}

    t = perf_counter()
    session_df, session_start, session_end = _build_session(prev_df, curr_df)
    times["build_session"] = perf_counter() - t
    if session_df.empty:
        log(f"[WARN] {date_str}: empty session after splice")
        return

    t = perf_counter()
    front_month, is_roll_day = _get_front_month(session_df)
    if front_month is None:
        log(f"[WARN] {date_str}: no valid symbols found")
        return
    session_df = session_df[session_df["symbol"] == front_month]
    times["front_month"] = perf_counter() - t

    tag = "ROLL" if is_roll_day else "START"
    tick = _tick_size(front_month)
    log(f"[{tag}] {date_str}: front month {front_month} (tick {tick}, ±{N_TICKS} ticks)")
    if session_df.empty:
        log(f"[WARN] {date_str}: empty after front month filter")
        return

    log(f"[REPLAY] {date_str}: {len(session_df):,} MBO events")

    # Both halves are keyed by integer UTC epoch-second.
    t = perf_counter()
    trades = _aggregate_trades(session_df)
    times["aggregate_trades"] = perf_counter() - t

    t = perf_counter()
    book = _replay_book(session_df, trades, tick)
    times["replay_book"] = perf_counter() - t

    # Full per-second grid: every second of the 23h window (= 82,800 in UTC).
    start_sec = int(session_start.value // NS_PER_SEC)
    n_sec     = int((session_end - session_start) // pd.Timedelta(seconds=1))
    grid      = np.arange(start_sec, start_sec + n_sec, dtype=np.int64)

    t = perf_counter()
    bars = _merge_grid(grid, trades, book)

    # Integer UTC seconds -> tz-aware NY index (convert from UTC: no DST ambiguity).
    bars.index = (
        pd.DatetimeIndex(grid * NS_PER_SEC)
        .tz_localize("UTC")
        .tz_convert("America/New_York")
    )
    bars.index.name = "timestamp"
    times["merge_grid"] = perf_counter() - t

    out_file.parent.mkdir(parents=True, exist_ok=True)
    t = perf_counter()
    bars.to_parquet(out_file)
    times["write_parquet"] = perf_counter() - t

    log(f"[DONE] {date_str}: {len(bars)} bars -> {out_file.name}")
    _print_breakdown(date_str, times)


def _print_breakdown(date_str: str, times: dict[str, float]) -> None:
    """Print the per-day section breakdown (excl. file loads, timed separately)."""
    if not TIMING:
        return
    width = max(len(k) for k in times)
    _tlog(f"{date_str} breakdown (load times printed above):")
    for label, secs in times.items():
        print(f"            {label:>{width}}  {secs:7.2f}s", flush=True)
    print(f"            {'subtotal':>{width}}  {sum(times.values()):7.2f}s", flush=True)


def _merge_grid(
    grid: np.ndarray, trades: pd.DataFrame, book: pd.DataFrame
) -> pd.DataFrame:
    """Reindex both halves onto the full second grid and gap-fill.

    Book fields are forward-filled (resting book persists through quiet seconds);
    trade OHLC is gap-filled flat and volumes zero-filled.
    """
    # ── Book half: ffill across quiet seconds ────────────────────────────────
    book = book.reindex(grid)
    book["best_bid"]  = book["best_bid"].ffill()
    book["best_ask"]  = book["best_ask"].ffill()
    book["bid_depth"] = book["bid_depth"].ffill().fillna("{}")
    book["ask_depth"] = book["ask_depth"].ffill().fillna("{}")

    # ── Trade half: flat gap-fill (same as candles_1m reference) ─────────────
    if trades.empty:
        idx = pd.Index(grid)
        trades = pd.DataFrame(
            {
                "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                "volume": 0, "buy_volume": 0, "sell_volume": 0,
                "aggressor_volume": "{}",
            },
            index=idx,
        )
    else:
        trades = trades.reindex(grid)
        trades["close"] = trades["close"].ffill().bfill()
        trades["open"]  = trades["open"].fillna(trades["close"])
        trades["high"]  = trades["high"].fillna(trades["close"])
        trades["low"]   = trades["low"].fillna(trades["close"])
        trades[["volume", "buy_volume", "sell_volume"]] = (
            trades[["volume", "buy_volume", "sell_volume"]].fillna(0)
        )
        trades["aggressor_volume"] = trades["aggressor_volume"].fillna("{}")

    trades["volume"]      = trades["volume"].astype(np.int64)
    trades["buy_volume"]  = trades["buy_volume"].astype(np.int32)
    trades["sell_volume"] = trades["sell_volume"].astype(np.int32)

    bars = pd.DataFrame({
        "open":             trades["open"].astype(np.float64),
        "high":             trades["high"].astype(np.float64),
        "low":              trades["low"].astype(np.float64),
        "close":            trades["close"].astype(np.float64),
        "volume":           trades["volume"],
        "buy_volume":       trades["buy_volume"],
        "sell_volume":      trades["sell_volume"],
        "best_bid":         book["best_bid"].astype(np.float64),
        "best_ask":         book["best_ask"].astype(np.float64),
        "aggressor_volume": trades["aggressor_volume"],
        "bid_depth":        book["bid_depth"],
        "ask_depth":        book["ask_depth"],
    })
    return bars


def _aggregate_trades(session_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized OHLCV / volume / aggressor_volume from trade prints only.

    Returns a DataFrame indexed by integer UTC epoch-second.
    """
    trades = session_df[session_df["action"] == "T"]
    if trades.empty:
        return pd.DataFrame()

    # Bucket by 1s using integer ns — far faster than tz-aware floor().
    bar = trades.index.view("int64") // NS_PER_SEC

    price = trades["price"].to_numpy(dtype=np.float64)
    size  = trades["size"].to_numpy(dtype=np.int64)
    side  = trades["side"].to_numpy()

    df = pd.DataFrame({
        "bar":   bar,
        "price": price,
        "size":  size,
        "side":  side,
    })
    df["buy_volume"]  = np.where(side == "B", size, 0)
    df["sell_volume"] = np.where(side == "A", size, 0)

    grouped = df.groupby("bar")
    bars = grouped.agg(
        open        = ("price",       "first"),
        high        = ("price",       "max"),
        low         = ("price",       "min"),
        close       = ("price",       "last"),
        volume      = ("size",        "sum"),
        buy_volume  = ("buy_volume",  "sum"),
        sell_volume = ("sell_volume", "sum"),
    )
    bars["volume"]      = bars["volume"].astype(np.int64)
    bars["buy_volume"]  = bars["buy_volume"].astype(np.int32)
    bars["sell_volume"] = bars["sell_volume"].astype(np.int32)

    bars["aggressor_volume"] = _build_aggressor_volume(df)

    return bars


def _build_aggressor_volume(df: pd.DataFrame) -> pd.Series:
    """Per 1s bar: {str(price): [buy_qty, sell_qty]}, JSON-encoded."""
    grouped = (
        df.groupby(["bar", "price", "side"])["size"]
        .sum()
        .unstack(level="side", fill_value=0)
    )
    for col in ("B", "A"):
        if col not in grouped.columns:
            grouped[col] = 0

    result: dict = {}
    for (bar, price), row in grouped.iterrows():
        result.setdefault(bar, {})[str(price)] = [int(row["B"]), int(row["A"])]

    return pd.Series({k: orjson.dumps(v).decode() for k, v in result.items()})


def _encode_events(session_df: pd.DataFrame):
    """Encode the event stream into the int arrays the Rust kernel consumes."""
    acode = session_df["action"].map(_ACODE).fillna(6).astype(np.int8).to_numpy()
    scode = session_df["side"].map(_SCODE).fillna(2).astype(np.int8).to_numpy()
    prices  = session_df["price"].to_numpy(dtype=np.float64)
    price_i = np.where(np.isnan(prices), 0, np.round(prices * PRICE_SCALE)).astype(np.int64)
    size = session_df["size"].to_numpy(dtype=np.int64)
    oid  = session_df["order_id"].to_numpy(dtype=np.int64)
    sec  = (session_df.index.view("int64") // NS_PER_SEC).astype(np.int64)
    return acode, scode, price_i, size, oid, sec


def _replay_book(session_df: pd.DataFrame, trades: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Sequential L3 replay → per-second CROPPED book snapshots (Rust kernel).

    Window is [trade_low - N_TICKS, trade_high + N_TICKS] per second (falling
    back to best_bid/best_ask on no-trade seconds), plus far "big" levels whose
    size >= BIG_ORDER_MULT × the rolling near-book median (last
    BASELINE_WINDOW_MIN minutes). Returns a DataFrame indexed by int UTC second.
    """
    acode, scode, price_i, size, oid, sec = _encode_events(session_df)

    if trades.empty:
        trade_sec = np.empty(0, dtype=np.int64)
        trade_lo  = np.empty(0, dtype=np.int64)
        trade_hi  = np.empty(0, dtype=np.int64)
    else:
        trade_sec = trades.index.to_numpy().astype(np.int64)
        trade_lo  = np.round(trades["low"].to_numpy(dtype=np.float64)  * PRICE_SCALE).astype(np.int64)
        trade_hi  = np.round(trades["high"].to_numpy(dtype=np.float64) * PRICE_SCALE).astype(np.int64)

    tick_i      = int(round(tick * PRICE_SCALE))
    window_sec  = int(BASELINE_WINDOW_MIN * 60)

    t = perf_counter()
    secs, bb, ba, bj, aj = heatmap_rs.replay_cropped(
        acode, scode, price_i, size, oid, sec,
        int(N_TICKS), tick_i, float(BIG_ORDER_MULT), window_sec,
        trade_sec, trade_lo, trade_hi,
    )
    _tlog(
        f"replay_book(rust,cropped): rust={perf_counter() - t:6.2f}s  "
        f"(events={len(acode):,}, snapshots={len(secs):,})"
    )

    if not secs:
        return pd.DataFrame(columns=["best_bid", "best_ask", "bid_depth", "ask_depth"])
    return pd.DataFrame(
        {"best_bid": bb, "best_ask": ba, "bid_depth": bj, "ask_depth": aj},
        index=pd.Index(secs, dtype=np.int64),
    )
