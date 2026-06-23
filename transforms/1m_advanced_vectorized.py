import databento as db
import pandas as pd
from pathlib import Path
from time import perf_counter
import gc
import numpy as np
import orjson

# Per-section timing to stdout (terminal running `streamlit run`). Flip to
# False to silence.
TIMING = True


def _tlog(msg: str) -> None:
    if TIMING:
        print(f"[TIMING] {msg}", flush=True)


def _print_breakdown(date_str: str, times: dict) -> None:
    if not TIMING:
        return
    width = max(len(k) for k in times)
    _tlog(f"{date_str} breakdown (load times printed above):")
    for label, secs in times.items():
        print(f"            {label:>{width}}  {secs:7.2f}s", flush=True)
    print(f"            {'subtotal':>{width}}  {sum(times.values()):7.2f}s", flush=True)


def _get_front_month(df: pd.DataFrame) -> tuple[str, bool]:
    """
    Returns (front_month_symbol, is_roll_day).
    Roll day = back month volume > 20% of front month volume.
    """
    volume_by_symbol = df.groupby("symbol")["size"].sum()

    if volume_by_symbol.empty:
        return None, False

    front_month = volume_by_symbol.idxmax()

    sorted_volume = volume_by_symbol.sort_values(ascending=False)
    is_roll_day = False
    if len(sorted_volume) > 1:
        front_volume = sorted_volume.iloc[0]
        back_volume  = sorted_volume.iloc[1]
        is_roll_day  = back_volume > 0.2 * front_volume

    return front_month, is_roll_day


def _build_tick_volume(session_df: pd.DataFrame, bars: pd.DatetimeIndex) -> pd.Series:
    """
    Build {price: [buy_qty, sell_qty]} per 1-minute bar.
    bars: pre-computed floored index — shared with _build_passive_orders.
    """
    df = pd.DataFrame({
        "bar":   bars,
        "price": session_df["price"].values,
        "side":  session_df["side"].values,
        "size":  session_df["size"].values,
    })

    grouped = (
        df.groupby(["bar", "price", "side"])["size"]
        .sum()
        .unstack(level="side", fill_value=0)
    )
    for col in ("B", "A"):
        if col not in grouped.columns:
            grouped[col] = 0

    # Zip the grouped arrays instead of iterrows() (no per-row Series). Keep `bar`
    # tz-aware (NOT .to_numpy(), which strips the tz); .tolist() yields native
    # Python float/int for the price keys (np scalars aren't orjson-serializable).
    bar_idx   = grouped.index.get_level_values("bar")
    price_arr = grouped.index.get_level_values("price").to_numpy().tolist()
    buy_arr   = grouped["B"].to_numpy().tolist()
    sell_arr  = grouped["A"].to_numpy().tolist()

    result: dict = {}
    for bar, price, b, a in zip(bar_idx, price_arr, buy_arr, sell_arr):
        result.setdefault(bar, {})[price] = [int(b), int(a)]

    return pd.Series({k: orjson.dumps(v, option=orjson.OPT_NON_STR_KEYS).decode() for k, v in result.items()})


def _build_passive_orders(session_df: pd.DataFrame, opens: pd.Series, bars: pd.DatetimeIndex) -> pd.Series:
    """
    Build resting bid/ask liquidity per 1-minute bar.
    bars: pre-computed floored index — shared with _build_tick_volume.

    Buy aggressor  → ask levels above bar open (passive sellers overhead)
    Sell aggressor → bid levels below bar open (passive buyers below)
    """
    df = pd.DataFrame({
        "bar":             bars,
        "side":            session_df["side"].values,
        "ask_price_level": session_df["ask_price_level"].values,
        "ask_size":        session_df["ask_size"].values,
        "ask_count":       session_df["ask_count"].values,
        "bid_price_level": session_df["bid_price_level"].values,
        "bid_size":        session_df["bid_size"].values,
        "bid_count":       session_df["bid_count"].values,
    })
    df["bar_open"] = df["bar"].map(opens)

    result: dict = {}

    # ── buy aggressors ────────────────────────────────────────────────────────
    # Zip arrays instead of to_dict("records"); keep `bar` tz-aware via DatetimeIndex.
    buys = df[df["side"] == "B"]
    buys = buys[buys["ask_price_level"] > buys["bar_open"]]
    buys = buys.drop_duplicates(subset=["bar", "ask_price_level"])

    for bar, level, sz, ct in zip(
        pd.DatetimeIndex(buys["bar"]),
        buys["ask_price_level"].to_numpy().tolist(),
        buys["ask_size"].to_numpy().tolist(),
        buys["ask_count"].to_numpy().tolist(),
    ):
        result.setdefault(bar, {})[level] = [int(sz), int(ct)]

    # ── sell aggressors ───────────────────────────────────────────────────────
    sells = df[df["side"] == "A"]
    sells = sells[sells["bid_price_level"] < sells["bar_open"]]
    sells = sells.drop_duplicates(subset=["bar", "bid_price_level"])

    for bar, level, sz, ct in zip(
        pd.DatetimeIndex(sells["bar"]),
        sells["bid_price_level"].to_numpy().tolist(),
        sells["bid_size"].to_numpy().tolist(),
        sells["bid_count"].to_numpy().tolist(),
    ):
        bar_dict = result.setdefault(bar, {})
        if level not in bar_dict:
            bar_dict[level] = [int(sz), int(ct)]

    return pd.Series({k: orjson.dumps(v, option=orjson.OPT_NON_STR_KEYS).decode() for k, v in result.items()})


_UNDEF_PRICE = np.iinfo(np.int64).max  # Databento sentinel for "no price"


def _instrument_symbols(store, unique_ids) -> dict:
    """Resolve the handful of instrument_ids in a file to their raw symbols once
    (vs to_df mapping a symbol string onto every row)."""
    imap = db.common.symbology.InstrumentMap()
    imap.insert_metadata(store.metadata)
    date = pd.Timestamp(store.metadata.start, unit="ns").date()
    out = {}
    for iid in unique_ids:
        try:
            sym = imap.resolve(int(iid), date)
        except Exception:
            sym = None
        out[int(iid)] = sym if sym is not None else str(int(iid))
    return out


def _px(raw: np.ndarray) -> np.ndarray:
    """int64 fixed-point (1e-9) -> float price; undefined-price sentinel -> NaN."""
    return np.where(raw == _UNDEF_PRICE, np.nan, raw.astype("float64") / 1e9)


def _load_and_clean(path: Path) -> pd.DataFrame:
    """
    Load a DBN MBP-1/TBBO file and apply all filters.

    Uses to_ndarray() (raw decode ~0.06s) instead of to_df() (~0.8s): to_df builds
    a full pandas frame and maps a symbol string onto every row. We resolve symbols
    once for the ~8 instrument_ids. Output is identical to the previous loader
    (price/bid/ask = int64 fixed-point /1e9 with sentinel -> NaN; size/counts uint32).
    """
    t0 = perf_counter()
    store = db.DBNStore.from_file(str(path))
    arr = store.to_ndarray()

    idx = pd.DatetimeIndex(arr["ts_event"].astype("int64"), tz="UTC")
    codes, uniq = pd.factorize(arr["instrument_id"])
    id2sym = _instrument_symbols(store, uniq)
    sym_lookup = np.array([id2sym[int(u)] for u in uniq], dtype=object)

    df = pd.DataFrame(
        {
            "side":            arr["side"].astype("U1").astype(object),
            "price":           _px(arr["price"]),
            "size":            arr["size"],
            "bid_price_level": _px(arr["bid_px_00"]),
            "ask_price_level": _px(arr["ask_px_00"]),
            "bid_size":        arr["bid_sz_00"],
            "ask_size":        arr["ask_sz_00"],
            "bid_count":       arr["bid_ct_00"],
            "ask_count":       arr["ask_ct_00"],
            "symbol":          sym_lookup[codes],
        },
        index=idx,
    )

    df = df[~df["symbol"].str.contains("-", na=False)]
    df = df[df["side"].isin(["A", "B"])]

    _tlog(f"load {Path(path).name}: {perf_counter() - t0:6.2f}s  ({len(df):,} rows)")
    return df


def _build_session(previous_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splice previous + current into one 23-hour Globex session.
    Session window: 22:00 UTC prev day → 21:00 UTC current day.
    """
    prev_date     = previous_df.index[0].date()
    session_start = pd.Timestamp(f"{prev_date} 22:00:00", tz="UTC")
    session_end   = session_start + pd.Timedelta(hours=23)

    session_df = pd.concat([
        previous_df[previous_df.index >= session_start],
        current_df[current_df.index  <  session_end],
    ]).sort_index()

    return session_df


def build_candles(
        previous_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_folder_path: Path,
        skip_existing: bool = False,
        on_log: callable = None,
):
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    times: dict = {}

    t = perf_counter()
    session_df = _build_session(previous_df, current_df)
    times["build_session"] = perf_counter() - t

    t = perf_counter()
    front_month, is_roll_day = _get_front_month(session_df)
    if front_month is None:
        log("No valid symbols found — skipping")
        return None
    if is_roll_day:
        log(f"Roll day detected — front: {front_month}")

    session_df = session_df[session_df["symbol"] == front_month]

    if session_df.empty:
        log("Empty session after front month filter — skipping")
        return None

    session_df.index = session_df.index.tz_convert("America/New_York")
    times["front_month"] = perf_counter() - t

    t = perf_counter()
    # RTH date = the current session's calendar date (RTH 13:30-21:00 UTC sits in
    # the curr UTC day, which equals the NY date). Derive it from a midpoint row,
    # then check the 09:30-16:00 NY window with a vectorized absolute-timestamp
    # comparison instead of the slow per-row `.time` accessor.
    rth_date  = current_df.index[len(current_df) // 2].tz_convert("UTC").date()
    rth_start = pd.Timestamp(f"{rth_date} 09:30", tz="America/New_York")
    rth_end   = pd.Timestamp(f"{rth_date} 16:00", tz="America/New_York")
    has_rth   = bool(((session_df.index >= rth_start) & (session_df.index <= rth_end)).any())
    times["rth_mask"] = perf_counter() - t

    if not has_rth:
        log("No RTH bars found — skipping")
        return None

    trade_date  = rth_date.isoformat()
    output_path = output_folder_path / f"{trade_date}.parquet"

    if skip_existing and output_path.exists():
        log(f"↷ Skipping {trade_date} — already processed")
        return None

    t = perf_counter()
    session_df["buy_volume"]  = session_df["size"].where(session_df["side"] == "B", 0)
    session_df["sell_volume"] = session_df["size"].where(session_df["side"] == "A", 0)

    candles = session_df.groupby(pd.Grouper(freq="1min")).agg(
        open        = ("price",       "first"),
        high        = ("price",       "max"),
        low         = ("price",       "min"),
        close       = ("price",       "last"),
        volume      = ("size",        "sum"),
        buy_volume  = ("buy_volume",  "sum"),
        sell_volume = ("sell_volume", "sum"),
    )

    # gap fill
    candles["close"] = candles["close"].ffill().bfill()
    candles["open"]  = candles["open"].fillna(candles["close"])
    candles["high"]  = candles["high"].fillna(candles["close"])
    candles["low"]   = candles["low"].fillna(candles["close"])
    candles[["volume", "buy_volume", "sell_volume"]] = (
        candles[["volume", "buy_volume", "sell_volume"]].fillna(0)
    )

    # volume delta
    candles["buy_volume"]   = candles["buy_volume"].astype("int32")
    candles["sell_volume"]  = candles["sell_volume"].astype("int32")
    candles["volume_delta"] = (candles["buy_volume"] - candles["sell_volume"]).astype("int32")

    buy   = candles["buy_volume"].to_numpy()
    sell  = candles["sell_volume"].to_numpy()
    delta = candles["volume_delta"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(
            (buy == 0) | (sell == 0), 100.0,
            np.where(buy  > sell, (delta / sell) * 100,
            np.where(sell > buy,  (delta / buy)  * 100,
            0.0))
        )

    candles["volume_delta_pct"] = pct.round(1)
    candles.loc[candles["volume"] == 0, "volume_delta_pct"] = 0.0
    times["groupby_agg"] = perf_counter() - t

    # compute bar index once — integer arithmetic on ns is ~800x faster than tz-aware floor()
    ns         = session_df.index.view("int64")
    floored_ns = (ns // 60_000_000_000) * 60_000_000_000
    bars       = pd.DatetimeIndex(floored_ns, tz="UTC").tz_convert("America/New_York")

    t = perf_counter()
    tick_vol = _build_tick_volume(session_df, bars)
    candles["tick_volume"] = tick_vol.reindex(candles.index).fillna("{}")
    times["tick_volume"] = perf_counter() - t

    t = perf_counter()
    passive = _build_passive_orders(session_df, candles["open"], bars)
    candles["passive_orders"] = passive.reindex(candles.index).fillna("{}")
    times["passive_orders"] = perf_counter() - t

    for col in ["open", "high", "low", "close"]:
        candles[col] = candles[col].astype("float64")
    candles["volume_delta_pct"] = candles["volume_delta_pct"].astype("float64")

    output_folder_path.mkdir(parents=True, exist_ok=True)
    t = perf_counter()
    candles.to_parquet(output_path)
    times["write_parquet"] = perf_counter() - t

    log(f"✓ Saved {trade_date}")
    _print_breakdown(trade_date, times)
    return candles


def run_all(
        input_folder: str,
        output_folder: str,
        skip_existing: bool = True,
        on_progress: callable = None,
) -> None:
    input_path  = Path(input_folder)
    output_path = Path(output_folder)

    files = sorted(input_path.glob("*.dbn.zst"))

    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 files to build a session.")
        return

    total = len(files) - 1

    prev_df = _load_and_clean(files[0])

    for i in range(total):
        current_file = files[i + 1]

        def on_log(msg: str, _i=i):
            if on_progress:
                on_progress(_i + 1, total, msg)

        curr_df = _load_and_clean(current_file)

        try:
            build_candles(
                previous_df        = prev_df,
                current_df         = curr_df,
                output_folder_path = output_path,
                skip_existing      = skip_existing,
                on_log             = on_log,
            )
        except Exception as e:
            if on_progress:
                on_progress(i + 1, total, f"ERROR {current_file.name}: {e}")
            prev_df = curr_df
            continue

        if on_progress:
            on_progress(i + 1, total, "")

        del prev_df
        gc.collect()
        prev_df = curr_df

    del prev_df
    gc.collect()


'''
Known limitations:
- Tick volume accuracy: multi-level fills recorded only at the reported price.
- If there is no trading between 22:00-24:00 in previous day then it will only
  return current day.
'''