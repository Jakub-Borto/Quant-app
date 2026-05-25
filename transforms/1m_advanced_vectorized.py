import databento as db
import pandas as pd
from pathlib import Path
import gc
import numpy as np
import orjson


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

    result: dict = {}
    for (bar, price), row in grouped.iterrows():
        result.setdefault(bar, {})[price] = [int(row["B"]), int(row["A"])]

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
    buys = df[df["side"] == "B"]
    buys = buys[buys["ask_price_level"] > buys["bar_open"]]
    buys = buys.drop_duplicates(subset=["bar", "ask_price_level"])

    for row in buys[["bar", "ask_price_level", "ask_size", "ask_count"]].to_dict("records"):
        result.setdefault(row["bar"], {})[row["ask_price_level"]] = [
            int(row["ask_size"]), int(row["ask_count"])
        ]

    # ── sell aggressors ───────────────────────────────────────────────────────
    sells = df[df["side"] == "A"]
    sells = sells[sells["bid_price_level"] < sells["bar_open"]]
    sells = sells.drop_duplicates(subset=["bar", "bid_price_level"])

    for row in sells[["bar", "bid_price_level", "bid_size", "bid_count"]].to_dict("records"):
        bar      = row["bar"]
        level    = row["bid_price_level"]
        bar_dict = result.setdefault(bar, {})
        if level not in bar_dict:
            bar_dict[level] = [int(row["bid_size"]), int(row["bid_count"])]

    return pd.Series({k: orjson.dumps(v, option=orjson.OPT_NON_STR_KEYS).decode() for k, v in result.items()})


def _load_and_clean(path: Path) -> pd.DataFrame:
    """
    Load a DBN file and apply all filters.
    Returns a clean DataFrame ready for session splicing.
    """
    df = db.DBNStore.from_file(path).to_df()
    df = df.set_index("ts_event") if "ts_event" in df.columns else df

    KEEP = {
        "side":      "side",
        "price":     "price",
        "size":      "size",
        "bid_px_00": "bid_price_level",
        "ask_px_00": "ask_price_level",
        "bid_sz_00": "bid_size",
        "ask_sz_00": "ask_size",
        "bid_ct_00": "bid_count",
        "ask_ct_00": "ask_count",
        "symbol":    "symbol",
    }

    df = df[list(KEEP.keys())].rename(columns=KEEP)
    df = df[~df["symbol"].str.contains("-")]
    df = df[df["side"].isin(["A", "B"])]

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

    session_df = _build_session(previous_df, current_df)

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

    rth_mask = (
        (session_df.index.time >= pd.Timestamp("09:30").time()) &
        (session_df.index.time <= pd.Timestamp("16:00").time())
    )
    rth_bars = session_df[rth_mask]

    if rth_bars.empty:
        log("No RTH bars found — skipping")
        return None

    trade_date  = rth_bars.index[0].date().isoformat()
    output_path = output_folder_path / f"{trade_date}.parquet"

    if skip_existing and output_path.exists():
        log(f"↷ Skipping {trade_date} — already processed")
        return None

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

    # compute bar index once — integer arithmetic on ns is ~800x faster than tz-aware floor()
    ns         = session_df.index.view("int64")
    floored_ns = (ns // 60_000_000_000) * 60_000_000_000
    bars       = pd.DatetimeIndex(floored_ns, tz="UTC").tz_convert("America/New_York")

    tick_vol = _build_tick_volume(session_df, bars)
    candles["tick_volume"] = tick_vol.reindex(candles.index).fillna("{}")

    passive = _build_passive_orders(session_df, candles["open"], bars)
    candles["passive_orders"] = passive.reindex(candles.index).fillna("{}")

    for col in ["open", "high", "low", "close"]:
        candles[col] = candles[col].astype("float64")
    candles["volume_delta_pct"] = candles["volume_delta_pct"].astype("float64")

    output_folder_path.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(output_path)
    log(f"✓ Saved {trade_date}")
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