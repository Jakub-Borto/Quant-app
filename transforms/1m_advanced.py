import databento as db
import pandas as pd
from pathlib import Path
import gc
import numpy as np
import json


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


def _build_tick_volume(group: pd.DataFrame) -> dict:
    buy  = group[group["side"] == "B"].groupby("price")["size"].sum()
    sell = group[group["side"] == "A"].groupby("price")["size"].sum()
    
    prices = set(buy.index) | set(sell.index)
    return {p: (int(buy.get(p, 0)), int(sell.get(p, 0))) for p in prices}


def _build_passive_orders(group: pd.DataFrame, bar_open: float) -> dict:
    passive_orders = {}
    
    buys = group[group["side"] == "B"][["ask_price_level", "ask_size", "ask_count"]]
    buys = buys.drop_duplicates(subset="ask_price_level")
    buys = buys[buys["ask_price_level"] > bar_open]
    for _, row in buys.iterrows():
        passive_orders[row["ask_price_level"]] = [int(row["ask_size"]), int(row["ask_count"])]

    sells = group[group["side"] == "A"][["bid_price_level", "bid_size", "bid_count"]]
    sells = sells.drop_duplicates(subset="bid_price_level")
    sells = sells[sells["bid_price_level"] < bar_open]
    for _, row in sells.iterrows():
        if row["bid_price_level"] not in passive_orders:
            passive_orders[row["bid_price_level"]] = [int(row["bid_size"]), int(row["bid_count"])]

    return passive_orders


def _modify_and_join(previous_day_df: pd.DataFrame, current_day_df: pd.DataFrame) -> pd.DataFrame:
    previous_day_df = previous_day_df.set_index("ts_event")
    current_day_df = current_day_df.set_index("ts_event")
    
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

    previous_day_df = previous_day_df[list(KEEP.keys())].rename(columns=KEEP)
    current_day_df  = current_day_df[list(KEEP.keys())].rename(columns=KEEP)

    # filter out spreads — CME symbology only
    previous_day_df = previous_day_df[~previous_day_df["symbol"].str.contains("-")]
    current_day_df  = current_day_df[~current_day_df["symbol"].str.contains("-")]

    previous_day_df = previous_day_df[previous_day_df["side"].isin(["A", "B"])]
    current_day_df  = current_day_df[current_day_df["side"].isin(["A", "B"])]
    
    prev_date     = previous_day_df.index[0].date()
    session_start = pd.Timestamp(f"{prev_date} 22:00:00", tz="UTC")
    session_end   = session_start + pd.Timedelta(hours=23)

    session_df = pd.concat([
        previous_day_df[previous_day_df.index >= session_start],
        current_day_df[current_day_df.index  <  session_end]
    ])

    session_df = session_df.sort_index()
    return session_df


def _ohlcv_bv_sv(session_df: pd.DataFrame) -> pd.DataFrame:
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
    return candles


def _fill_candle_gaps(candles: pd.DataFrame) -> pd.DataFrame:
    candles["close"]  = candles["close"].ffill().bfill()
    candles["open"]   = candles["open"].fillna(candles["close"])
    candles["high"]   = candles["high"].fillna(candles["close"])
    candles["low"]    = candles["low"].fillna(candles["close"])
    candles[["volume", "buy_volume", "sell_volume"]] = (
        candles[["volume", "buy_volume", "sell_volume"]].fillna(0)
    )
    return candles


def _volume_delta_vdpct(candles: pd.DataFrame) -> pd.DataFrame:
    candles["buy_volume"]   = candles["buy_volume"].astype("int32")
    candles["sell_volume"]  = candles["sell_volume"].astype("int32")
    candles["volume_delta"] = (candles["buy_volume"] - candles["sell_volume"]).astype("int32")

    buy   = candles["buy_volume"].to_numpy()
    sell  = candles["sell_volume"].to_numpy()
    delta = candles["volume_delta"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(
            (buy == 0) | (sell == 0), 100.0,
            np.where(buy  > sell, ( delta / sell) * 100,
            np.where(sell > buy,  ( delta / buy)  * 100,
            0.0))
        )

    candles["volume_delta_pct"] = pct.round(1)
    candles.loc[candles["volume"] == 0, "volume_delta_pct"] = 0.0
    return candles


def _change_columns_type(candles: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close"]:
        candles[col] = candles[col].astype("float32")
    return candles


def build_candles(
        previous_day_path: Path,
        current_day_path: Path,
        output_folder_path: Path,
        skip_existing: bool = False,
        on_log: callable = None,
):
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    previous_day_df = db.DBNStore.from_file(previous_day_path).to_df()
    current_day_df  = db.DBNStore.from_file(current_day_path).to_df()

    session_df = _modify_and_join(previous_day_df, current_day_df)

    del previous_day_df, current_day_df
    gc.collect()

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

    candles = _ohlcv_bv_sv(session_df)
    candles = _fill_candle_gaps(candles)
    candles = _volume_delta_vdpct(candles)

    tick_volume = session_df.groupby(pd.Grouper(freq="1min")).apply(_build_tick_volume)
    candles["tick_volume"] = tick_volume.apply(lambda x: json.dumps(x) if x else "{}")

    opens   = candles["open"]
    passive = {}
    for bar_time, group in session_df.groupby(pd.Grouper(freq="1min")):
        if bar_time in opens.index:
            passive[bar_time] = _build_passive_orders(group, float(opens[bar_time]))

    candles["passive_orders"] = pd.Series(passive).apply(
        lambda x: json.dumps(x) if x else "{}"
    )

    candles = _change_columns_type(candles)

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

    for i in range(total):
        previous_file = files[i]
        current_file  = files[i + 1]

        def on_log(msg: str, _i=i):
            if on_progress:
                on_progress(_i + 1, total, msg)

        try:
            build_candles(
                previous_day_path  = previous_file,
                current_day_path   = current_file,
                output_folder_path = output_path,
                skip_existing      = skip_existing,
                on_log             = on_log,
            )
        except Exception as e:
            if on_progress:
                on_progress(i + 1, total, f"ERROR {current_file.name}: {e}")
            continue

        if on_progress:
            on_progress(i + 1, total, "")


'''
Known limitations:
- tick volume isn't 100% right. Multi-level fills recorded only at the reported price.
- If there is no trading between 22:00-24:00 in previous day then it will only
  return current day.
'''