# transforms/big_trades.py
import databento as db
import pandas as pd
from pathlib import Path
import gc

'''
ES  30  50
NQ  10  20
'''


RTH_START         = pd.Timestamp("09:30").time()
PRE_RTH_THRESHOLD = 30
RTH_THRESHOLD     = 50


def _get_front_month(df: pd.DataFrame) -> str | None:
    volume_by_symbol = df.groupby("symbol")["size"].sum()
    if volume_by_symbol.empty:
        return None
    return volume_by_symbol.idxmax()


def build(
        previous_day_path: Path, current_day_path: Path, output_folder_path: Path,
        skip_existing: bool = True, on_log: callable = None
        ):

    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    previous_day_df = db.DBNStore.from_file(previous_day_path).to_df()
    current_day_df  = db.DBNStore.from_file(current_day_path).to_df()

    previous_day_df = previous_day_df.set_index("ts_event")
    current_day_df  = current_day_df.set_index("ts_event")

    # filter spreads
    previous_day_df = previous_day_df[~previous_day_df["symbol"].str.contains("-")]
    current_day_df  = current_day_df[~current_day_df["symbol"].str.contains("-")]

    # stitch full globex session
    prev_date     = previous_day_df.index[0].date()
    session_start = pd.Timestamp(f"{prev_date} 22:00:00", tz="UTC")
    session_end   = session_start + pd.Timedelta(hours=23)

    session_df = pd.concat([
        previous_day_df[previous_day_df.index >= session_start],
        current_day_df[current_day_df.index < session_end]
    ]).sort_index()

    del previous_day_df, current_day_df
    gc.collect()

    # front month
    front_month = _get_front_month(session_df)
    if front_month is None:
        log("No valid symbols found — skipping")
        return None
    session_df = session_df[session_df["symbol"] == front_month]

    # convert to NY time
    session_df.index = session_df.index.tz_convert("America/New_York")

    # find trade date from first RTH bar
    rth_mask  = (session_df.index.time >= RTH_START) & \
                (session_df.index.time <= pd.Timestamp("16:00").time())
    rth_bars  = session_df[rth_mask]

    if rth_bars.empty:
        log("No RTH bars found — skipping")
        return None

    trade_date  = rth_bars.index[0].date().isoformat()
    output_path = output_folder_path / f"{trade_date}.parquet"

    if skip_existing and output_path.exists():
        log(f"↷ Skipping {trade_date} — already processed")
        return None

    # filter big trades
    big = session_df[
        ((session_df["size"] >= PRE_RTH_THRESHOLD) & (session_df.index.time < RTH_START)) |
        ((session_df["size"] >= RTH_THRESHOLD)     & (session_df.index.time >= RTH_START))
    ][["price", "size", "side"]].copy()

    if big.empty:
        log(f"No big trades found for {trade_date} — skipping")
        return None

    output_folder_path.mkdir(parents=True, exist_ok=True)
    big.to_parquet(output_path)
    log(f"✓ Saved {trade_date}  ({len(big)} big trades)")
    return big


def run_all(
        input_folder: str, output_folder: str,
        skip_existing: bool = True, on_progress: callable = None
        ):

    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    files       = sorted(input_path.glob("*.dbn.zst"))

    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 files to build a session.")
        return

    total = len(files) - 1

    for i in range(total):
        def on_log(msg: str):
            if on_progress:
                on_progress(i + 1, total, msg)

        try:
            build(
                previous_day_path  = files[i],
                current_day_path   = files[i + 1],
                output_folder_path = output_path,
                skip_existing      = skip_existing,
                on_log             = on_log,
            )
        except Exception as e:
            if on_progress:
                on_progress(i + 1, total, f"ERROR {files[i + 1].name}: {e}")
            continue

        if on_progress:
            on_progress(i + 1, total, "")