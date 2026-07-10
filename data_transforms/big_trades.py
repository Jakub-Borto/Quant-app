# transforms/big_trades.py
import databento as db
import pandas as pd
from pathlib import Path
import gc


# UI-configurable parameters (Data Formatter renders widgets from this dict,
# exactly like strategy PARAMS in the Backtester). Defaults reproduce the
# original hardcoded behavior.
PARAMS = {
    "rth_start":         "09:30",
    "rth_end":           "16:00",
    "pre_rth_threshold": 10,
    "rth_threshold":     10,
}


def _get_front_month(df: pd.DataFrame) -> str | None:
    volume_by_symbol = df.groupby("symbol")["size"].sum()
    if volume_by_symbol.empty:
        return None
    return volume_by_symbol.idxmax()


def _load_and_clean(path: Path) -> pd.DataFrame:
    """Load one DBN trade file, index by ts_event, drop spread/combo symbols."""
    df = db.DBNStore.from_file(path).to_df()
    df = df.set_index("ts_event")
    df = df[~df["symbol"].str.contains("-")]
    return df


def _process_day(
        prev_df: pd.DataFrame, curr_df: pd.DataFrame,
        out_file: Path, date_str: str, log: callable, p: dict
        ):
    rth_start         = pd.Timestamp(p["rth_start"]).time()
    rth_end           = pd.Timestamp(p["rth_end"]).time()
    pre_rth_threshold = p["pre_rth_threshold"]
    rth_threshold     = p["rth_threshold"]

    # stitch full globex session
    prev_date     = prev_df.index[0].date()
    session_start = pd.Timestamp(f"{prev_date} 22:00:00", tz="UTC")
    session_end   = session_start + pd.Timedelta(hours=23)

    session_df = pd.concat([
        prev_df[prev_df.index >= session_start],
        curr_df[curr_df.index < session_end]
    ]).sort_index()

    # front month
    front_month = _get_front_month(session_df)
    if front_month is None:
        log("No valid symbols found — skipping")
        return None
    session_df = session_df[session_df["symbol"] == front_month]

    # convert to NY time
    session_df.index = session_df.index.tz_convert("America/New_York")

    # require an RTH session (skip holidays / data gaps)
    rth_mask  = (session_df.index.time >= rth_start) & \
                (session_df.index.time <= rth_end)
    if not rth_mask.any():
        log("No RTH bars found — skipping")
        return None

    # filter big trades
    big = session_df[
        ((session_df["size"] >= pre_rth_threshold) & (session_df.index.time < rth_start)) |
        ((session_df["size"] >= rth_threshold)     & (session_df.index.time >= rth_start))
    ][["price", "size", "side"]].copy()

    if big.empty:
        log(f"No big trades found for {date_str} — skipping")
        return None

    out_file.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out_file)
    log(f"✓ Saved {date_str}  ({len(big)} big trades)")
    return big


def run_all(
        input_folder: str, output_folder: str,
        skip_existing: bool = True, on_progress: callable = None,
        params: dict = None,
        ):
    # same merge convention as strategies: UI values over PARAMS defaults
    p = {**PARAMS, **(params or {})}

    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    files       = sorted(input_path.glob("*.dbn.zst"))

    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 files to build a session.")
        return

    total = len(files) - 1

    # Deferred load + forward cache: on a skip we do NOTHING (no load, no decode).
    # prev_df is loaded on demand only when a day actually needs processing, and
    # cached forward so a run of consecutive new days loads each file once.
    prev_df   = None
    prev_file = None   # which Path prev_df currently holds

    for i in range(total):
        current_file = files[i + 1]
        # Filename stem like "glbx-mdp3-20260517.tbbo.dbn" -> ISO date "2026-05-17".
        ymd      = current_file.stem.split(".")[0].split("-")[-1]
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        out_file = output_path / f"{date_str}.parquet"

        def on_log(msg: str, _i=i):
            if on_progress:
                on_progress(_i + 1, total, msg)

        # 0) WEEKEND — a Sat/Sun curr file never has an RTH session, so it never
        # produces an output parquet. There is nothing to skip-against on disk, so
        # without this it would reload (+ its prev) every run. It is still used as
        # the next trading day's prev (loaded on demand below). Always skip it.
        if pd.Timestamp(date_str).weekday() >= 5:
            on_log(f"↷ Skipping {date_str} — weekend (no session)")
            continue

        # 1) SKIP FIRST — zero work; drop any cached prev so we don't hold memory.
        if skip_existing and out_file.exists():
            on_log(f"↷ Skipping {date_str} — already processed")
            if prev_df is not None:
                del prev_df
                gc.collect()
                prev_df, prev_file = None, None
            continue

        # 2) Need to process — ensure prev_df holds files[i] (load on demand, reuse cache).
        if prev_file != files[i]:
            if prev_df is not None:
                del prev_df
                gc.collect()
            prev_df   = _load_and_clean(files[i])
            prev_file = files[i]

        curr_df = _load_and_clean(current_file)

        try:
            _process_day(
                prev_df  = prev_df,
                curr_df  = curr_df,
                out_file = out_file,
                date_str = date_str,
                log      = on_log,
                p        = p,
            )
        except Exception as e:
            if on_progress:
                on_progress(i + 1, total, f"ERROR {current_file.name}: {e}")
            del prev_df
            gc.collect()
            prev_df, prev_file = curr_df, current_file
            continue

        if on_progress:
            on_progress(i + 1, total, "")

        # 3) Cache curr forward as next prev.
        del prev_df
        gc.collect()
        prev_df, prev_file = curr_df, current_file

    if prev_df is not None:
        del prev_df
        gc.collect()