"""
RegimeContext — the pull-style helper that owns everything a detector script
shouldn't have to get right twice:

- the trading calendar (the sorted input file list IS the calendar);
- the lookback window as "what do I need vs what do I have" reconciliation
  (a dict of date -> summary, NEVER push/pop — push/pop silently goes stale
  across skip_existing gaps and produces wrong numbers with no error);
- the clock-anchored snapshot grid derived from each file's own bar range
  (half days and DST just produce fewer/more snapshots);
- slice(before=ts) — the ONLY implementation of the strict-before rule: the
  snapshot stamped T uses bars strictly before T, so the 10:00 row's last
  bar is the one stamped 09:59;
- atomic daily writes (tmp + os.replace — a cancellation mid-run never
  leaves a half-written parquet) and incremental meta.json accumulation.

Cancellation: on_progress(current, total, message) is the universal callback;
the UI raises inside it (modules/common/ui/workers.py). days() calls it once
per day, so the exception propagates out of the script's run_all with the
window dict and partial output intact — skip_existing resumes the run.

Detector scripts drive their own loop (see regime_detectors/_scaffold_example.py):

    ctx = RegimeContext(input_folder, output_folder, params=params, ...)
    for day in ctx.days():
        window = ctx.lookback(day)          # {date: summarize(bars)} dict
        if ctx.should_skip(day):
            continue                        # window already reconciled
        bars = ctx.bars(day)
        rows = [my_compute(ctx.slice(bars, before=ts), window, ts)
                for ts in ctx.snapshot_times(day)]
        ctx.write(day, rows)                # each row dict includes "ts"
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import io as rio
from . import schema

NY = "America/New_York"

# The shared param defaults every detector spreads into its own PARAMS
# (via regime_detectors/base.py) so rth_start etc. never drift per script.
SHARED_PARAMS = {
    "rth_start": "09:30",
    "rth_end": "16:00",
    "snapshot_minutes": 30,
    "lookback_days": 120,
}


def read_contract(path: Path) -> str:
    """Contract symbol from the input parquet's kv metadata — footer-only
    read. Enriched sets store it as 'symbol'; some transforms only write
    'front_month' (same value on non-roll days), so fall back to that."""
    try:
        meta = pq.read_metadata(path).metadata or {}
    except Exception:  # noqa: BLE001 — a bad footer just means "unknown"
        return ""
    for key in (b"symbol", b"front_month"):
        if key in meta:
            return meta[key].decode("utf-8", errors="replace")
    return ""


def snapshot_grid(index: pd.DatetimeIndex,
                  minutes: int) -> pd.DatetimeIndex:
    """Clock-anchored snapshot grid from a day's bar index: every `minutes`
    boundary STRICTLY AFTER the first bar, through the first boundary
    strictly after the last bar. Shared by RegimeContext.snapshot_times and
    by detector summarize() functions that need the same grid on prior days."""
    if not len(index):
        return pd.DatetimeIndex([], tz=NY)
    step = pd.Timedelta(minutes=minutes)
    first, last = index[0], index[-1]
    start = first.ceil(step)
    if start <= first:
        start += step
    end = last.ceil(step)
    if end <= last:
        end += step
    return pd.date_range(start, end, freq=step)


def _merge_constancy(state: str | None, constant_this_file: bool) -> str:
    """Aggregate per-file constancy into 'all' / 'some' / 'none'. 'some' is a
    signal, not an error — constant on 306 days and varying on one means
    something happened that day."""
    if state is None:
        return "all" if constant_this_file else "none"
    if state == "all" and not constant_this_file:
        return "some"
    if state == "none" and constant_this_file:
        return "some"
    return state


def _derive_root_and_asset(input_folder: Path) -> tuple[str, str]:
    """(data_root, ASSET) from a .../parquet/{type}/{ASSET}/{dataset} path;
    empty strings when the path doesn't follow the blueprint."""
    parts = Path(input_folder).parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "parquet":
            root = str(Path(*parts[:i])) if i else ""
            asset = parts[i + 2] if i + 2 < len(parts) - 1 else ""
            return root, asset
    return "", ""


class RegimeContext:
    """One run of one detector over one dataset. Pure computation — safe in
    worker threads and tests; never imports Qt."""

    def __init__(self, input_folder, output_folder, params: dict,
                 script_name: str, script_version: str,
                 states: dict, tiers: dict,
                 summarize=None, skip_existing: bool = True,
                 on_progress=None, script_file: str = "",
                 constant_columns=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.params = dict(params)
        self.summarize = summarize
        self.skip_existing = bool(skip_existing)
        self.on_progress = on_progress

        self.rth_start = str(self.params["rth_start"])
        self.rth_end = str(self.params["rth_end"])
        self.snapshot_minutes = int(self.params["snapshot_minutes"])
        self.lookback_days = int(self.params["lookback_days"])
        self._step = pd.Timedelta(minutes=self.snapshot_minutes)

        self._files = rio.day_files(self.input_folder)
        self._dates = list(self._files)                 # sorted by day_files
        self._pos = {d: i for i, d in enumerate(self._dates)}
        self._window: dict[str, dict] = {}              # date -> summary

        # per-day state, reset by days() each iteration
        self._day: str | None = None
        self._bars: pd.DataFrame | None = None
        self._contract: str = ""
        self._lookback_used: int | None = None

        # meta accumulation
        self._t0 = time.time()
        self._counts = {"processed": 0, "skipped": 0, "missing_input": 0,
                        "short_lookback": 0}
        self._contracts: list[str] = []
        self._session_start: str | None = None          # earliest first-bar wall time
        self._session_end: str | None = None            # latest grid-end wall time
        # measured per-column within-day constancy, aggregated across files:
        # 'all' / 'some' / 'none' (a fact about the output — never inferred
        # from the column name)
        self._col_state: dict[str, str] = {}
        self._constant_columns = list(constant_columns or [])
        # free-form dict a script may fill (e.g. chain-reset days); lands in
        # meta.json under "script_extras"
        self.meta_extra: dict = {}
        self._tiers = {tier: list(tiers.get(tier, [])) for tier in schema.TIERS}
        self._meta_static = {
            "script_name": script_name,
            "script_version": script_version,
            "script_file": str(script_file),
            "params": dict(params),
            "input_dataset_path": str(self.input_folder),
            "states": states,
        }
        root, asset = _derive_root_and_asset(self.input_folder)
        self._meta_static["input_data_root"] = root
        self._meta_static["asset"] = asset
        self._meta_static["run_started"] = pd.Timestamp.now().isoformat()
        self._states = states

        # Resuming into an existing run: dataset-level facts (session bounds,
        # contracts seen, measured column constancy) carry over from the
        # previous meta — a resume that skips every day must not blank them.
        try:
            prev = rio.read_meta(self.output_folder)
            self._session_start = prev.get("globex_session", {}).get("start")
            self._session_end = prev.get("globex_session", {}).get("end")
            self._contracts = list(prev.get("contracts", []))
            prev_cols = prev.get("schema", {}).get("columns", {})
            if isinstance(prev_cols, dict):
                self._col_state = {
                    col: info["constant_within_day"]
                    for col, info in prev_cols.items()
                    if isinstance(info, dict)
                    and info.get("constant_within_day") in ("all", "some",
                                                            "none")
                }
        except (OSError, ValueError):
            pass

    # ── the day walk ─────────────────────────────────────────────────────────
    def days(self):
        """Yield RTH dates (YYYY-MM-DD strings) in order. Reports progress /
        honors cancellation once per day; writes the final meta.json when the
        loop completes."""
        total = len(self._dates)
        for i, day in enumerate(self._dates):
            if self.on_progress is not None:
                self.on_progress(i, total, day)
            self._day = day
            self._bars = None
            self._contract = ""
            self._lookback_used = None
            yield day
        self.write_meta()
        if self.on_progress is not None:
            for warning in self._constancy_warnings():
                self.on_progress(total, total, f"WARNING: {warning}")
            self.on_progress(total, total, "done")

    def out_path(self, day: str) -> Path:
        return self.output_folder / f"{day}.parquet"

    def should_skip(self, day: str) -> bool:
        """True when the output already exists and skip_existing is on. Call
        AFTER lookback() — the window is already reconciled, which is exactly
        what makes skipping safe (see the module docstring)."""
        if self.skip_existing and self.out_path(day).exists():
            self._counts["skipped"] += 1
            self.write_meta()
            return True
        return False

    # ── lookback window ──────────────────────────────────────────────────────
    def lookback(self, day: str) -> dict[str, dict]:
        """{date: summary} for the `lookback_days` trading days immediately
        before `day`, reconciled need-vs-have: reads only missing days, drops
        days that rolled out. Never stale by construction."""
        if self.summarize is None:
            raise RuntimeError("RegimeContext needs summarize= to build "
                               "lookback windows")
        i = self._pos[day]
        need = self._dates[max(0, i - self.lookback_days):i]
        need_set = set(need)
        for stale in [d for d in self._window if d not in need_set]:
            del self._window[stale]
        for d in need:
            if d not in self._window:
                self._window[d] = self.summarize(pd.read_parquet(self._files[d]))
        self._lookback_used = len(need)
        return {d: self._window[d] for d in need}       # date order

    # ── per-day data ─────────────────────────────────────────────────────────
    def bars(self, day: str) -> pd.DataFrame:
        """The day's bars (cached for the current day) + contract metadata."""
        if self._day == day and self._bars is not None:
            return self._bars
        path = self._files[day]
        df = pd.read_parquet(path)
        contract = read_contract(path)
        self._day, self._bars, self._contract = day, df, contract
        if contract and contract not in self._contracts:
            self._contracts.append(contract)
        if len(df):
            first_wall = df.index[0].strftime("%H:%M")
            if self._session_start is None or \
                    self._session_key(first_wall) < self._session_key(self._session_start):
                self._session_start = first_wall
        return df

    @staticmethod
    def _session_key(wall: str) -> str:
        """Sort key treating evening times (>= 17:00) as earlier than the
        after-midnight portion — 18:00 opens a session that 03:00 continues."""
        return ("0" + wall) if wall >= "17:00" else ("1" + wall)

    def contract(self, day: str) -> str:
        self.bars(day)
        return self._contract

    def snapshot_times(self, day: str) -> pd.DatetimeIndex:
        """Clock-anchored grid from the file's own bar range (see
        snapshot_grid); the last boundary sees the whole session and becomes
        the final row."""
        bars = self.bars(day)
        grid = snapshot_grid(bars.index, self.snapshot_minutes)
        if len(grid):
            wall = grid[-1].strftime("%H:%M")
            if self._session_end is None or \
                    self._session_key(wall) > self._session_key(self._session_end):
                self._session_end = wall
        return grid

    @staticmethod
    def slice(bars: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
        """Bars STRICTLY before `before` — the one implementation of the
        snapshot-boundary rule. The 10:00 slice ends with the 09:59 bar; the
        10:00 bar isn't finished until 10:01 and belongs to the next snapshot."""
        return bars.iloc[:bars.index.searchsorted(before, side="left")]

    # ── output ───────────────────────────────────────────────────────────────
    def write(self, day: str, rows: list[dict]) -> Path | None:
        """Assemble, validate and atomically write the day's frame. Each row
        dict must carry its snapshot timestamp under "ts"; the always-present
        columns the script didn't set are filled in here from the day's bars
        and the current lookback window."""
        if not rows:
            self._counts["missing_input"] += 1
            self.write_meta()
            return None
        bars = self.bars(day)

        df = pd.DataFrame(rows)
        if "ts" not in df.columns:
            raise ValueError("every row dict must include 'ts' (the snapshot "
                             "timestamp)")
        idx = pd.DatetimeIndex(df.pop("ts"))
        idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
        df.index = idx
        df = df.sort_index()

        positions = bars.index.searchsorted(df.index, side="left")
        if "price" not in df.columns:
            closes = bars["close"].to_numpy(dtype=float)
            df["price"] = [closes[p - 1] if p > 0 else np.nan
                           for p in positions]
        if "n_bars_gbx" not in df.columns:
            df["n_bars_gbx"] = positions.astype(int)
        if "n_bars_rth" not in df.columns:
            lo = pd.Timestamp(f"{day} {self.rth_start}", tz=NY)
            hi = pd.Timestamp(f"{day} {self.rth_end}", tz=NY)
            in_rth = ((bars.index >= lo) & (bars.index < hi)).cumsum()
            in_rth = np.concatenate([[0], in_rth])      # prefix counts
            df["n_bars_rth"] = in_rth[positions].astype(int)
        if "contract" not in df.columns:
            df["contract"] = self._contract
        if "lookback_days_used" not in df.columns:
            df["lookback_days_used"] = int(self._lookback_used or 0)
        df["is_final"] = False
        df.iloc[-1, df.columns.get_loc("is_final")] = True

        errors = schema.validate_day_frame(df, self._states)
        if errors:
            raise ValueError(f"{day}: invalid regime frame — "
                             + "; ".join(errors))

        # constancy is measured for EVERY column, never inferred from names
        for col in df.columns:
            constant = bool(df[col].nunique(dropna=False) == 1)
            self._col_state[col] = _merge_constancy(
                self._col_state.get(col), constant)

        self.output_folder.mkdir(parents=True, exist_ok=True)
        out = self.out_path(day)
        tmp = out.with_name(out.name + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, out)

        self._counts["processed"] += 1
        if self._lookback_used is not None \
                and self._lookback_used < self.lookback_days:
            self._counts["short_lookback"] += 1
        self.write_meta()
        return out

    # ── meta.json ────────────────────────────────────────────────────────────
    def _constancy_warnings(self) -> list[str]:
        """Declared-vs-measured mismatches. A warning, never a failure — the
        declaration is the script author's expectation, the measurement is
        the fact. Declared columns never observed are skipped (param-named
        columns like vol_har_5 may legitimately not exist for this run)."""
        return [
            f"column '{col}' was declared constant-within-day but measured "
            f"'{self._col_state[col]}'"
            for col in self._constant_columns
            if col in self._col_state and self._col_state[col] != "all"
        ]

    def write_meta(self) -> None:
        """Rewrite meta.json (tiny — rewritten after every day, so a crash or
        cancellation always leaves counts matching the files on disk)."""
        p = self.params
        meta = {
            **self._meta_static,
            "rth_start": p["rth_start"], "rth_end": p["rth_end"],
            "snapshot_minutes": p["snapshot_minutes"],
            "lookback_days": p["lookback_days"],
            "globex_session": {"start": self._session_start,
                               "end": self._session_end},
            "date_range": {"first": self._dates[0] if self._dates else None,
                           "last": self._dates[-1] if self._dates else None},
            "counts": dict(self._counts),
            "contracts": list(self._contracts),
            "schema": {
                "tiers": {t: list(cols) for t, cols in self._tiers.items()},
                "columns": {
                    col: {
                        "scope": schema.parse_column(col)[0],
                        "is_hist": schema.parse_column(col)[1],
                        "constant_within_day": state,
                    }
                    for col, state in sorted(self._col_state.items())
                },
            },
            "warnings": self._constancy_warnings(),
            "run_duration_s": round(time.time() - self._t0, 1),
            "meta_version": 2,
        }
        if self.meta_extra:
            meta["script_extras"] = dict(self.meta_extra)
        rio.write_meta(self.output_folder, meta)
