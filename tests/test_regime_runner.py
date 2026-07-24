"""The regime module's invariant tests — the reasons the module exists:

1. snapshot boundary is strict (bars strictly before the stamp);
2. the lookback window never goes stale across skip_existing gaps;
3. gbx_hist_* columns are constant within a file;
4. is_final marks exactly the last row, including on half days;
5. meta.json accumulates and round-trips through a real run;
6. the scaffold detector loads through the REAL plugin loader
   (exec without sys.modules registration — the dataclass gotcha path);
7. cancellation propagates and never leaves a half-written parquet.

All synthetic parquet days — no Qt, no real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modules.common.backend.plugins import PluginRef, list_plugins, load_module
from modules.regime_detector.backend import io as rio
from modules.regime_detector.backend.runner import RegimeContext, SHARED_PARAMS

NY = "America/New_York"
REPO = Path(__file__).resolve().parents[1]
DETECTORS_DIR = REPO / "regime_detectors"


# ── synthetic data ───────────────────────────────────────────────────────────

def make_day(folder: Path, date: str, end_wall: str = "16:59",
             session_open: float = 5000.0, session_close: float = 5010.0,
             symbol: str = "ESH6") -> None:
    """One synthetic globex day: 1m bars from 18:00 the evening before
    through `end_wall` on the RTH date, closes linear from open to close,
    contract symbol in the parquet kv metadata (like the real transforms)."""
    day = pd.Timestamp(date)
    start = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 18:00"
    idx = pd.date_range(pd.Timestamp(start, tz=NY),
                        pd.Timestamp(f"{date} {end_wall}", tz=NY), freq="1min")
    closes = np.linspace(session_open, session_close, len(idx))
    df = pd.DataFrame({
        "open": np.concatenate([[session_open], closes[:-1]]),
        "high": closes + 0.25, "low": closes - 0.25, "close": closes,
        "volume": np.full(len(idx), 100, dtype=np.int64),
    }, index=idx)
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"symbol": symbol.encode()})
    pq.write_table(table, folder / f"{date}.parquet")


def scaffold_ref() -> PluginRef:
    refs = list_plugins([DETECTORS_DIR])
    by_name = {r.name: r for r in refs}
    assert "_scaffold_example" in by_name, "scaffold missing from plugin scan"
    assert "base" not in by_name, "base.py must be excluded from discovery"
    return by_name["_scaffold_example"]


@pytest.fixture(scope="module")
def scaffold():
    """Test 6 (plugin load): the scaffold must load through the REAL loader —
    spec_from_file_location + exec_module WITHOUT sys.modules registration —
    not a normal import, so the `from base import ...` shim and the
    py3.13 dataclass gotcha are actually exercised."""
    module = load_module(scaffold_ref())
    from modules.regime_detector.backend import schema
    assert schema.validate_detector(module) == []
    return module


def test_plugin_load_survives_foreign_base_module():
    """Several plugin folders own a base.py; the first one imported anywhere
    wins sys.modules['base'] (the MC methods folder in the real app). The
    scaffold must load regardless — its imports must never go through the
    contested 'base' name."""
    import sys
    import types
    poisoned = types.ModuleType("base")               # someone else's base.py
    saved = sys.modules.get("base")
    sys.modules["base"] = poisoned
    try:
        module = load_module(scaffold_ref())
        from modules.regime_detector.backend import schema
        assert schema.validate_detector(module) == []
    finally:
        if saved is None:
            del sys.modules["base"]
        else:
            sys.modules["base"] = saved


def run_scaffold(scaffold, input_folder, output_folder, skip_existing=True,
                 on_progress=None, **param_overrides):
    params = {**scaffold.PARAMS, **param_overrides}
    scaffold.run_all(str(input_folder), str(output_folder), skip_existing,
                     on_progress, params)


# ── 1. snapshot boundary ─────────────────────────────────────────────────────

def test_snapshot_boundary_is_strict(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    make_day(src, "2026-01-06")
    run_scaffold(scaffold, src, out)

    bars = pd.read_parquet(src / "2026-01-06.parquet")
    labels = pd.read_parquet(out / "2026-01-06.parquet")

    for ts, row in labels.iterrows():
        used = bars[bars.index < ts]
        assert used.index.max() < ts
        assert row["n_bars_gbx"] == len(used)
        assert row["price"] == used["close"].iloc[-1]

    # the 10:00 row's last bar is the one stamped 09:59
    ten = pd.Timestamp("2026-01-06 10:00", tz=NY)
    assert ten in labels.index
    last_used = bars[bars.index < ten].index.max()
    assert last_used == pd.Timestamp("2026-01-06 09:59", tz=NY)
    # 30 RTH bars at 10:00 (09:30..09:59) — noisier than the close, by design
    assert labels.loc[ten, "n_bars_rth"] == 30

    # grid shape: first snapshot 18:30 the evening before, final 17:00
    assert labels.index[0] == pd.Timestamp("2026-01-05 18:30", tz=NY)
    assert labels.index[-1] == pd.Timestamp("2026-01-06 17:00", tz=NY)


# ── the user-facing point: a positive and a negative day, even ±0.01% ────────

def test_scaffold_labels_positive_and_negative_return(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    make_day(src, "2026-01-06", session_open=5000.0,
             session_close=5000.0 * 1.0001)          # +0.01%
    make_day(src, "2026-01-07", session_open=5000.0,
             session_close=5000.0 * 0.9999)          # -0.01%
    make_day(src, "2026-01-08", session_open=5000.0,
             session_close=5000.0)                   # exactly flat
    run_scaffold(scaffold, src, out)

    final = {d: pd.read_parquet(out / f"{d}.parquet").iloc[-1]
             for d in ("2026-01-06", "2026-01-07", "2026-01-08")}
    assert final["2026-01-06"]["gbx_ret_sign"] == "positive"
    assert final["2026-01-06"]["gbx_ret_pct"] == pytest.approx(0.01)
    assert final["2026-01-07"]["gbx_ret_sign"] == "negative"
    assert final["2026-01-07"]["gbx_ret_pct"] == pytest.approx(-0.01)
    assert final["2026-01-08"]["gbx_ret_sign"] == "flat"
    assert final["2026-01-08"]["gbx_ret_pct"] == 0.0


# ── 2. lookback reconciliation ───────────────────────────────────────────────

def test_lookback_never_stale_across_skip_gaps(tmp_path):
    """The section-8 bug: 300 days, outputs pre-exist for days 1-100 and
    111-150 (1-indexed). With skip_existing, the window used for day 151 must
    contain exactly days 31-150 — push/pop would have lost days 111-150."""
    src, out = tmp_path / "in", tmp_path / "out"
    dates = [d.strftime("%Y-%m-%d")
             for d in pd.date_range("2024-01-01", periods=300, freq="D")]
    # 300 full globex days would be slow — 1-hour stub days are enough here
    src.mkdir(parents=True)
    for i, d in enumerate(dates):
        idx = pd.date_range(pd.Timestamp(f"{d} 09:00", tz=NY), periods=60,
                            freq="1min")
        pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                      "close": float(i), "volume": 1}, index=idx
                     ).to_parquet(src / f"{d}.parquet")

    out.mkdir(parents=True)
    pre_existing = dates[0:100] + dates[110:150]
    for d in pre_existing:
        pd.DataFrame({"x": [1]}).to_parquet(out / f"{d}.parquet")

    windows: dict[str, dict] = {}
    ctx = RegimeContext(src, out, params=dict(SHARED_PARAMS),
                        script_name="probe", script_version="1",
                        states={}, tiers={},
                        summarize=lambda bars: {"c": float(bars["close"].iloc[-1])},
                        skip_existing=True)
    for day in ctx.days():
        window = ctx.lookback(day)
        windows[day] = dict(window)
        if ctx.should_skip(day):
            continue
        ts = ctx.snapshot_times(day)[-1]
        ctx.write(day, [{"ts": ts, "gbx_probe": 1.0}])

    day_151 = dates[150]
    assert list(windows[day_151]) == dates[30:150]
    # summaries must be the real per-day values, not stale copies
    assert windows[day_151][dates[149]] == {"c": 149.0}
    assert windows[day_151][dates[30]] == {"c": 30.0}
    # and every non-skipped day produced output
    assert (out / f"{day_151}.parquet").exists()


# ── 3. gbx_hist_ constancy ───────────────────────────────────────────────────

def test_gbx_hist_columns_constant_within_file(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    for i, d in enumerate(["2026-01-05", "2026-01-06", "2026-01-07"]):
        make_day(src, d, session_close=5000.0 + (i - 1) * 10)
    run_scaffold(scaffold, src, out)

    for f in sorted(out.glob("*.parquet")):
        df = pd.read_parquet(f)
        for col in df.columns:
            if col.startswith("gbx_hist_"):
                assert df[col].nunique(dropna=False) == 1, \
                    f"{f.name}: {col} varies within the day — future leaked"
    meta = rio.read_meta(out)
    assert meta["hist_constant_within_day"]["gbx_hist_up_ratio"] is True


# ── 4. is_final, including half days ─────────────────────────────────────────

def test_is_final_exactly_one_true_on_half_day(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    make_day(src, "2026-07-03", end_wall="12:59")     # 13:00 early close
    run_scaffold(scaffold, src, out)

    df = pd.read_parquet(out / "2026-07-03.parquet")
    finals = df.index[df["is_final"]]
    assert len(finals) == 1
    assert finals[0] == df.index[-1]
    # not hardcoded to a clock time — the half day ends at 13:00, not 17:00
    assert df.index[-1] == pd.Timestamp("2026-07-03 13:00", tz=NY)


# ── 5. meta.json through a real run ──────────────────────────────────────────

def test_meta_accumulates_through_run(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    make_day(src, "2026-01-06", symbol="ESH6")
    make_day(src, "2026-01-07", symbol="ESM6")
    run_scaffold(scaffold, src, out, lookback_days=5)

    meta = rio.read_meta(out)
    assert meta["script_name"] == "_scaffold_example"
    assert meta["script_version"] == scaffold.SCRIPT_VERSION
    assert meta["params"]["lookback_days"] == 5
    assert meta["counts"]["processed"] == 2
    assert meta["counts"]["skipped"] == 0
    assert meta["counts"]["short_lookback"] == 2      # dataset start
    assert meta["contracts"] == ["ESH6", "ESM6"]
    assert meta["date_range"] == {"first": "2026-01-06", "last": "2026-01-07"}
    assert meta["globex_session"] == {"start": "18:00", "end": "17:00"}
    assert meta["schema"]["regime"] == ["gbx_ret_sign"]
    assert meta["states"]["gbx_ret_sign"]["states"] == \
        ["positive", "flat", "negative"]
    assert meta["meta_version"] == 1

    # resume: everything already on disk -> only skip counts move, and the
    # dataset-level facts (contracts, session) survive the rewrite
    run_scaffold(scaffold, src, out, lookback_days=5)
    meta2 = rio.read_meta(out)
    assert meta2["counts"]["skipped"] == 2
    assert meta2["counts"]["processed"] == 0
    assert meta2["contracts"] == ["ESH6", "ESM6"]
    assert meta2["globex_session"] == {"start": "18:00", "end": "17:00"}


# ── 7. cancellation ──────────────────────────────────────────────────────────

class _Cancelled(Exception):
    pass


def test_cancellation_leaves_no_half_written_file(tmp_path, scaffold):
    src, out = tmp_path / "in", tmp_path / "out"
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    for d in dates:
        make_day(src, d)

    def on_progress(current, total, message):
        if current == 2:                              # cancel before day 3
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        run_scaffold(scaffold, src, out, on_progress=on_progress)

    written = sorted(f.name for f in out.glob("*.parquet"))
    assert written == ["2026-01-05.parquet", "2026-01-06.parquet"]
    assert not list(out.glob("*.tmp"))                # atomic writes only
    for f in out.glob("*.parquet"):
        df = pd.read_parquet(f)                       # every file is readable
        assert df["is_final"].sum() == 1

    # resume picks up where the cancel stopped
    run_scaffold(scaffold, src, out)
    assert sorted(f.stem for f in out.glob("*.parquet")) == dates
    meta = rio.read_meta(out)
    assert meta["counts"]["skipped"] == 2
    assert meta["counts"]["processed"] == 2
