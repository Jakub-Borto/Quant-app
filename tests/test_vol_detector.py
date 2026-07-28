"""Tests for regime_detectors/vol_realized_z.py (detector plan §10).

All synthetic. Days are built with alternating ±σ log returns, so every
window's per-bar volatility is exactly its σ by construction — no randomness,
no tolerance games. The module is loaded through the REAL plugin loader.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.common.backend.plugins import list_plugins, load_module
from modules.regime_detector.backend import io as rio
from modules.regime_detector.backend import schema

NY = "America/New_York"
REPO = Path(__file__).resolve().parents[1]
DETECTORS_DIR = REPO / "regime_detectors"

# small, fast param set used by most tests (defaults need 60+ days)
FAST = {"lookback_days": 8, "min_lookback_days": 4, "har_horizons": "1,2,3"}


def _load():
    refs = {r.name: r for r in list_plugins([DETECTORS_DIR])}
    return load_module(refs["vol_realized_z"])


@pytest.fixture(scope="module")
def vol():
    module = _load()
    assert schema.validate_detector(module) == []
    return module


def make_vol_day(folder: Path, date: str, sigma_on=0.001, sigma_rth=0.001,
                 sigma_morning=None, end_wall="16:59", drop=(),
                 zero_vol=(), base=5000.0) -> None:
    """One synthetic globex day (18:00 prev evening → end_wall) whose log
    returns alternate ±σ: overnight bars use sigma_on, RTH bars sigma_rth,
    and optionally the first 30 RTH minutes use sigma_morning. `drop` removes
    wall-clock minutes ("HH:MM" on the RTH date) to create data gaps."""
    day = pd.Timestamp(date)
    prev = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    idx = pd.date_range(pd.Timestamp(f"{prev} 18:00", tz=NY),
                        pd.Timestamp(f"{date} {end_wall}", tz=NY), freq="1min")
    if drop:
        gone = {pd.Timestamp(f"{date} {m}", tz=NY) for m in drop}
        idx = idx[~idx.isin(list(gone))]

    lo = pd.Timestamp(f"{date} 09:30", tz=NY)
    morning_hi = lo + pd.Timedelta(minutes=30)
    sig = np.full(len(idx), float(sigma_on))
    sig[np.asarray(idx >= lo)] = float(sigma_rth)
    if sigma_morning is not None:
        sig[np.asarray((idx >= lo) & (idx < morning_hi))] = float(sigma_morning)

    r = sig * np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)
    r[0] = 0.0
    closes = base * np.exp(np.cumsum(r))
    volume = np.full(len(idx), 100, dtype=np.int64)
    if zero_vol:
        zv = {pd.Timestamp(f"{date} {m}", tz=NY) for m in zero_vol}
        volume[np.asarray(idx.isin(list(zv)))] = 0
    df = pd.DataFrame({
        "open": np.concatenate([[base], closes[:-1]]),
        "high": closes * 1.0001, "low": closes * 0.9999, "close": closes,
        "volume": volume,
    }, index=idx)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{date}.parquet")


def run(vol, src, out, skip=True, **overrides):
    params = {**vol.PARAMS, **FAST, **overrides}
    vol.run_all(str(src), str(out), skip, None, params)


def dates_from(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, periods=n, freq="D")]


# ── 1. log space ─────────────────────────────────────────────────────────────

def test_yardstick_center_is_log_not_raw(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 9)
    sigmas = [0.0010 * (1.0 + 0.06 * (i % 3)) for i in range(len(days))]
    for d, s in zip(days, sigmas):
        make_vol_day(src, d, sigma_on=s, sigma_rth=s)
    run(vol, src, out)

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    prior = sigmas[:-1]                                   # all 8 in lookback
    expected_ln = np.median(np.log(prior))
    got = df["on_hist_vol_center"].iloc[-1]
    assert got == pytest.approx(expected_ln, abs=1e-9)
    assert abs(got - np.median(prior)) > 1.0              # NOT the raw value
    # daily center too, and σ=0 never became -inf anywhere
    assert df["rth_hist_daily_vol_center"].iloc[-1] == pytest.approx(
        expected_ln, abs=1e-9)
    for col in ("vol_z", "vol_z_obs", "vol_z_fcst"):
        assert not np.isinf(df[col].to_numpy(dtype=float)).any()


# ── 2. excluded returns ──────────────────────────────────────────────────────

def test_excluded_returns_counted_and_kept_out(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    halt = [f"16:{m:02d}" for m in range(15, 30)]         # break-style halt
    make_vol_day(src, "2026-01-06", sigma_on=0.001, sigma_rth=0.001,
                 drop=["13:00", "13:01", "13:02"] + halt)  # + a plain data gap
    run(vol, src, out)

    df = pd.read_parquet(out / "2026-01-06.parquet")
    final = df.iloc[-1]
    # three excluded: first bar of session, the 13:03 gap re-entry, the
    # 16:30 halt re-entry
    assert final["diag_excluded_returns"] == 3
    assert final["n_bars_gbx"] - final["gbx_vol_n"] == 3
    # none of them entered the sum: every valid return is ±σ, so per-bar
    # vol is exactly σ — a gap return (3 or 15 minutes of drift) would move it
    assert final["gbx_vol"] == pytest.approx(0.001, rel=1e-9)


# ── 3. forecast / daily-history constancy ────────────────────────────────────

def test_fcst_and_daily_hist_constant_within_file(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 12)
    for i, d in enumerate(days):
        s = 0.0010 * (1.0 + 0.05 * (i % 4))
        make_vol_day(src, d, sigma_on=s, sigma_rth=s)
    run(vol, src, out)

    for f in sorted(out.glob("*.parquet")):
        df = pd.read_parquet(f)
        for col in ("vol_z_fcst", "rth_hist_daily_vol_center",
                    "rth_hist_daily_vol_scale", "vol_har_1", "vol_har_2",
                    "vol_har_3"):
            assert df[col].nunique(dropna=False) == 1, (f.name, col)

    meta = rio.read_meta(out)
    cols = meta["schema"]["columns"]
    assert cols["vol_z_fcst"]["constant_within_day"] == "all"
    assert cols["rth_hist_daily_vol_center"]["constant_within_day"] == "all"
    # matched-window history legitimately VARIES by snapshot — the reason
    # constancy is measured, not inferred from the _hist_ prefix
    assert cols["rth_hist_vol_center"]["constant_within_day"] != "all"
    assert meta["warnings"] == []


# ── 4. on_vol freezes at the open ────────────────────────────────────────────

def test_overnight_frozen_from_0930(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_vol_day(src, d, sigma_on=0.0008, sigma_rth=0.0020)
    run(vol, src, out)

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    lo = pd.Timestamp(f"{days[-1]} 09:30", tz=NY)
    post = df[df.index >= lo]
    assert len(post) > 5
    assert post["on_vol"].nunique() == 1
    assert post["on_vol"].iloc[0] == pytest.approx(0.0008, rel=1e-9)
    assert post["on_vol_n"].nunique() == 1
    assert post["vol_z_on"].nunique(dropna=False) == 1
    # pre-open it accumulates (globex == overnight before the open)
    pre = df[df.index < lo]
    assert (pre["on_vol_n"].to_numpy()[1:]
            > pre["on_vol_n"].to_numpy()[:-1]).all()


# ── 5. matched windows ───────────────────────────────────────────────────────

def test_1000_yardstick_uses_prior_mornings_only(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 9)
    for i, d in enumerate(days):
        wiggle = 1.0 + 0.04 * (i % 3)
        make_vol_day(src, d, sigma_on=0.0008 * wiggle,
                     sigma_rth=0.0010 * wiggle,
                     sigma_morning=0.0040 * wiggle)       # wild first 30 min
    run(vol, src, out)

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    ten = df.loc[pd.Timestamp(f"{days[-1]} 10:00", tz=NY)]

    wiggles = [1.0 + 0.04 * (i % 3) for i in range(len(days) - 1)]
    expected_morning = np.median(np.log([0.0040 * w for w in wiggles]))
    assert ten["rth_hist_vol_center"] == pytest.approx(expected_morning,
                                                       abs=1e-9)
    # nowhere near a full-day yardstick (mostly 0.0010-σ bars)
    full_day_ln = np.log(0.0012)
    assert abs(ten["rth_hist_vol_center"] - full_day_ln) > 0.5
    # and today's own morning measures exactly its σ
    assert ten["rth_vol"] == pytest.approx(0.0040 * (1.0 + 0.04 * (8 % 3)),
                                           rel=1e-9)


# ── 6. hysteresis ────────────────────────────────────────────────────────────

def test_schmitt_trigger_sequence(vol):
    z = [0.5, 0.8, 0.5, 0.3, 0.9, -0.8, -0.5, -0.2, 0.0, np.nan, 0.4]
    got = vol._schmitt(z, "normal", 0.75, -0.75, 0.35, -0.35)
    assert got == ["normal",          # 0.5 below enter_high
                   "high",            # 0.8 enters
                   "high",            # 0.5 above exit_high — dead zone holds
                   "normal",          # 0.3 exits
                   "high",            # 0.9 re-enters
                   "low",             # -0.8 exits high AND enters low
                   "low",             # -0.5 below exit_low — holds
                   "normal",          # -0.2 exits
                   "normal",
                   "unknown",         # NaN — no answer, chain untouched
                   "normal"]          # 0.4 continues from normal, not unknown

def test_schmitt_seed_respected(vol):
    # dead-zone z: only the seed decides
    assert vol._schmitt([0.5], "high", 0.75, -0.75, 0.35, -0.35) == ["high"]
    assert vol._schmitt([0.5], "normal", 0.75, -0.75, 0.35, -0.35) == ["normal"]
    assert vol._schmitt([-0.5], "low", 0.75, -0.75, 0.35, -0.35) == ["low"]
    # unknown / garbage seeds fall back to normal
    assert vol._schmitt([0.5], "unknown", 0.75, -0.75, 0.35, -0.35) == ["normal"]


# ── 7. state chain across files ──────────────────────────────────────────────

def test_state_chain_reads_previous_file_and_resets_on_gap(tmp_path):
    vol = _load()                     # private instance — we wrap _schmitt
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 8)
    for i, d in enumerate(days):
        s = 0.0010 * (1.0 + 0.05 * (i % 4))
        make_vol_day(src, d, sigma_on=s, sigma_rth=s)

    seeds = []
    orig = vol._schmitt
    vol._schmitt = lambda z, seed, *a: (seeds.append(seed), orig(z, seed, *a))[1]
    run(vol, src, out)

    # day k's seed is day k-1's final written state, straight from the file
    finals = {d: pd.read_parquet(out / f"{d}.parquet")["vol_state"].iloc[-1]
              for d in days}
    for i in range(1, len(days)):
        prev_final = finals[days[i - 1]]
        expected = prev_final if prev_final in ("low", "normal", "high") \
            else "normal"
        assert seeds[i] == expected
    # resets: the first day, plus every day whose predecessor ended unknown
    # (warm-up days do — an unknown final state cannot seed a chain)
    expected_resets = [days[0]] + [
        days[i] for i in range(1, len(days))
        if finals[days[i - 1]] not in ("low", "normal", "high")]
    meta = rio.read_meta(out)
    assert meta["script_extras"]["chain_resets"] == expected_resets
    assert len(expected_resets) < len(days)           # chain did run intact
    intact_day = days[-1]                             # predecessor was known
    assert finals[days[-2]] in ("low", "normal", "high")
    df_ok = pd.read_parquet(out / f"{intact_day}.parquet")
    assert not df_ok["diag_state_chain_reset"].any()

    # break the chain: day 1's input and output disappear, day 2 recomputed
    (src / f"{days[0]}.parquet").unlink()
    (out / f"{days[0]}.parquet").unlink()
    (out / f"{days[1]}.parquet").unlink()
    seeds.clear()
    run(vol, src, out)                # skip_existing resumes the rest
    assert seeds == ["normal"]        # no yesterday -> reset
    d2 = pd.read_parquet(out / f"{days[1]}.parquet")
    assert d2["diag_state_chain_reset"].all()
    meta = rio.read_meta(out)
    assert days[1] in meta["script_extras"]["chain_resets"]


# ── 8. warm-up ───────────────────────────────────────────────────────────────

def test_warmup_is_unknown_never_nan(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    for d in dates_from("2026-01-05", 3):
        make_vol_day(src, d)
    # defaults: min_lookback_days=60 — 3 days is deep warm-up
    vol.run_all(str(src), str(out), True, None, dict(vol.PARAMS))

    for f in sorted(out.glob("*.parquet")):
        df = pd.read_parquet(f)
        assert set(df["vol_state"].unique()) == {"unknown"}
        assert df["vol_state"].notna().all()
        assert not df["vol_state"].isin(["nan", "None", ""]).any()


# ── param guards ─────────────────────────────────────────────────────────────

def test_lookback_must_exceed_har_horizon(tmp_path, vol):
    src, out = tmp_path / "in", tmp_path / "out"
    make_vol_day(src, "2026-01-06")
    with pytest.raises(ValueError, match="lookback_days"):
        vol.run_all(str(src), str(out), True, None,
                    {**vol.PARAMS, "lookback_days": 10,
                     "har_horizons": "1,5,22"})


def test_parse_horizons(vol):
    assert vol._parse_horizons("1,5,22") == [1, 5, 22]
    assert vol._parse_horizons("22, 1, 5") == [1, 5, 22]
    with pytest.raises(ValueError):
        vol._parse_horizons("1,zwei")
    with pytest.raises(ValueError):
        vol._parse_horizons("0,5")
