"""Tests for regime_detectors/mr_multi.py (detector plan §10).

All synthetic. Where a measure has a closed-form answer the test asserts it
exactly (alternating returns, fixed drift); where it doesn't, the test pins the
hand-rolled batched implementation against the readable reference in the module
and — for the ADF — against statsmodels. The module is loaded through the REAL
plugin loader, which is what catches the dataclass/annotation class of bug that
a plain import misses.

Note on test 4 (ADF sign): the plan's draft asserted a mean-reverting spread
gives a NEGATIVE raw stat but a POSITIVE trendiness z. That is the bug it was
written to prevent — the ADF t-statistic is already monotone increasing in
trendiness, so a reverting spread must read negative in BOTH. The test below
asserts that, and asserts reversion never reads as trend.
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
FAST = {"lookback_days": 8, "min_lookback_days": 4}


def _load():
    refs = {r.name: r for r in list_plugins([DETECTORS_DIR])}
    return load_module(refs["mr_multi"])


@pytest.fixture(scope="module")
def mr():
    module = _load()
    assert schema.validate_detector(module) == []
    return module


# ── synthetic days ───────────────────────────────────────────────────────────

def returns_for(kind: str, n: int, sigma: float, seed: int = 0) -> np.ndarray:
    """Return series with a KNOWN reversion character.

    alternating — +σ, −σ, +σ ... perfectly negatively autocorrelated, so any
                  multi-bar move cancels: VR far below 1.
    runs        — σ with a sign that flips only every 20 bars, so moves compound
                  over any q <= 10: VR above 1. (A constant DRIFT would not do
                  this — demeaning removes it and VR stays ~1, which is correct
                  and is why the drift case is tested separately.)
    walk        — iid gaussian: VR ~ 1.
    """
    if kind == "alternating":
        return sigma * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if kind == "runs":
        return sigma * np.where((np.arange(n) // 20) % 2 == 0, 1.0, -1.0)
    if kind == "drift":
        return np.full(n, sigma)
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def make_day(folder: Path, date: str, kind="walk", kind_morning=None,
             sigma=0.0008, base=5000.0, end_wall="16:59", drop=(),
             seed=0) -> None:
    """One synthetic globex day (18:00 prev evening -> end_wall). `kind_morning`
    overrides the first 30 RTH minutes, which is what the matched-window test
    needs: a 10:00 yardstick must see only 09:30-09:59."""
    day = pd.Timestamp(date)
    prev = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    idx = pd.date_range(pd.Timestamp(f"{prev} 18:00", tz=NY),
                        pd.Timestamp(f"{date} {end_wall}", tz=NY), freq="1min")
    if drop:
        gone = {pd.Timestamp(f"{date} {m}", tz=NY) for m in drop}
        idx = idx[~idx.isin(list(gone))]

    n = len(idx)
    r = returns_for(kind, n, sigma, seed)
    if kind_morning is not None:
        lo = pd.Timestamp(f"{date} 09:30", tz=NY)
        hi = lo + pd.Timedelta(minutes=30)
        mask = np.asarray((idx >= lo) & (idx < hi))
        r = np.where(mask, returns_for(kind_morning, n, sigma, seed + 1), r)
    r = np.asarray(r, dtype=float).copy()
    r[0] = 0.0

    closes = base * np.exp(np.cumsum(r))
    df = pd.DataFrame({
        "open": np.concatenate([[base], closes[:-1]]),
        "high": closes * 1.0002, "low": closes * 0.9998, "close": closes,
        "volume": np.full(n, 100, dtype=np.int64),
    }, index=idx)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{date}.parquet")


def run(mr, src, out, skip=True, **overrides):
    mr.run_all(str(src), str(out), skip, None, {**mr.PARAMS, **FAST,
                                                **overrides})


def dates_from(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, periods=n, freq="D")]


def row_at(df: pd.DataFrame, date: str, clock: str) -> pd.Series:
    return df.loc[pd.Timestamp(f"{date} {clock}", tz=NY)]


# ── 1. q=1 rejected at parse ─────────────────────────────────────────────────

def test_q_value_1_rejected(mr):
    assert mr._parse_q_values("2,5,10") == [2, 5, 10]
    assert mr._parse_q_values("10, 2") == [2, 10]
    with pytest.raises(ValueError, match="VR\\(1\\) is exactly 1.0"):
        mr._parse_q_values("1,5")
    with pytest.raises(ValueError, match="comma-separated ints"):
        mr._parse_q_values("2,zwei")


# ── 2. VR sign ───────────────────────────────────────────────────────────────

def test_vr_sign_trending_reverting_walk(mr):
    n = 600
    ones = np.ones(n, dtype=bool)
    ones[0] = False

    reverting = mr._vr_reference(returns_for("alternating", n, 1e-3), ones, 10)
    trending = mr._vr_reference(returns_for("runs", n, 1e-3), ones, 10)
    walk = mr._vr_reference(returns_for("walk", 4000, 1e-3, seed=7),
                            np.r_[False, np.ones(3999, dtype=bool)], 10)

    assert reverting[0] < 0.2, reverting          # moves cancel
    assert trending[0] > 2.0, trending            # moves compound
    assert walk[0] == pytest.approx(1.0, abs=0.15), walk
    # and the significance z agrees with the direction
    assert reverting[1] < -2 and trending[1] > 2
    assert abs(walk[1]) < 2

    # DRIFT is not autocorrelation. A walk with a strong constant drift is a
    # trend to the eye (and to ER and ADF against a fixed anchor), but VR
    # demeans, so it correctly reads ~1. This is the measure behaving as
    # specified, not a miss — the two families answer different questions.
    walk_r = returns_for("walk", 4000, 1e-3, seed=7)
    drifted = mr._vr_reference(walk_r + 1e-3, np.r_[False, np.ones(3999, bool)],
                               10)
    assert drifted[0] == pytest.approx(walk[0], rel=1e-6), (drifted, walk)

    # a perfectly flat window (no movement at all) has no answer, not a number
    flat = mr._vr_reference(np.zeros(n), ones, 10)
    assert np.isnan(flat[0])


# ── 3. overlapping, not disjoint, blocks ─────────────────────────────────────

def test_vr_blocks_are_overlapping(tmp_path, mr):
    n, q = 300, 10
    valid = np.r_[False, np.ones(n - 1, dtype=bool)]
    _vr, _z, blocks = mr._vr_reference(returns_for("walk", n, 1e-3), valid, q)
    n_valid = n - 1
    assert blocks == n_valid - q + 1            # overlapping
    assert blocks != n_valid // q               # NOT disjoint chunks

    # and through the run path, where the count is a stored diagnostic
    src, out = tmp_path / "in", tmp_path / "out"
    for d in dates_from("2026-01-05", 6):
        make_day(src, d)
    run(mr, src, out)
    day = dates_from("2026-01-05", 6)[-1]
    df = pd.read_parquet(out / f"{day}.parquet")
    r12 = row_at(df, day, "12:00")
    assert r12["rth_mr_vr_blocks_q10"] == r12["n_bars_rth"] - 10 + 1


# ── 4. ADF sign convention (the plan's test 4, corrected) ────────────────────

def test_adf_sign_convention(mr):
    """A strongly mean-reverting spread must read negative in the raw stat AND
    negative in the trendiness z. Reversion must never read as trend."""
    n = 400
    rng = np.random.default_rng(3)
    # OU-ish: strongly pulled back to zero
    reverting = np.zeros(n)
    for i in range(1, n):
        reverting[i] = 0.3 * reverting[i - 1] + rng.normal(0, 1.0)
    # random walk spread: no pull
    walk = np.cumsum(rng.normal(0, 1.0, n))

    ok = np.ones(n, dtype=bool)
    t_rev, _ = mr._adf_reference(reverting, ok, 1)
    t_walk, _ = mr._adf_reference(walk, ok, 1)

    assert t_rev < -5.0, t_rev                    # strong reversion
    assert t_walk > t_rev                         # walk is "more trending"
    # the z keeps the same direction: trendiness is NOT sign-flipped
    center, scale = t_walk, 1.0
    assert mr._z(t_rev, center, scale) < 0
    assert mr._to_z_space("adf_open", t_rev) == pytest.approx(t_rev)


def test_z_space_transforms(mr):
    """VR is z-scored in ln space, ER in logit space, ADF and Hurst raw."""
    assert mr._to_z_space("vr_q10", np.e) == pytest.approx(1.0)
    assert np.isnan(mr._to_z_space("vr_q10", 0.0))       # ln 0 has no answer
    assert mr._to_z_space("er_open", 0.5) == pytest.approx(0.0)
    assert mr._to_z_space("er_open", 0.01) < -4          # the pile-at-zero end
    assert np.isfinite(mr._to_z_space("er_open", 0.0))   # clipped, not -inf
    assert np.isfinite(mr._to_z_space("er_open", 1.0))
    for key, value in (("adf_open", -3.4), ("hurst", 0.42),
                       ("hurst_rs", 0.61)):
        assert mr._to_z_space(key, value) == pytest.approx(value)


def test_adf_matches_statsmodels(mr):
    statsmodels = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(11)
    for series in (np.cumsum(rng.normal(0, 1, 250)),
                   rng.normal(0, 1, 250),
                   np.sin(np.linspace(0, 25, 250)) * 3 + rng.normal(0, .2, 250)):
        mine, nobs = mr._adf_reference(series, np.ones(len(series), bool), 1)
        theirs = statsmodels.adfuller(series, maxlag=1, regression="c",
                                      autolag=None)
        assert mine == pytest.approx(theirs[0], rel=1e-9)
        assert nobs == theirs[3]


# ── 5. fresh VWAP, strictly causal ───────────────────────────────────────────

def test_fresh_vwap_is_causal_and_correct(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_day(src, d)
    run(mr, src, out)
    day = days[-1]
    df = pd.read_parquet(out / f"{day}.parquet")
    bars = pd.read_parquet(src / f"{day}.parquet")

    typ = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    vol = bars["volume"]
    lo = pd.Timestamp(f"{day} 09:30", tz=NY)       # anchor_open_time default

    for clock in ("10:00", "11:30", "14:00"):
        ts = pd.Timestamp(f"{day} {clock}", tz=NY)
        # bars STRICTLY before the snapshot, from the anchor reset
        window = (bars.index >= lo) & (bars.index < ts)
        expected = float((typ[window] * vol[window]).sum() / vol[window].sum())
        assert df.loc[ts, "mr_anchor_price_vwap_rth"] == pytest.approx(
            expected, rel=1e-12)
        # including the 10:00 bar would change the answer -> the rule bites
        with_extra = (bars.index >= lo) & (bars.index <= ts)
        leaky = float((typ[with_extra] * vol[with_extra]).sum()
                      / vol[with_extra].sum())
        assert leaky != pytest.approx(expected, rel=1e-15)

    # before the anchor reset the RTH VWAP does not exist
    assert np.isnan(row_at(df, day, "09:00")["mr_anchor_price_vwap_rth"])
    # the globex VWAP does (it resets at the session's first bar)
    assert np.isfinite(row_at(df, day, "09:00")["mr_anchor_price_vwap_gbx"])


def test_vwap_rth_freezes_at_anchor_close(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_day(src, d)
    run(mr, src, out, anchor_close_time="15:00")
    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    after = df[df.index >= pd.Timestamp(f"{days[-1]} 15:30", tz=NY)]
    assert len(after) > 2
    assert after["mr_anchor_price_vwap_rth"].nunique() == 1


# ── 6. anchor fan-out ────────────────────────────────────────────────────────

def test_anchor_fanout_exactly_matches_param(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_day(src, d)
    run(mr, src, out, anchors="open,ema")

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    for family in ("adf", "er"):
        for scope in ("gbx", "rth"):
            assert f"{scope}_mr_{family}_open" in df.columns
            assert f"{scope}_mr_{family}_ema" in df.columns
            for gone in ("vwap_gbx", "vwap_rth", "rmean"):
                assert f"{scope}_mr_{family}_{gone}" not in df.columns
    assert "mr_state_adf_open" in df.columns
    assert "mr_state_er_ema" in df.columns
    assert "mr_state_adf_rmean" not in df.columns
    # the declared states in meta.json follow the run, not the module default
    meta = rio.read_meta(out)
    assert set(meta["states"]) == {"mr_state", "mr_state_hurst",
                                   "mr_state_adf_open", "mr_state_er_open",
                                   "mr_state_adf_ema", "mr_state_er_ema"}

    with pytest.raises(ValueError, match="unknown anchor"):
        run(mr, src, out, anchors="open,vwop")


# ── 7. q_label fallback ──────────────────────────────────────────────────────

def test_q_label_fallback_is_recorded(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_day(src, d)
    run(mr, src, out, q_values="2,5", q_label=10)

    meta = rio.read_meta(out)
    extras = meta["script_extras"]
    assert extras["q_label_used"] == 5
    assert extras["q_label_fallback"]["requested"] == 10
    assert extras["q_label_fallback"]["used"] == 5

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    assert "rth_mr_vr_q5" in df.columns and "rth_mr_vr_q10" not in df.columns
    # the label really came off q=5: same z, same threshold, same states
    z = np.where(df["n_bars_rth"] > 0, df["rth_mr_vr_z_q5"],
                 df["gbx_mr_vr_z_q5"])
    seed = df["mr_state"].iloc[0]
    expected = mr._schmitt(z, seed if seed in mr.STATES else "neutral",
                           mr.PARAMS["enter_trend"], mr.PARAMS["enter_revert"],
                           mr.PARAMS["exit_trend"], mr.PARAMS["exit_revert"])
    assert list(df["mr_state"]) == expected


# ── 8. matched window ────────────────────────────────────────────────────────

def test_1000_yardstick_uses_prior_mornings_only(tmp_path, mr):
    """The 10:00 centre must be built from prior days' 09:30-09:59 windows."""
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 9)
    for i, d in enumerate(days):
        # strongly trending mornings, ordinary walk for the rest of the day
        make_day(src, d, kind="walk", kind_morning="runs", seed=i)
    run(mr, src, out)

    day = days[-1]
    df = pd.read_parquet(out / f"{day}.parquet")
    ten = row_at(df, day, "10:00")

    # rebuild the yardstick by hand from the prior days' own bars
    expected = []
    for d in days[:-1]:
        bars = pd.read_parquet(src / f"{d}.parquet")
        lo = bars.index.searchsorted(pd.Timestamp(f"{d} 09:30", tz=NY))
        hi = bars.index.searchsorted(pd.Timestamp(f"{d} 10:00", tz=NY))
        stats = mr._return_stats(bars, False)
        vr = mr._vr_reference(stats["r"][lo:hi], stats["valid"][lo:hi], 2)[0]
        expected.append(np.log(vr))
    assert ten["rth_hist_vr_q2_n"] == len(expected)
    assert ten["rth_hist_vr_q2_center"] == pytest.approx(
        float(np.median(expected)), rel=1e-9)

    # and it is nowhere near a full-day yardstick: the mornings trend hard
    # (VR ~ 2) while the rest of the day is a walk (VR ~ 1), so the 10:00
    # centre must sit well above the 16:00 one. A single unmatched yardstick
    # would make both the same number.
    close = row_at(df, day, "16:00")
    assert ten["rth_hist_vr_q2_center"] - close["rth_hist_vr_q2_center"] > 0.3


# ── 9. hysteresis + cross-day chain ──────────────────────────────────────────

def test_schmitt_trigger_sequence(mr):
    z = [0.5, 0.95, 0.5, 0.2, 1.1, -1.0, -0.5, -0.1, 0.0, np.nan, 0.4]
    got = mr._schmitt(z, "neutral", 0.90, -0.90, 0.30, -0.30)
    assert got == ["neutral",        # 0.5 below enter_trend
                   "trending",       # 0.95 enters
                   "trending",       # 0.5 above exit_trend — dead zone holds
                   "neutral",        # 0.2 exits
                   "trending",       # 1.1 re-enters
                   "reverting",      # -1.0 exits trending AND enters reverting
                   "reverting",      # -0.5 below exit_revert — holds
                   "neutral",        # -0.1 exits
                   "neutral",
                   "unknown",        # NaN — no answer, chain untouched
                   "neutral"]        # continues from neutral, not unknown


def test_schmitt_seed_respected(mr):
    args = (0.90, -0.90, 0.30, -0.30)
    assert mr._schmitt([0.5], "trending", *args) == ["trending"]
    assert mr._schmitt([0.5], "neutral", *args) == ["neutral"]
    assert mr._schmitt([-0.5], "reverting", *args) == ["reverting"]
    assert mr._schmitt([0.5], "unknown", *args) == ["neutral"]


def test_state_chain_reads_previous_file_and_resets_on_gap(tmp_path):
    mr = _load()                      # private instance — we wrap _schmitt
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 8)
    for i, d in enumerate(days):
        make_day(src, d, kind=("runs" if i % 3 == 0 else "walk"), seed=i)

    seeds = []
    orig = mr._schmitt
    mr._schmitt = lambda z, seed, *a: (seeds.append(seed), orig(z, seed, *a))[1]
    run(mr, src, out)
    mr._schmitt = orig

    # one seed per state column per day; the FIRST of each day's batch is the
    # primary chain (secondary columns always reset to neutral)
    per_day = len(seeds) // len(days)
    primary_seeds = [seeds[i * per_day] for i in range(len(days))]
    finals = {d: pd.read_parquet(out / f"{d}.parquet")["mr_state"].iloc[-1]
              for d in days}
    for i in range(1, len(days)):
        prev = finals[days[i - 1]]
        assert primary_seeds[i] == (prev if prev in mr.STATES else "neutral")
    assert all(s == "neutral" for s in seeds[1:per_day])   # secondaries reset

    expected_resets = [days[0]] + [days[i] for i in range(1, len(days))
                                   if finals[days[i - 1]] not in mr.STATES]
    meta = rio.read_meta(out)
    assert meta["script_extras"]["chain_resets"] == expected_resets
    assert len(expected_resets) < len(days)          # the chain did run intact

    # break the chain: day 1 disappears, day 2 is recomputed with no yesterday
    (src / f"{days[0]}.parquet").unlink()
    (out / f"{days[0]}.parquet").unlink()
    (out / f"{days[1]}.parquet").unlink()
    seeds.clear()
    mr._schmitt = lambda z, seed, *a: (seeds.append(seed), orig(z, seed, *a))[1]
    run(mr, src, out)                 # skip_existing resumes the rest
    mr._schmitt = orig
    assert seeds[0] == "neutral"      # no yesterday -> reset
    d2 = pd.read_parquet(out / f"{days[1]}.parquet")
    assert d2["diag_state_chain_reset"].all()
    assert days[1] in rio.read_meta(out)["script_extras"]["chain_resets"]


# ── 10. warm-up + thin measure ───────────────────────────────────────────────

def test_warmup_is_unknown_never_nan(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    for d in dates_from("2026-01-05", 3):
        make_day(src, d)
    mr.run_all(str(src), str(out), True, None, dict(mr.PARAMS))  # defaults: 60

    for f in sorted(out.glob("*.parquet")):
        df = pd.read_parquet(f)
        for col in [c for c in df.columns if c.startswith("mr_state")]:
            assert set(df[col].unique()) == {"unknown"}, col
            assert not df[col].isin(["nan", "None", ""]).any()


def test_thin_measure_is_unknown_while_others_label(tmp_path, mr):
    """A measure whose window is too short goes `unknown` on its own — the
    other measures at the same snapshot still label. This is what separates
    'neutral because random-walk' from 'neutral because 6 bars'."""
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 8)
    for i, d in enumerate(days):
        make_day(src, d, seed=i)
    run(mr, src, out, q_values="2,10,20", q_label=10)

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    ten = row_at(df, days[-1], "10:00")
    assert ten["n_bars_rth"] == 30
    # q-aware floors: q=2 needs 20 bars, q=10 needs 30, q=20 needs 60
    assert np.isfinite(ten["rth_mr_vr_q2"])
    assert np.isfinite(ten["rth_mr_vr_q10"])
    assert np.isnan(ten["rth_mr_vr_q20"])
    # ER (min_bars_for_measure=20) still answers at the same snapshot
    assert np.isfinite(ten["rth_mr_er_open"])
    assert ten["mr_state"] != "unknown"


# ── the hand-rolled batched math vs the readable reference ───────────────────

def test_frozen_tape_answers_unknown(tmp_path, mr):
    """A limit-locked / frozen session must produce `unknown`, not a confident
    label. Regression test for a real case: on 2020-03-16 ES was locked
    limit-down overnight, 98.5% of bars had zero price change, and the ADF
    against the rolling-mean anchor read t = -84 (vs ~-6 on normal days) off
    14 moving bars in a 930-bar window. Bars are not evidence; moves are.
    """
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 8)
    for i, d in enumerate(days[:-1]):
        make_day(src, d, seed=i)

    # the last day: price locked for all but a handful of minutes
    frozen = days[-1]
    make_day(src, frozen, seed=99)
    bars = pd.read_parquet(src / f"{frozen}.parquet")
    closes = bars["close"].to_numpy().copy()
    closes[:] = closes[0]
    closes[::120] = closes[0] * 1.0005        # a few real moves, ~12 in total
    closes = np.maximum.accumulate(np.where(np.arange(len(closes)) % 120 == 0,
                                            closes, closes[0]))
    bars = bars.assign(close=closes, open=closes, high=closes, low=closes)
    bars.to_parquet(src / f"{frozen}.parquet")

    run(mr, src, out)
    df = pd.read_parquet(out / f"{frozen}.parquet")
    final = df.iloc[-1]
    assert final["gbx_mr_n_moves"] < mr.PARAMS["min_bars_for_measure"]
    for col in [c for c in df.columns if c.startswith("mr_state")]:
        assert final[col] == "unknown", col
    for col in ("gbx_mr_vr_q10", "gbx_mr_adf_rmean", "gbx_mr_er_rmean",
                "gbx_mr_hurst", "gbx_mr_hurst_rs"):
        assert np.isnan(final[col]), col

    # a normal day right before it still labels — the guard is not a blanket
    normal = pd.read_parquet(out / f"{days[-2]}.parquet").iloc[-1]
    assert normal["gbx_mr_n_moves"] >= mr.PARAMS["min_bars_for_measure"]
    assert np.isfinite(normal["gbx_mr_vr_q10"])


def test_prefix_path_matches_reference_implementations(tmp_path, mr):
    """The run path computes VR / ADF / ER from prefix sums, vectorized over
    every snapshot at once; the reference functions compute them from an
    explicit window. Batched linear algebra is exactly where silent errors
    live, so the two must agree to floating-point noise."""
    src = tmp_path / "in"
    day = "2026-01-06"
    make_day(src, day, kind="walk", seed=5, drop=["13:00", "13:01", "13:02"])
    bars = pd.read_parquet(src / f"{day}.parquet")

    idx = bars.index
    cfg = {"q_values": [2, 5, 10], "anchors": list(mr.ANCHOR_NAMES),
           "ema_n": 20, "rmean_n": 20, "adf_maxlag": 1, "min_bars": 20,
           "drop_zero": False,
           "pos_anchor_open": int(idx.searchsorted(
               pd.Timestamp(f"{day} 09:30", tz=NY))),
           "pos_anchor_close": int(idx.searchsorted(
               pd.Timestamp(f"{day} 16:00", tz=NY)))}
    dm = mr.DayMeasures(bars, cfg)

    a = cfg["pos_anchor_open"]
    ends = np.array([a + 40, a + 150, len(bars)])
    block = dm.all_measures(a, ends)
    stats = dm.stats

    for j, b in enumerate(ends):
        for q in cfg["q_values"]:
            vr, _lm, blocks = mr._vr_reference(stats["r"][a:b],
                                               stats["valid"][a:b], q)
            got = block["measures"][f"vr_q{q}"][j]
            assert got == pytest.approx(vr, rel=1e-9), (q, b)
            assert block["extra"][f"vr_blocks_q{q}"][j] == blocks
        for name in cfg["anchors"]:
            anchor = dm.anchor[name]
            start = max(a, anchor["start"])
            t, nobs = mr._adf_reference(anchor["s"][start:b],
                                        anchor["step_ok"][start:b], 1)
            assert block["measures"][f"adf_{name}"][j] == pytest.approx(
                t, rel=1e-7), (name, b)
            assert block["extra"][f"adf_nobs_{name}"][j] == nobs
            er = mr._er_reference(anchor["s"][start:b],
                                  anchor["step_ok"][start:b])
            assert block["measures"][f"er_{name}"][j] == pytest.approx(
                er, rel=1e-9), (name, b)


def test_hurst_is_half_on_a_random_walk(mr):
    """Variance-of-differences Hurst: Var(k-bar returns) grows as k under a
    random walk, so the ln-ln slope is 1 and H is 0.5. Regressing the per-bar
    variance instead (which is flat in k) yields H ~ 0 — that mistake is
    invisible in the output unless a test pins the level."""
    n = 4000
    bars_idx = pd.date_range(pd.Timestamp("2026-01-05 18:00", tz=NY),
                             periods=n, freq="1min")
    r = returns_for("walk", n, 1e-3, seed=17)
    closes = 5000 * np.exp(np.cumsum(np.r_[0.0, r[1:]]))
    bars = pd.DataFrame({"open": closes, "high": closes * 1.0002,
                         "low": closes * 0.9998, "close": closes,
                         "volume": np.full(n, 100, np.int64)}, index=bars_idx)
    cfg = {"q_values": [2], "anchors": ["open"], "ema_n": 20, "rmean_n": 20,
           "adf_maxlag": 1, "min_bars": 20, "drop_zero": False,
           "pos_anchor_open": 0, "pos_anchor_close": n}
    dm = mr.DayMeasures(bars, cfg)
    h = dm.hurst_vod(0, np.array([n]))[0]
    assert h == pytest.approx(0.5, abs=0.06), h

    # alternating returns are the opposite extreme, and R/S must not read the
    # walk as strongly trending either
    alt = np.r_[0.0, returns_for("alternating", n - 1, 1e-3)]
    closes_alt = 5000 * np.exp(np.cumsum(alt))
    bars_alt = bars.assign(close=closes_alt, open=closes_alt,
                           high=closes_alt * 1.0002, low=closes_alt * 0.9998)
    dm_alt = mr.DayMeasures(bars_alt, cfg)
    assert dm_alt.hurst_vod(0, np.array([n]))[0] < 0.2


# ── module contract / param guards ───────────────────────────────────────────

def test_declared_columns_match_the_frame(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    days = dates_from("2026-01-05", 6)
    for d in days:
        make_day(src, d)
    run(mr, src, out)

    df = pd.read_parquet(out / f"{days[-1]}.parquet")
    meta = rio.read_meta(out)
    declared = {c for cols in meta["schema"]["tiers"].values() for c in cols}
    missing = declared - set(df.columns)
    assert not missing, sorted(missing)
    extra = set(df.columns) - declared - set(schema.ALWAYS_COLUMNS)
    assert not extra, sorted(extra)
    assert meta["script_extras"]["primary_regime_column"] == "mr_state"
    assert meta["warnings"] == []


def test_param_guards(tmp_path, mr):
    src, out = tmp_path / "in", tmp_path / "out"
    make_day(src, "2026-01-06")
    with pytest.raises(ValueError, match="adf_maxlag must be 1"):
        run(mr, src, out, adf_maxlag=2)
    with pytest.raises(ValueError, match="lookback_days"):
        mr.run_all(str(src), str(out), True, None,
                   {**mr.PARAMS, "lookback_days": 10, "min_lookback_days": 60})
