"""Tests for the options dealer-exposure transforms (options_gex_1m +
options_greeks_eod). All synthetic — no DBN files, no Qt, no real data.
Settlement prices are generated WITH the module's own Black-76 pricer at known
sigma, so IV recovery and put-call parity are exact by construction."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_transforms import options_gex_1m as gx
from data_transforms import options_greeks_eod as eod

R_PCT = 4.5
R = R_PCT / 100.0
NY = "America/New_York"


def ns(ts: str, tz: str = "UTC") -> int:
    return int(pd.Timestamp(ts, tz=tz).value)


# ===========================================================================
# Math: Black-76 + IV solver
# ===========================================================================

def _grid():
    F0, ratio, T, sig, right = np.meshgrid(
        np.array([50.0, 5000.0]),
        np.array([0.7, 0.9, 1.0, 1.1, 1.3]),
        np.array([1 / 365, 0.05, 0.5, 2.0]),
        np.array([0.08, 0.2, 0.8]),
        np.array([True, False]),
        indexing="ij",
    )
    F0, ratio, T, sig = (a.ravel().astype(float) for a in (F0, ratio, T, sig))
    return F0, F0 * ratio, sig, T, right.ravel().astype(bool)


def test_iv_round_trip():
    F, K, sig, T, is_call = _grid()
    price = gx.black76_price(F, K, sig, T, R, is_call)
    iv = gx.implied_vol(price, F, K, T, R, is_call)
    # recoverability is about TIME VALUE, not price: deep-ITM rows have big
    # prices but ~zero premium over intrinsic and legitimately return NaN
    intrinsic = np.exp(-R * T) * np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    tv = price - intrinsic
    solvable = tv > 1e-6
    assert solvable.sum() > 100
    np.testing.assert_allclose(iv[solvable], sig[solvable], atol=1e-6)
    assert np.isnan(iv[tv <= 1e-12]).all()


def test_price_put_call_parity():
    F, K, sig, T, _ = _grid()
    c = gx.black76_price(F, K, sig, T, R, np.full(F.shape, True))
    p = gx.black76_price(F, K, sig, T, R, np.full(F.shape, False))
    np.testing.assert_allclose(c - p, np.exp(-R * T) * (F - K), rtol=1e-12, atol=1e-9)


def test_gamma_vanna_call_put_equal():
    F, K, sig, T, _ = _grid()
    gc = gx.black76_greeks(F, K, sig, T, R, np.full(F.shape, True))
    gp = gx.black76_greeks(F, K, sig, T, R, np.full(F.shape, False))
    np.testing.assert_allclose(gc["gamma"], gp["gamma"], rtol=1e-12)
    np.testing.assert_allclose(gc["vanna"], gp["vanna"], rtol=1e-12)


def test_vanna_signs():
    atm = gx.black76_greeks(100.0, 100.0, 0.2, 0.5, R, True)  # d2 < 0 at the money
    assert atm["vanna"] > 0
    itm = gx.black76_greeks(100.0, 60.0, 0.2, 0.5, R, True)   # deep ITM call: d2 >> 0
    assert itm["vanna"] < 0


def test_charm_parity_identity():
    F, K, sig, T, _ = _grid()
    gc = gx.black76_greeks(F, K, sig, T, R, np.full(F.shape, True))
    gp = gx.black76_greeks(F, K, sig, T, R, np.full(F.shape, False))
    np.testing.assert_allclose(gc["charm"] - gp["charm"], R * np.exp(-R * T),
                               rtol=1e-10, atol=1e-12)


def test_charm_matches_finite_difference():
    F, K, sig, T = 100.0, 105.0, 0.25, 0.3
    h = 1e-6
    for is_call in (True, False):
        d_lo = gx.black76_greeks(F, K, sig, T + h, R, is_call)["delta"]
        d_hi = gx.black76_greeks(F, K, sig, T - h, R, is_call)["delta"]
        fd = (d_hi - d_lo) / (2 * h)  # dDelta/dt = -dDelta/dT
        charm = gx.black76_greeks(F, K, sig, T, R, is_call)["charm"]
        np.testing.assert_allclose(charm, fd, rtol=1e-4)


def test_iv_nan_cases_and_batch_isolation():
    F = np.array([100.0, 100.0, 100.0, 100.0])
    K = np.array([100.0, 80.0, 100.0, 100.0])
    T = np.array([0.5, 0.5, 0.0, 0.5])
    is_call = np.full(4, True)
    good = float(gx.black76_price(100.0, 100.0, 0.3, 0.5, R, True))
    prices = np.array([
        good,
        np.exp(-R * 0.5) * 20.0 * 0.9,   # below discounted intrinsic
        good,                            # T = 0
        np.exp(-R * 0.5) * 100.0 * 1.01, # above the sigma=5 upper bound
    ])
    iv = gx.implied_vol(prices, F, K, T, R, is_call)
    assert abs(iv[0] - 0.3) < 1e-6       # the good row still solves
    assert np.isnan(iv[1:]).all()


def test_zero_dte_floor():
    now = ns("2026-07-01 16:00", NY)
    T = gx.time_to_expiry_years(np.array([now - 10**9]), now, floor_minutes=2.0)
    np.testing.assert_allclose(T, 2.0 * 60e9 / gx.NS_PER_YEAR)
    g = gx.black76_greeks(100.0, 100.0, 0.3, float(T[0]), R, True)
    assert all(np.isfinite(v) for v in g.values())


def test_price_monotone_in_sigma():
    sigs = np.linspace(0.05, 2.0, 40)
    prices = gx.black76_price(100.0, 110.0, sigs, 0.3, R, np.full(40, True))
    assert (np.diff(prices) > 0).all()


# ===========================================================================
# Sign models + expiry helpers
# ===========================================================================

def test_sign_registry():
    assert set(gx.SIGN_MODELS) == {"type", "inverse", "absolute", "moneyness"}
    is_call = np.array([True, True, False, False])
    K = np.array([90.0, 110.0, 90.0, 110.0])  # ITM-call, OTM-call, OTM-put, ITM-put
    F = np.full(4, 100.0)
    np.testing.assert_array_equal(gx.SIGN_MODELS["type"](is_call, K, F), [1, 1, -1, -1])
    np.testing.assert_array_equal(gx.SIGN_MODELS["inverse"](is_call, K, F), [-1, -1, 1, 1])
    np.testing.assert_array_equal(gx.SIGN_MODELS["absolute"](is_call, K, F), [1, 1, 1, 1])
    np.testing.assert_array_equal(gx.SIGN_MODELS["moneyness"](is_call, K, F), [0, 0, -1, 0])


def test_sign_unknown_lists_models():
    with pytest.raises(ValueError) as e:
        gx.get_sign_model("bogus")
    for name in gx.SIGN_MODELS:
        assert name in str(e.value)


def test_underlying_root():
    assert gx.underlying_root("ESU6") == "ES"
    assert gx.underlying_root("ESZ26") == "ES"
    assert gx.underlying_root("CLM7") == "CL"
    assert gx.underlying_root("6EU6") == "6E"
    assert gx.underlying_root("ES") == "ES"  # no month code -> unchanged


def test_classify_settle_type_dst_proof():
    exp = pd.Series(pd.to_datetime([
        "2026-07-17 20:00:00+00:00",  # July 16:00 NY (EDT)
        "2026-01-15 21:00:00+00:00",  # January 16:00 NY (EST)
        "2026-06-19 13:30:00+00:00",  # July 09:30 NY
        "2026-12-18 14:30:00+00:00",  # December 09:30 NY
    ]))
    np.testing.assert_array_equal(gx.classify_settle_type(exp), ["PM", "PM", "AM", "AM"])


def test_classify_exercise_style():
    styles = gx.classify_exercise_style(np.array(["ES", "EW3", "NQ"]),
                                        np.array(["ES", "ES", "NQ"]))
    np.testing.assert_array_equal(styles, ["american", "european", "american"])


def test_settlement_reference_ns():
    times = np.array([ns("2026-06-30 20:00"), ns("2026-06-30 20:05"), ns("2026-06-30 20:10")])
    assert gx.settlement_reference_ns(times, "2026-06-30") == ns("2026-06-30 20:05")
    assert gx.settlement_reference_ns(np.array([]), "2026-06-30") == ns("2026-06-30 16:00", NY)


# ===========================================================================
# IO / path resolution
# ===========================================================================

def _make_hierarchy(root: Path, dates=("20260630", "20260701")) -> dict:
    asset = root / "raw_dbn" / "Options_on_futures" / "XX"
    dirs = {
        "defs": asset / "XX_A_DEFINITION",
        "stats": asset / "XX_A_STATISTICS",
        "tbbo": asset / "XX_A_TBBO",
        "fut": root / "raw_dbn" / "Futures" / "XX" / "XX_A_STATISTICS",
        "candles": root / "parquet" / "Futures" / "XX" / "XX_1m_ohlcv_globex",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    for d in dates:
        (dirs["defs"] / f"glbx-mdp3-{d}.definition.dbn.zst").touch()
        (dirs["stats"] / f"glbx-mdp3-{d}.statistics.dbn.zst").touch()
        (dirs["tbbo"] / f"glbx-mdp3-{d}.tbbo.dbn.zst").touch()
        (dirs["fut"] / f"glbx-mdp3-{d}.statistics.dbn.zst").touch()
        (dirs["candles"] / f"{d[:4]}-{d[4:6]}-{d[6:]}.parquet").touch()
    return dirs


def test_resolution_from_any_of_the_three_folders(tmp_path):
    root = tmp_path / "renamed_market_data"  # root name must not matter
    dirs = _make_hierarchy(root)
    for entry in ("defs", "stats", "tbbo"):
        ds = gx.resolve_option_datasets(str(dirs[entry]))
        assert ds["defs_dir"] == dirs["defs"]
        assert ds["stats_dir"] == dirs["stats"]
        assert ds["root"] == root
        assert ds["asset"] == "XX"


def test_resolution_errors(tmp_path):
    dirs = _make_hierarchy(tmp_path / "r")
    # ambiguity: a second *statistics* sibling (both with daily files)
    other = dirs["stats"].parent / "XX_B_STATISTICS"
    other.mkdir()
    (other / "glbx-mdp3-20260701.statistics.dbn.zst").touch()
    with pytest.raises(ValueError, match="statistic"):
        gx.resolve_option_datasets(str(dirs["tbbo"]))
    # wrong depth (not under raw_dbn)
    with pytest.raises(ValueError, match="raw_dbn"):
        gx.resolve_option_datasets(str(dirs["candles"]))


def test_resolution_prefers_daily_over_monthly_leftover(tmp_path):
    dirs = _make_hierarchy(tmp_path / "r")
    # a monthly-batched leftover next to the daily re-download is skipped
    monthly = dirs["fut"].parent / "XX_STATISTICS_monthly"
    monthly.mkdir()
    (monthly / "glbx-mdp3-20250501-20250531.statistics.dbn.zst").touch()
    fut_stats, _ = gx.resolve_futures_dirs(tmp_path / "r", "Futures", "XX", "", "")
    assert fut_stats == dirs["fut"]


def test_index_dbn_files(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "glbx-mdp3-20260701.statistics.dbn.zst").touch()
    (d / "glbx-mdp3-20250519-20250531.statistics.dbn.zst").touch()  # monthly -> ignored
    (d / "metadata.json").touch()
    idx = gx.index_dbn_files(d)
    assert list(idx) == ["2026-07-01"]

    monthly_only = tmp_path / "monthly"
    monthly_only.mkdir()
    (monthly_only / "glbx-mdp3-20250519-20250531.statistics.dbn.zst").touch()
    with pytest.raises(FileNotFoundError, match="monthly"):
        gx.index_dbn_files(monthly_only)


def test_loads_through_the_plugin_loader():
    """The Data Formatter loads transforms via spec_from_file_location WITHOUT
    sys.modules registration — regression test for the Python 3.13 dataclass
    crash with `from __future__ import annotations` (string field annotations
    make dataclasses._is_type dereference sys.modules.get(module) == None)."""
    from modules.common.backend.plugins import list_plugins, load_module
    refs = {r.label: r for r in list_plugins([Path(gx.__file__).parent])}
    for name in ("options_gex_1m", "options_greeks_eod"):
        assert name in refs, f"{name} not discovered by the plugin scan"
        mod = load_module(refs[name])
        assert callable(mod.run_all) and mod.PARAMS and mod.PARAM_SECTIONS


def test_import_chain_is_qt_free():
    repo = Path(gx.__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(repo)!r}); "
        "import data_transforms.options_gex_1m, data_transforms.options_greeks_eod; "
        "assert 'PySide6' not in sys.modules, 'transforms must stay Qt-free'"
    )
    # cwd=tests, never the repo root (a local inspect.py there shadows stdlib)
    res = subprocess.run([sys.executable, "-c", code], cwd=str(repo / "tests"),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ===========================================================================
# Exposures / aggregation / zero gamma
# ===========================================================================

def test_compute_exposures_scaling():
    g = {"delta": np.array([0.5]), "gamma": np.array([0.02]),
         "vanna": np.array([0.1]), "charm": np.array([-0.3])}
    base = gx.compute_exposures(g, np.array([100.0]), np.array([50.0]),
                                np.array([100.0]), np.array([1.0]))
    double_oi = gx.compute_exposures(g, np.array([200.0]), np.array([50.0]),
                                     np.array([100.0]), np.array([1.0]))
    double_f = gx.compute_exposures(g, np.array([100.0]), np.array([50.0]),
                                    np.array([200.0]), np.array([1.0]))
    for k in base:
        np.testing.assert_allclose(double_oi[k], 2 * base[k])
    np.testing.assert_allclose(double_f["gex"], 4 * base["gex"])   # quadratic in F
    np.testing.assert_allclose(double_f["dex"], 2 * base["dex"])   # linear in F


def test_aggregate_by_strike_matches_groupby():
    rng = np.random.default_rng(7)
    n, m, n_strikes = 50, 7, 5
    vals = rng.normal(size=(n, m))
    codes = rng.integers(0, n_strikes, size=n)
    agg = gx.aggregate_by_strike(vals, codes, n_strikes)
    for j in range(m):
        ref = pd.Series(vals[:, j]).groupby(codes).sum()
        got = pd.Series(agg[j]).loc[ref.index]
        np.testing.assert_allclose(got.to_numpy(), ref.to_numpy())


def _tiny_book():
    K = np.array([105.0, 95.0])
    F = np.array([100.0, 100.0])
    sig = np.array([0.2, 0.25])
    T = np.array([0.1, 0.1])
    oi = np.array([300.0, 260.0])
    mult = np.array([50.0, 50.0])
    sign = np.array([1.0, -1.0])  # +call gamma at 105, -put gamma at 95
    return F, K, sig, T, oi, mult, sign


def _dense_gamma_total(u, F, K, sig, T, oi, mult, sign):
    tot = 0.0
    for i in range(len(K)):
        g = gx.black76_greeks(F[i] * u, K[i], sig[i], T[i], R, True)["gamma"]
        tot += sign[i] * g * oi[i] * mult[i] * (F[i] * u) ** 2 * 0.01
    return tot


def test_zero_gamma_curve_matches_dense():
    F, K, sig, T, oi, mult, sign = _tiny_book()
    u_grid = np.linspace(0.92, 1.08, 41)
    st = sig * np.sqrt(T)
    d1_at_F = (np.log(F / K) + 0.5 * sig**2 * T) / st
    w0 = sign * oi * mult * 0.01 * np.exp(-R * T) * F * gx.INV_SQRT_2PI / st
    G = gx.zero_gamma_curve(u_grid, d1_at_F, 1.0 / st, w0)
    dense = np.array([_dense_gamma_total(u, F, K, sig, T, oi, mult, sign) for u in u_grid])
    np.testing.assert_allclose(G, dense, rtol=1e-10)


def test_zero_gamma_level_vs_brute_force():
    F, K, sig, T, oi, mult, sign = _tiny_book()
    fine = np.linspace(0.92, 1.08, 100_001)
    dense = np.array([_dense_gamma_total(u, F, K, sig, T, oi, mult, sign) for u in
                      np.linspace(0.92, 1.08, 2001)])
    fine2001 = np.linspace(0.92, 1.08, 2001)
    flips = np.nonzero(np.sign(dense[:-1]) * np.sign(dense[1:]) < 0)[0]
    assert flips.size, "test book must have a crossing"
    ref = 100.0 * fine2001[flips[0]]

    u_grid = np.linspace(0.92, 1.08, 41)
    st = sig * np.sqrt(T)
    d1_at_F = (np.log(F / K) + 0.5 * sig**2 * T) / st
    w0 = sign * oi * mult * 0.01 * np.exp(-R * T) * F * gx.INV_SQRT_2PI / st
    G = gx.zero_gamma_curve(u_grid, d1_at_F, 1.0 / st, w0)
    z = gx.zero_gamma_level(100.0, u_grid, G)
    assert np.isfinite(z)
    assert abs(z - ref) < 0.05  # coarse grid + interpolation vs brute force
    del fine  # (kept the name for clarity of intent)


def test_zero_gamma_level_edge_cases():
    u = np.linspace(0.9, 1.1, 21)
    assert np.isnan(gx.zero_gamma_level(100.0, u, np.ones(21)))          # no crossing
    z = gx.zero_gamma_level(100.0, u, u - 1.02)                          # single crossing
    np.testing.assert_allclose(z, 102.0, rtol=1e-9)
    z2 = gx.zero_gamma_level(100.0, u, (u - 0.95) * (u - 1.03))          # nearest to spot wins
    np.testing.assert_allclose(z2, 103.0, rtol=1e-9)


# ===========================================================================
# 1m pipeline (synthetic DayInputs)
# ===========================================================================

TRADE, PREV, PREVPREV = "2026-07-01", "2026-06-30", "2026-06-29"
T_REF_PREV = ns("2026-06-30 20:00")
EXP_MAIN = ns("2026-07-17 16:00", NY)   # PM weekly
EXP_FAR = ns("2026-09-18 09:30", NY)    # AM quarterly
EXP_0DTE = ns("2026-07-01 16:00", NY)

#         id  right   K     month   exp       sigma
CONTRACTS = [
    (1, "C", 95.0, "XXU6", EXP_MAIN, 0.20),
    (2, "P", 95.0, "XXU6", EXP_MAIN, 0.20),
    (3, "C", 105.0, "XXU6", EXP_MAIN, 0.22),
    (4, "P", 105.0, "XXU6", EXP_MAIN, 0.22),
    (5, "C", 105.0, "XXZ6", EXP_FAR, 0.30),
    (6, "P", 95.0, "XXZ6", EXP_FAR, 0.30),   # settlement flagged theoretical
    (7, "C", 100.0, "XXU6", EXP_0DTE, 0.35),
    (8, "P", 100.0, "XXU6", EXP_0DTE, 0.35),
]
F_BY_MONTH = {"XXU6": 100.0, "XXZ6": 102.0}
OI_A = {1: 100, 2: 200, 3: 300, 4: 80, 5: 50, 6: 20, 7: 0, 8: 0}
OI_B = {**OI_A, 1: 120, 7: 30, 8: 40}
OI_C = {**OI_B, 1: 110}
MULT = 50.0


def _make_defs():
    rows = []
    for cid, right, k, month, exp, _sig in CONTRACTS:
        rows.append({
            "instrument_id": cid, "raw_symbol": f"XX{cid} {right}{int(k)}",
            "parent": "XW", "underlying": month, "right": right, "strike": k,
            "expiration_ns": exp, "multiplier": MULT, "min_price_increment": 0.05,
            "underlying_root": "XX",
            "settle_type": "AM" if pd.Timestamp(exp, tz="UTC").tz_convert(NY).hour < 12 else "PM",
            "exercise_style": "european",
        })
    return pd.DataFrame(rows)


def _settle_price(cid):
    _, right, k, month, exp, sig = next(c for c in CONTRACTS if c[0] == cid)
    T = gx.time_to_expiry_years(np.array([exp]), T_REF_PREV, 1.0)[0]
    return float(gx.black76_price(F_BY_MONTH[month], k, sig, T, R, right == "C"))


def _stat_rows(rows):
    base = {"price": np.nan, "quantity": np.nan, "stat_flags": 3,
            "update_action": 1, "ts_recv": 0}
    out = []
    for r in rows:
        d = {**base, **r}
        d.setdefault("ts_recv", d["ts_event"])
        out.append(d)
    df = pd.DataFrame(out)
    df["ts_recv"] = np.where(df["ts_recv"] == 0, df["ts_event"], df["ts_recv"])
    return df


def _make_opt_stats():
    rows = []
    for cid, *_ in CONTRACTS:
        rows.append({"instrument_id": cid, "stat_type": gx.STAT_SETTLE,
                     "price": _settle_price(cid),
                     "stat_flags": 1 if cid == 6 else 3,  # 1 = final, NOT actual -> theoretical
                     "ts_event": T_REF_PREV, "ts_ref_date": PREV})
    # real OI rows carry stat_flags = 0 — prelim/final only by publication time
    waves = [(OI_A, PREVPREV, ns("2026-06-30 14:00")),
             (OI_B, PREV, ns("2026-07-01 01:00")),
             (OI_C, PREV, ns("2026-07-01 14:00"))]
    for oi, ref, t in waves:
        for cid, q in oi.items():
            rows.append({"instrument_id": cid, "stat_type": gx.STAT_OI, "quantity": float(q),
                         "stat_flags": 0, "ts_event": t, "ts_ref_date": ref})
    return _stat_rows(rows)


def _make_fut_settles():
    return pd.DataFrame([
        {"symbol": m, "price": f, "stat_flags": 3, "is_final": True, "is_theoretical": False,
         "ts_event": T_REF_PREV, "ts_recv": T_REF_PREV, "ts_ref_date": PREV}
        for m, f in F_BY_MONTH.items()
    ])


def _make_candles():
    blocks = (
        pd.date_range("2026-06-30 18:00", periods=30, freq="min", tz=NY)   # regime A
        .append(pd.date_range("2026-06-30 21:30", periods=30, freq="min", tz=NY))  # B
        .append(pd.date_range("2026-07-01 10:30", periods=30, freq="min", tz=NY))  # C
    )
    # [us] index ON PURPOSE — exercises the ns normalization guard
    return pd.DataFrame({"close": 100.0}, index=blocks.as_unit("us"))


def _day_inputs():
    return gx.DayInputs(trade_date=TRADE, prev_date=PREV, defs=_make_defs(),
                        opt_stats=_make_opt_stats(), fut_settles=_make_fut_settles(),
                        candles=_make_candles())


@pytest.fixture(scope="module")
def day_result():
    df, meta, warns = gx.process_day(_day_inputs(), dict(gx.PARAMS))
    return df, meta, warns


def test_1m_shape_and_index(day_result):
    df, meta, _ = day_result
    assert meta["n_strikes"] == 3 and meta["n_minutes"] == 90
    assert len(df) == 90 * 3
    assert str(df.index.tz) == NY and df.index.dtype == "datetime64[ns, America/New_York]"
    np.testing.assert_array_equal(df["strike"].iloc[:3].to_numpy(), [95.0, 100.0, 105.0])
    assert meta["front_month"] == "XXU6"
    assert meta["n_contracts"] == 8


def test_1m_oi_regime_switching(day_result):
    df, _, _ = day_result
    minute = df.index.unique()

    def oi_at(minute_i, strike, col):
        block = df.loc[df.index == minute[minute_i]]
        return int(block.loc[block["strike"] == strike, col].iloc[0])

    # call OI at 95 comes solely from contract 1: 100 -> 120 -> 110
    assert oi_at(0, 95.0, "call_oi") == 100
    assert oi_at(35, 95.0, "call_oi") == 120
    assert oi_at(75, 95.0, "call_oi") == 110
    # the 0DTE pair at 100 opens overnight (regime A -> B)
    assert oi_at(0, 100.0, "call_oi") == 0 and oi_at(0, 100.0, "put_oi") == 0
    assert oi_at(35, 100.0, "call_oi") == 30 and oi_at(35, 100.0, "put_oi") == 40
    # regime column
    reg = df["oi_regime"].to_numpy().reshape(90, 3)[:, 0]
    np.testing.assert_array_equal(np.unique(reg[:30]), [0])
    np.testing.assert_array_equal(np.unique(reg[30:60]), [1])
    np.testing.assert_array_equal(np.unique(reg[60:]), [2])


def _reference_minute(t_ns_utc: int, oi: dict, sign_model="type"):
    """Independent per-strike reference from the seeds (uses only the tested
    math primitives, not the pipeline)."""
    per_strike = {}
    walls_in = {}
    for cid, right, k, month, exp, sig in CONTRACTS:
        F = 100.0 + (F_BY_MONTH[month] - F_BY_MONTH["XXU6"])
        T = float(gx.time_to_expiry_years(np.array([exp]), t_ns_utc, 1.0)[0])
        g = gx.black76_greeks(F, k, sig, T, R, right == "C")
        sign = float(gx.SIGN_MODELS[sign_model](np.array([right == "C"]),
                                                np.array([k]), np.array([F]))[0])
        gex = sign * g["gamma"] * oi[cid] * MULT * F * F * 0.01
        mag = g["gamma"] * oi[cid] * MULT * F * F * 0.01
        per_strike[k] = per_strike.get(k, 0.0) + gex
        key = (k, right)
        walls_in[key] = walls_in.get(key, 0.0) + mag
    return per_strike, walls_in


def test_1m_exposures_match_reference(day_result):
    df, _, _ = day_result
    minute = df.index.unique()
    t0 = int(minute[0].tz_convert("UTC").value)
    expected, walls_in = _reference_minute(t0, OI_A)
    block = df.loc[df.index == minute[0]].set_index("strike")
    for k, v in expected.items():
        np.testing.assert_allclose(block.loc[k, "gex"], v, rtol=5e-5)
    np.testing.assert_allclose(block["total_gex"].iloc[0], sum(expected.values()), rtol=5e-5)
    # walls from the same reference
    call_wall = max((k for (k, r) in walls_in if r == "C"), key=lambda k: walls_in[(k, "C")])
    put_wall = max((k for (k, r) in walls_in if r == "P"), key=lambda k: walls_in[(k, "P")])
    assert block["call_wall"].iloc[0] == call_wall
    assert block["put_wall"].iloc[0] == put_wall


def test_1m_sign_models(day_result):
    df, _, _ = day_result
    p_inv = {**gx.PARAMS, "sign_model": "inverse"}
    df_inv, _, _ = gx.process_day(_day_inputs(), p_inv)
    np.testing.assert_allclose(df_inv["gex"].to_numpy(), -df["gex"].to_numpy(), rtol=1e-6)
    p_abs = {**gx.PARAMS, "sign_model": "absolute"}
    df_abs, _, _ = gx.process_day(_day_inputs(), p_abs)
    np.testing.assert_allclose(df_abs["gex"].to_numpy(),
                               (df["call_gex"] + df["put_gex"]).to_numpy(), rtol=1e-5)


def test_1m_zero_gamma_vs_brute_force(day_result):
    df, _, _ = day_result
    minute = df.index.unique()
    j = 70  # a regime-C minute
    t = int(minute[j].tz_convert("UTC").value)

    def total(u):
        tot = 0.0
        for cid, right, k, month, exp, sig in CONTRACTS:
            F = (100.0 + (F_BY_MONTH[month] - F_BY_MONTH["XXU6"])) * u
            T = float(gx.time_to_expiry_years(np.array([exp]), t, 1.0)[0])
            g = gx.black76_greeks(F, k, sig, T, R, right == "C")
            sign = 1.0 if right == "C" else -1.0
            tot += sign * g["gamma"] * OI_C[cid] * MULT * F * F * 0.01
        return tot

    grid = np.linspace(0.92, 1.08, 4001)
    dense = np.array([total(u) for u in grid])
    flips = np.nonzero(np.sign(dense[:-1]) * np.sign(dense[1:]) < 0)[0]
    got = df.loc[df.index == minute[j], "zero_gamma"].iloc[0]
    if flips.size:
        crossings = 100.0 * grid[flips]
        ref = crossings[np.argmin(np.abs(crossings - 100.0))]
        assert abs(got - ref) < 0.1
    else:
        assert np.isnan(got)


def test_1m_metadata_and_validations(day_result):
    _, meta, _ = day_result
    for key in ("trade_date", "iv_basis", "front_month", "n_contracts", "n_strikes",
                "n_iv_failed", "pct_oi_theoretical", "oi_regimes", "oi_pf_changed_pct",
                "parity_max_error", "gex_magnitude_bound", "params_json",
                "transform_version"):
        assert key in meta, key
    assert meta["iv_basis"] == "prev_settle" and meta["iv_basis_date"] == PREV
    assert meta["n_iv_failed"] == 0
    # V2: contract 6 (OI max 20) of total max-OI 840
    total = sum(max(OI_A[c], OI_B[c], OI_C[c]) for c in OI_A)
    np.testing.assert_allclose(meta["pct_oi_theoretical"], 20 / total * 100, rtol=1e-9)
    # V6: exactly one of 8 instruments changed between prelim and final
    np.testing.assert_allclose(meta["oi_pf_changed_pct"], 100 / 8, rtol=1e-9)
    assert meta["oi_pf_abs_diff_total"] == 10
    # V1: prices generated by Black-76 -> parity is exact
    assert meta["parity_expiries_checked"] >= 2
    assert meta["parity_max_error"] < 1e-6


def test_1m_filters():
    base = dict(gx.PARAMS)
    _, m1, _ = gx.process_day(_day_inputs(), {**base, "min_open_interest": 60})
    assert m1["n_contracts"] == 4  # ids 1-4 (max OI 120/200/300/80)
    _, m2, _ = gx.process_day(_day_inputs(), {**base, "strike_range_pct": 4.0})
    assert m2["n_strikes"] == 1 and m2["n_contracts"] == 2  # only the 100s
    _, m3, _ = gx.process_day(_day_inputs(), {**base, "max_dte_days": 5})
    assert m3["n_contracts"] == 2  # the 0DTE pair
    _, m4, _ = gx.process_day(_day_inputs(), {**base, "drop_theoretical": True})
    assert m4["n_contracts"] == 7  # contract 6 dropped


# ===========================================================================
# run_all shell behavior (skip / cancel / first-day) — loaders must not run
# ===========================================================================

class _Boom(Exception):
    pass


def _patch_loaders(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("loader must not be called")
    for name in ("load_definitions", "load_stats", "load_futures_settlements"):
        monkeypatch.setattr(gx, name, boom)


def test_run_all_skip_existing_zero_work(tmp_path, monkeypatch):
    dirs = _make_hierarchy(tmp_path / "r")
    _patch_loaders(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "2026-06-30.parquet").touch()
    (out / "2026-07-01.parquet").touch()
    msgs = []
    gx.run_all(str(dirs["tbbo"]), str(out), skip_existing=True,
               on_progress=lambda c, t, m: msgs.append(m),
               params={"futures_asset": "XX"})
    assert sum("[SKIP]" in m and "exists" in m for m in msgs) == 2
    assert not any("[ERROR]" in m for m in msgs)


def test_run_all_first_day_skip_and_error_isolation(tmp_path, monkeypatch):
    dirs = _make_hierarchy(tmp_path / "r")

    def boom(*a, **k):
        raise RuntimeError("decode blew up")
    monkeypatch.setattr(gx, "load_definitions", boom)
    msgs = []
    gx.run_all(str(dirs["stats"]), str(tmp_path / "out"), skip_existing=False,
               on_progress=lambda c, t, m: msgs.append(m),
               params={"futures_asset": "XX"})
    assert any("[SKIP] 2026-06-30" in m and "previous" in m for m in msgs)
    assert any("[ERROR] 2026-07-01" in m and "decode blew up" in m for m in msgs)


def test_run_all_cancellation_propagates(tmp_path, monkeypatch):
    dirs = _make_hierarchy(tmp_path / "r")
    _patch_loaders(monkeypatch)
    calls = []

    def cancelling_progress(c, t, m):
        calls.append(m)
        if len(calls) >= 2:  # cancel mid-loop, like FunctionWorker does
            raise _Boom()

    with pytest.raises(_Boom):
        gx.run_all(str(dirs["defs"]), str(tmp_path / "out"), skip_existing=False,
                   on_progress=cancelling_progress, params={"futures_asset": "XX"})


# ===========================================================================
# EOD pipeline
# ===========================================================================

T_REF_D = ns("2026-07-01 20:00")
F_EOD = {"XXU6": 100.5, "XXZ6": 102.5}


def _eod_settle_price(cid):
    _, right, k, month, exp, sig = next(c for c in CONTRACTS if c[0] == cid)
    T = float(gx.time_to_expiry_years(np.array([exp]), T_REF_D, 1.0)[0])
    return float(gx.black76_price(F_EOD[month], k, sig, T, R, right == "C"))


def _make_eod_stats(oi_final=True, oi_present=True, perturb_cid=None):
    rows = []
    for cid, *_ in CONTRACTS:
        px = _eod_settle_price(cid) + (5.0 if cid == perturb_cid else 0.0)
        rows.append({"instrument_id": cid, "stat_type": gx.STAT_SETTLE, "price": px,
                     "stat_flags": 1 if cid == 6 else 3,
                     "ts_event": T_REF_D, "ts_ref_date": TRADE})
        rows.append({"instrument_id": cid, "stat_type": gx.STAT_VOLUME,
                     "quantity": float(cid * 11),
                     "ts_event": ns("2026-07-02 01:00"), "ts_ref_date": TRADE})
        for st, val in ((gx.STAT_OPEN, 1.0), (gx.STAT_HIGH, 2.0), (gx.STAT_LOW, 0.5),
                        (gx.STAT_LOW_OFFER, 0.4), (gx.STAT_HIGH_BID, 2.5)):
            rows.append({"instrument_id": cid, "stat_type": st, "price": val + cid,
                         "ts_event": T_REF_D, "ts_ref_date": TRADE})
        # real OI rows carry stat_flags = 0 — finality inferred from bursts
        if oi_present:
            rows.append({"instrument_id": cid, "stat_type": gx.STAT_OI,
                         "quantity": float(OI_C[cid] + 5), "stat_flags": 0,
                         "ts_event": ns("2026-07-02 01:00"), "ts_ref_date": TRADE})
            if oi_final:
                rows.append({"instrument_id": cid, "stat_type": gx.STAT_OI,
                             "quantity": float(OI_C[cid]), "stat_flags": 0,
                             "ts_event": ns("2026-07-02 14:00"), "ts_ref_date": TRADE})
        else:  # only stale prior-date OI available
            rows.append({"instrument_id": cid, "stat_type": gx.STAT_OI,
                         "quantity": float(OI_C[cid]), "stat_flags": 0,
                         "ts_event": ns("2026-07-01 14:00"), "ts_ref_date": PREV})
    return _stat_rows(rows)


def _make_eod_fut():
    return pd.DataFrame([
        {"symbol": m, "price": f, "stat_flags": 3, "is_final": True, "is_theoretical": False,
         "ts_event": T_REF_D, "ts_recv": T_REF_D, "ts_ref_date": TRADE}
        for m, f in F_EOD.items()
    ])


def _eod_inputs(**kw):
    return eod.EodInputs(trade_date=TRADE, defs=_make_defs(),
                         opt_stats=_make_eod_stats(**kw), fut_settles=_make_eod_fut())


@pytest.fixture(scope="module")
def eod_result():
    return eod.process_day_eod(_eod_inputs(), dict(eod.PARAMS))


def test_eod_rows_match_black76(eod_result):
    df, meta, _ = eod_result
    # the 0DTE pair (7/8) expires exactly at the settlement reference — no
    # recoverable time value left, so their IV is legitimately NaN
    assert len(df) == 8 and meta["n_iv_failed"] == 2
    for cid, right, k, month, exp, sig in CONTRACTS:
        row = df.loc[df["instrument_id"] == cid].iloc[0]
        if cid in (7, 8):
            assert np.isnan(row["iv"]) and np.isnan(row["gamma"])
        else:
            np.testing.assert_allclose(row["iv"], sig, atol=1e-6)
            T = float(gx.time_to_expiry_years(np.array([exp]), T_REF_D, 1.0)[0])
            g = gx.black76_greeks(F_EOD[month], k, sig, T, R, right == "C")
            np.testing.assert_allclose(row["delta"], g["delta"], rtol=1e-5)
            np.testing.assert_allclose(row["gamma"], g["gamma"], rtol=1e-5)
            sign = 1.0 if right == "C" else -1.0
            np.testing.assert_allclose(
                row["gex"], sign * g["gamma"] * OI_C[cid] * MULT * F_EOD[month] ** 2 * 0.01,
                rtol=1e-5)
        assert row["oi"] == OI_C[cid]
        assert row["cleared_volume"] == cid * 11
        assert row["lowest_offer"] == pytest.approx(0.4 + cid)
        assert row["highest_bid"] == pytest.approx(2.5 + cid)
    assert df["oi_is_final"].all() and not df["oi_is_stale"].any()
    assert meta["oi_source"] == "next_final"
    assert meta["spot_month"] == "XXU6"  # month with the largest OI
    assert meta["parity_max_error"] < 1e-6


def test_eod_index_shape(eod_result):
    df, _, _ = eod_result
    assert df.index.name == "expiration"
    assert str(df.index.tz) == NY
    assert df.index.is_monotonic_increasing
    first_exp = df.loc[df.index == df.index.min()]
    assert first_exp["strike"].is_monotonic_increasing


def test_eod_theoretical_flag_and_v2(eod_result):
    df, meta, _ = eod_result
    assert df.loc[df["instrument_id"] == 6, "settle_is_theoretical"].iloc[0]
    assert not df.loc[df["instrument_id"] == 1, "settle_is_theoretical"].iloc[0]
    total = sum(OI_C.values())
    np.testing.assert_allclose(meta["pct_oi_theoretical"], OI_C[6] / total * 100, rtol=1e-9)


def test_eod_oi_prelim_fallback():
    df, meta, _ = eod.process_day_eod(_eod_inputs(oi_final=False), dict(eod.PARAMS))
    assert meta["oi_source"] == "next_prelim"
    assert not df["oi_is_final"].any()
    assert (df["oi"].to_numpy() ==
            np.array([OI_C[c] + 5 for c in df["instrument_id"]])).all()


def test_eod_oi_stale_fallback():
    df, meta, warns = eod.process_day_eod(_eod_inputs(oi_present=False), dict(eod.PARAMS))
    assert meta["oi_source"] == "stale_prev"
    assert df["oi_is_stale"].all()
    assert any("stale" in w for w in warns)


def test_eod_parity_detects_bad_settlement():
    _, meta, _ = eod.process_day_eod(_eod_inputs(perturb_cid=1), dict(eod.PARAMS))
    assert meta["parity_max_error"] > 1.0


def test_eod_extremes_never_affect_greeks(eod_result):
    df, _, _ = eod_result
    stats = _make_eod_stats()
    stats.loc[stats["stat_type"].isin([gx.STAT_LOW_OFFER, gx.STAT_HIGH_BID]), "price"] = 999.0
    df2, _, _ = eod.process_day_eod(
        eod.EodInputs(trade_date=TRADE, defs=_make_defs(), opt_stats=stats,
                      fut_settles=_make_eod_fut()), dict(eod.PARAMS))
    np.testing.assert_allclose(df2["iv"].to_numpy(), df["iv"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(df2["gex"].to_numpy(), df["gex"].to_numpy(), rtol=1e-12)
    assert (df2["lowest_offer"] == 999.0).all()
