"""
Validation checks for a vol_realized_z regime run (detector plan §8).

Reads ONLY the run folder (daily parquets + meta.json) — never the input
candles — and prints:

1. Distribution  — vol_z_obs should be roughly bell-shaped, centered near 0,
                   mostly within ±2. All-positive or all-at-zero = broken
                   yardstick.
2. U-shape       — mean vol_z_obs per snapshot clock time should sit near 0
                   everywhere. A morning/afternoon tilt means the matched
                   windows aren't matched.
3. Predictiveness— days whose 10:00 vol_z exceeded +1 should show above-
                   average AFTERNOON realized vol (12:00→16:00), measured as
                   a z against a trailing 120-day robust yardstick. If not,
                   the detector carries no information — do not filter any
                   backtest with it.
4. Persistence   — day-to-day transition matrix of the final-row state.
                   Diagonal comfortably above 70% = regime; near 100% = dead
                   zone too wide / lookback too long.
5. Balance       — share of days per state, at final and at 10:00.

Usage: run from the Scripts module (or `python vol_regime_validation.py
[run_folder]`). Default run folder: the first configured data root holding
regimes/ES/ES_vol_realized_z.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUN = ("ES", "ES_vol_realized_z")
REAL_STATES = ("low", "normal", "high")


def find_run_folder() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    repo = Path(__file__).resolve().parents[1]
    roots = ["D:/market_data"]
    try:
        cfg = json.loads((repo / "settings.json").read_text(encoding="utf-8"))
        roots = cfg.get("data_roots", roots)
    except (OSError, json.JSONDecodeError):
        pass
    for root in roots:
        p = Path(root)
        p = p if p.is_absolute() else repo / p
        candidate = p / "regimes" / DEFAULT_RUN[0] / DEFAULT_RUN[1]
        if (candidate / "meta.json").exists():
            return candidate
    raise SystemExit(f"no {'/'.join(DEFAULT_RUN)} run found in any data root "
                     f"— pass the run folder as an argument")


def load(run_dir: Path):
    files = sorted(f for f in run_dir.glob("*.parquet")
                   if f.stem[0:1].isdigit())
    frames = {f.stem: pd.read_parquet(f) for f in files}
    print(f"run: {run_dir}\ndays loaded: {len(frames)}")
    return frames


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_distribution(frames) -> None:
    section("1. DISTRIBUTION of vol_z_obs (known rows)")
    z = np.concatenate([df["vol_z_obs"].to_numpy(dtype=float)
                        for df in frames.values()])
    z = z[np.isfinite(z)]
    print(f"n={len(z):,}  mean={z.mean():+.3f}  median={np.median(z):+.3f}  "
          f"std={z.std():.3f}")
    for lim in (1.0, 2.0, 3.0):
        print(f"  within ±{lim:.0f}: {np.mean(np.abs(z) <= lim) * 100:5.1f}%")
    deciles = np.percentile(z, np.arange(10, 100, 10))
    print("  deciles:", "  ".join(f"{d:+.2f}" for d in deciles))


def check_ushape(frames) -> None:
    section("2. U-SHAPE — mean vol_z_obs by snapshot clock time")
    rows = []
    for df in frames.values():
        z = df["vol_z_obs"]
        for ts, v in z.items():
            if np.isfinite(v):
                rows.append((ts.strftime("%H:%M"), v))
    g = pd.DataFrame(rows, columns=["clock", "z"]).groupby("clock")["z"]
    stats = g.agg(["mean", "count"])
    # session order: evening first
    order = sorted(stats.index, key=lambda t: (t < "17:00", t))
    worst = 0.0
    for t in order:
        mean, count = stats.loc[t, "mean"], int(stats.loc[t, "count"])
        bar = "#" * int(min(abs(mean), 1.0) * 40)
        flag = "  <-- off" if abs(mean) > 0.3 else ""
        worst = max(worst, abs(mean))
        print(f"  {t}  {mean:+.3f}  (n={count:5d}) {bar}{flag}")
    print(f"verdict: worst |mean| = {worst:.3f} "
          f"({'OK — windows are matched' if worst <= 0.3 else 'NOT matched'})")


def _row_at(df: pd.DataFrame, clock: str):
    hits = df.index.strftime("%H:%M") == clock
    return df[hits].iloc[0] if hits.any() else None


def _afternoon_ln_vol(df: pd.DataFrame) -> float:
    """ln per-bar vol 12:00→16:00, reconstructed from stored rth columns:
    ss = σ²·n at 16:00 minus at 12:00."""
    r12, r16 = _row_at(df, "12:00"), _row_at(df, "16:00")
    if r12 is None or r16 is None:
        return np.nan
    if not (np.isfinite(r12["rth_vol"]) and np.isfinite(r16["rth_vol"])):
        return np.nan
    ss = r16["rth_vol"] ** 2 * r16["rth_vol_n"] - \
        r12["rth_vol"] ** 2 * r12["rth_vol_n"]
    n = r16["rth_vol_n"] - r12["rth_vol_n"]
    return float(np.log(np.sqrt(ss / n))) if n > 0 and ss > 0 else np.nan


def check_predictiveness(frames) -> None:
    section("3. PREDICTIVENESS — 10:00 vol_z vs afternoon realized vol")
    recs = []
    for date, df in frames.items():
        ten = _row_at(df, "10:00")
        if ten is None:
            continue
        recs.append((date, float(ten["vol_z"]), _afternoon_ln_vol(df)))
    t = pd.DataFrame(recs, columns=["date", "z10", "ln_aft"]).set_index("date")

    # trailing robust z of afternoon vol, so 2010 and 2022 are comparable
    med = t["ln_aft"].rolling(120, min_periods=60).median().shift(1)
    mad = (t["ln_aft"] - med).abs().rolling(120, min_periods=60) \
        .median().shift(0) * 1.4826
    t["aft_z"] = (t["ln_aft"] - med) / mad.replace(0.0, np.nan)
    t = t[np.isfinite(t["z10"]) & np.isfinite(t["aft_z"])]

    groups = [("10:00 z > +1", t[t["z10"] > 1.0]),
              ("10:00 z in [-1, +1]", t[t["z10"].abs() <= 1.0]),
              ("10:00 z < -1", t[t["z10"] < -1.0])]
    print(f"days evaluated: {len(t)}")
    for label, grp in groups:
        if not len(grp):
            print(f"  {label:22s}  (no days)")
            continue
        above = np.mean(grp["aft_z"] > 0) * 100
        print(f"  {label:22s}  n={len(grp):4d}  mean aft_z={grp['aft_z'].mean():+.3f}  "
              f"median={grp['aft_z'].median():+.3f}  above-median share={above:.0f}%")
    rho = t["z10"].rank().corr(t["aft_z"].rank())
    print(f"rank correlation (z10 vs afternoon z): {rho:+.3f}")
    hi = t[t["z10"] > 1.0]
    ok = len(hi) and hi["aft_z"].median() > 0
    print(f"verdict: {'carries information' if ok else 'NO information — do not filter backtests with this'}")


def check_persistence(frames) -> None:
    section("4. PERSISTENCE — day-to-day transition matrix (final row)")
    finals = {d: str(df["vol_state"].iloc[-1]) for d, df in frames.items()}
    days = sorted(finals)
    pairs = [(finals[a], finals[b]) for a, b in zip(days, days[1:])
             if finals[a] in REAL_STATES and finals[b] in REAL_STATES]
    counts = pd.crosstab(
        pd.Series([a for a, _ in pairs], name="from"),
        pd.Series([b for _, b in pairs], name="to")).reindex(
        index=REAL_STATES, columns=REAL_STATES, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    print(counts.to_string())
    print()
    print((probs * 100).round(1).to_string())
    diag = [probs.loc[s, s] for s in REAL_STATES if counts.loc[s].sum() > 0]
    lo, hi = min(diag) * 100, max(diag) * 100
    if lo < 70:
        verdict = "diagonal below 70% — consider widening the dead zone"
    elif hi > 97:
        verdict = "diagonal near 100% — dead zone too wide or z barely moves"
    else:
        verdict = "healthy persistence"
    print(f"verdict: diagonal {lo:.0f}%–{hi:.0f}% — {verdict}")


def check_balance(frames) -> None:
    section("5. BALANCE — share of days per state")
    def at_1000(df):
        row = _row_at(df, "10:00")
        return row["vol_state"] if row is not None else "missing"

    for label, getter in (("final row", lambda df: df["vol_state"].iloc[-1]),
                          ("at 10:00", at_1000)):
        states = pd.Series([getter(df) for df in frames.values()])
        share = states.value_counts(normalize=True) * 100
        counts = states.value_counts()
        line = "  ".join(f"{s}: {counts.get(s, 0)} ({share.get(s, 0):.1f}%)"
                         for s in list(REAL_STATES) + ["unknown"])
        print(f"  {label:10s}  {line}")
    print("note: three states over one tradable year is ~100 days each at "
          "best — enough to see a large effect, nowhere near enough for "
          "per-regime parameter tuning. Do not use these labels to search.")


if __name__ == "__main__":
    run_dir = find_run_folder()
    frames = load(run_dir)
    check_distribution(frames)
    check_ushape(frames)
    check_predictiveness(frames)
    check_persistence(frames)
    check_balance(frames)
