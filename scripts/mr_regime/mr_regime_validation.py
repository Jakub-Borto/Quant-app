"""
Validation checks for an mr_multi regime run (detector plan §9).

This matters MORE here than it did for volatility. Vol's validation confirmed a
signal we expected to exist; this one is genuinely deciding whether the
mean-reversion axis is usable at all. Nothing downstream should join these
labels before reading the output of this script.

Checks, in the order they should be read:

1. Distribution  - every z should be bell-ish and centred near 0. A VR-z that
                   isn't centred means the ln transform or the yardstick is off.
2. U-shape       - mean z per snapshot clock time should sit near 0 EVERYWHERE.
                   A morning/afternoon tilt in the z (not the raw) means the
                   matched windows aren't matched.
3. Agreement     - the headline diagnostic. Cross-tabulate the measures'
                   verdicts on the same snapshots. Prediction: VR most stable,
                   ER close, ADF marginal, R/S nearly random. If everything
                   disagrees with everything, the axis is noise at this
                   timescale - a real, useful finding.
4. Stability     - state-change rate of the primary label by time of day. If
                   10:00 is chaotic and 11:30 is calm, IVB (which reads 10:00)
                   cannot use this axis as early as it uses volatility.
5. Persistence   - day-to-day transition matrix of the final-row state. Expect
                   a diagonal WEAKER than vol's. Near 33% (three states, no
                   persistence) means the point-in-time label carries little
                   day-ahead information, and this axis is for retrospective
                   research, not live filtering.
6. Predictiveness- days with a strongly trending 10:00 label: did the afternoon
                   actually trend more? This is the real test. It needs the
                   input candles (an afternoon VR cannot be reconstructed from
                   the stored columns - variance ratios are not additive), so
                   this is the one check that reads outside the run folder.
7. Balance       - share of days per state.

A weak result in 5 and 6 is the EXPECTED outcome, not a failure. It would mean:
use these labels to ask "does IVB structurally work on trending days"
(retrospective, legitimate) and NOT to filter live trades (fitting noise).

Usage: run from the Scripts module (or `python mr_regime_validation.py
[run_folder]`). Default run folder: the first configured data root holding
regimes/ES/ES_mr_multi.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN = ("ES", "ES_mr_multi")
REAL_STATES = ("reverting", "neutral", "trending")


def find_run_folder() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    roots = ["D:/market_data"]
    try:
        cfg = json.loads((REPO / "settings.json").read_text(encoding="utf-8"))
        roots = cfg.get("data_roots", roots)
    except (OSError, json.JSONDecodeError):
        pass
    for root in roots:
        p = Path(root)
        p = p if p.is_absolute() else REPO / p
        candidate = p / "regimes" / DEFAULT_RUN[0] / DEFAULT_RUN[1]
        if (candidate / "meta.json").exists():
            return candidate
    raise SystemExit(f"no {'/'.join(DEFAULT_RUN)} run found in any data root "
                     f"- pass the run folder as an argument")


def load(run_dir: Path):
    files = sorted(f for f in run_dir.glob("*.parquet")
                   if f.stem[0:1].isdigit())
    frames = {f.stem: pd.read_parquet(f) for f in files}
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"run: {run_dir}\ndays loaded: {len(frames)}")
    extras = meta.get("script_extras", {})
    print(f"primary column: {extras.get('primary_regime_column')}  "
          f"(q_label used: {extras.get('q_label_used')})")
    if extras.get("q_label_fallback"):
        print(f"  NOTE q_label fell back: {extras['q_label_fallback']}")
    return frames, meta


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def effective(df: pd.DataFrame, base: str) -> np.ndarray:
    """The scope handoff the detector labels from: rth_ once the RTH window is
    open, gbx_ before it. `base` is a column name without the scope prefix."""
    post = df["n_bars_rth"].to_numpy() > 0
    return np.where(post, df[f"rth_{base}"].to_numpy(dtype=float),
                    df[f"gbx_{base}"].to_numpy(dtype=float))


def z_columns(meta: dict) -> list[str]:
    """Every scope-less z base in the run (the Lo-MacKinlay significance z is
    excluded - it is a per-snapshot statistic, not a matched-window z)."""
    return sorted(c[len("rth_"):] for c in meta["schema"]["columns"]
                  if c.startswith("rth_mr_") and "_z" in c
                  and "_lm_z" not in c)


# ── 1. distribution ──────────────────────────────────────────────────────────

def check_distribution(frames, meta) -> None:
    section("1. DISTRIBUTION of every z (effective scope, known rows)")
    print(f"{'column':26s} {'n':>7s} {'mean':>7s} {'med':>7s} {'std':>6s} "
          f"{'|z|<=1':>7s} {'|z|<=2':>7s}  {'deciles 10/50/90':>22s}")
    for base in z_columns(meta):
        z = np.concatenate([effective(df, base) for df in frames.values()])
        z = z[np.isfinite(z)]
        if not len(z):
            print(f"{base:26s} {'(no finite values)':>40s}")
            continue
        d10, d50, d90 = np.percentile(z, [10, 50, 90])
        # robust scaling centres the MEDIAN, so a skewed measure legitimately
        # has a non-zero mean. Flag on the median (the thing that should be 0)
        # and call out skew separately - they are different problems.
        flag = ""
        if abs(np.median(z)) > 0.15:
            flag = "  <-- not centred"
        elif abs(z.mean()) > 0.25:
            flag = "  <-- skewed (median ok; states will lean one way)"
        print(f"{base:26s} {len(z):7d} {z.mean():+7.3f} {np.median(z):+7.3f} "
              f"{z.std():6.3f} {np.mean(np.abs(z) <= 1) * 100:6.1f}% "
              f"{np.mean(np.abs(z) <= 2) * 100:6.1f}%  "
              f"{d10:+6.2f} {d50:+6.2f} {d90:+6.2f}{flag}")
    print("\na z that is not centred near zero means the transform or the "
          "yardstick is off for that measure - not that the market changed.")


# ── 2. U-shape ───────────────────────────────────────────────────────────────

def check_ushape(frames, meta) -> None:
    section("2. U-SHAPE - mean z by snapshot clock time (must be ~0 everywhere)")
    bases = z_columns(meta)
    rows = []
    for df in frames.values():
        clocks = df.index.strftime("%H:%M")
        for base in bases:
            z = effective(df, base)
            for clock, v in zip(clocks, z):
                if np.isfinite(v):
                    rows.append((base, clock, v))
    tab = pd.DataFrame(rows, columns=["base", "clock", "z"])
    piv = tab.pivot_table(index="clock", columns="base", values="z",
                          aggfunc="mean")
    order = sorted(piv.index, key=lambda t: (t < "17:00", t))
    piv = piv.reindex(order)
    worst = piv.abs().max()
    show = [b for b in bases if b in piv.columns][:8]
    print(piv[show].round(2).to_string())
    print("\nworst |mean z| per column:")
    for base in bases:
        if base not in worst:
            continue
        verdict = "OK" if worst[base] <= 0.35 else "NOT matched"
        print(f"  {base:26s} {worst[base]:5.2f}  {verdict}")


# ── 3. agreement matrix (the headline diagnostic) ────────────────────────────

def check_agreement(frames, meta) -> None:
    section("3. AGREEMENT MATRIX - do the measures agree with each other?")
    state_cols = [c for c in meta["schema"]["columns"] if c.startswith("mr_state")]
    # one representative per family, plus every anchor for ADF/ER on `open`
    picks = [c for c in ("mr_state", "mr_state_hurst", "mr_state_adf_open",
                         "mr_state_er_open", "mr_state_adf_vwap_rth",
                         "mr_state_er_vwap_rth", "mr_state_adf_ema",
                         "mr_state_er_ema") if c in state_cols]

    # R/S deliberately has no state column in the detector (it is the noisy
    # comparator, not a label), but it is the ONE measure that is genuinely
    # independent of the variance-ratio family - leaving it out of the
    # agreement matrix would omit the only real outside opinion. Derive a
    # state here by thresholding its z, no hysteresis, and say so.
    enter = float(meta.get("params", {}).get("enter_trend", 0.90))
    rs_col = "rs_thresholded"
    parts = []
    for df in frames.values():
        sub = df[picks].copy()
        z = effective(df, "mr_hurst_rs_z")
        sub[rs_col] = np.where(~np.isfinite(z), "unknown",
                               np.where(z > enter, "trending",
                                        np.where(z < -enter, "reverting",
                                                 "neutral")))
        parts.append(sub)
    picks = picks + [rs_col]
    big = pd.concat(parts, ignore_index=True)
    print(f"({rs_col} = hurst_rs z thresholded at +/-{enter:.2f}, no "
          f"hysteresis - the detector emits no state for R/S)")
    known = big[(big != "unknown").all(axis=1)]
    print(f"snapshots where every listed measure has an answer: "
          f"{len(known):,} of {len(big):,}")
    if not len(known):
        print("nothing to compare")
        return

    print("\npairwise agreement %  (chance with 3 states is ~33%, but the "
          "states are unbalanced so compare to the marginal-match baseline)")
    header = "".join(f"{c.replace('mr_state', 'primary' if c == 'mr_state' else '')[:11]:>12s}"
                     for c in picks)
    print(f"{'':26s}{header}")
    for a in picks:
        cells = ""
        for b in picks:
            if a == b:
                cells += f"{'-':>12s}"
                continue
            agree = float((known[a] == known[b]).mean()) * 100
            cells += f"{agree:11.1f}%"
        print(f"{a:26s}{cells}")

    # baseline: what agreement would independent columns with these marginals
    # produce? Anything at or below it is noise agreeing by accident.
    print("\nindependence baseline (sum p_a(s)*p_b(s)) and lift over it:")
    for i, a in enumerate(picks):
        for b in picks[i + 1:]:
            pa = known[a].value_counts(normalize=True)
            pb = known[b].value_counts(normalize=True)
            base = float(sum(pa.get(s, 0) * pb.get(s, 0) for s in REAL_STATES))
            got = float((known[a] == known[b]).mean())
            print(f"  {a:24s} vs {b:24s} {got * 100:5.1f}% vs "
                  f"{base * 100:5.1f}% baseline  (lift {got - base:+.3f})")

    print("\nfull 3x3 cross-tab, primary vs each other measure:")
    for b in picks[1:]:
        ct = pd.crosstab(known["mr_state"], known[b]).reindex(
            index=REAL_STATES, columns=REAL_STATES, fill_value=0)
        print(f"\n  primary (rows) vs {b} (cols)")
        print("  " + (ct / ct.to_numpy().sum() * 100).round(1).to_string()
              .replace("\n", "\n  "))


# ── 4. when does the label stabilise? ────────────────────────────────────────

def check_stability(frames) -> None:
    section("4. STABILITY - primary label change rate by time of day")
    rows = []
    for df in frames.values():
        states = df["mr_state"].to_numpy()
        clocks = df.index.strftime("%H:%M")
        for i in range(1, len(states)):
            if states[i] == "unknown" or states[i - 1] == "unknown":
                continue
            rows.append((clocks[i], states[i] != states[i - 1]))
    if not rows:
        print("no known transitions")
        return
    tab = pd.DataFrame(rows, columns=["clock", "changed"])
    g = tab.groupby("clock")["changed"].agg(["mean", "count"])
    order = sorted(g.index, key=lambda t: (t < "17:00", t))
    for t in order:
        rate, n = g.loc[t, "mean"] * 100, int(g.loc[t, "count"])
        bar = "#" * int(rate / 2)
        print(f"  {t}  {rate:5.1f}%  (n={n:5d}) {bar}")
    rth = [t for t in order if "09:30" <= t <= "16:00"]
    if len(rth) >= 3:
        early = g.loc[rth[:2], "mean"].mean() * 100
        late = g.loc[rth[-3:], "mean"].mean() * 100
        print(f"\nearly RTH change rate {early:.1f}% vs late {late:.1f}%")
        print("CAVEAT: the first post-open snapshot is the scope HANDOFF - the "
              "primary label switches from the overnight (gbx) window to a "
              "30-bar RTH window there, so a high change rate at that one "
              "snapshot is structural, not noise. Read the SECOND and third "
              "RTH rows for how noisy the label genuinely is.")
        if early > late * 1.5:
            print("  -> the label is materially noisier early. IVB reads "
                  "10:00; it cannot use this axis as early as it uses vol.")
        else:
            print("  -> no strong early/late difference in label churn.")


# ── 5. persistence ───────────────────────────────────────────────────────────

def check_persistence(frames) -> None:
    section("5. PERSISTENCE - day-to-day transition matrix (final row)")
    finals = {d: str(df["mr_state"].iloc[-1]) for d, df in frames.items()}
    days = sorted(finals)
    pairs = [(finals[a], finals[b]) for a, b in zip(days, days[1:])
             if finals[a] in REAL_STATES and finals[b] in REAL_STATES]
    if not pairs:
        print("no consecutive known days")
        return
    counts = pd.crosstab(pd.Series([a for a, _ in pairs], name="from"),
                         pd.Series([b for _, b in pairs], name="to")).reindex(
        index=REAL_STATES, columns=REAL_STATES, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    print(counts.to_string())
    print()
    print((probs * 100).round(1).to_string())
    marg = counts.sum(axis=0) / counts.to_numpy().sum()
    diag = {s: probs.loc[s, s] for s in REAL_STATES if counts.loc[s].sum() > 0}
    print("\nstate      P(stay)   unconditional   lift")
    for s, p in diag.items():
        print(f"  {s:10s} {p * 100:5.1f}%      {marg[s] * 100:5.1f}%       "
              f"{p - marg[s]:+.3f}")
    lo = min(diag.values()) * 100
    lift = np.mean([diag[s] - marg[s] for s in diag])
    if lift < 0.05:
        verdict = ("NO day-ahead information - this axis DESCRIBES, it does "
                   "not filter. Use it retrospectively only.")
    elif lo < 50:
        verdict = ("weak persistence - usable for research, treat live "
                   "filtering as unproven")
    else:
        verdict = "persistent enough to condition on"
    print(f"\nverdict: mean lift over unconditional {lift:+.3f} - {verdict}")


# ── 6. predictiveness (the real test; reads the input candles) ───────────────

def _load_detector():
    from modules.common.backend.plugins import list_plugins, load_module
    refs = {r.name: r for r in list_plugins([REPO / "regime_detectors"])}
    return load_module(refs["mr_multi"])


def check_predictiveness(frames, meta) -> None:
    section("6. PREDICTIVENESS - 10:00 label vs the afternoon's actual VR")
    src = Path(meta.get("input_dataset_path", ""))
    if not src.exists():
        print(f"input dataset not found ({src}) - skipping. This is the one "
              f"check that needs the candles: a variance ratio is not additive,"
              f" so the afternoon's VR cannot be rebuilt from stored columns.")
        return
    mr = _load_detector()
    q = int(meta.get("script_extras", {}).get("q_label_used", 10))
    tz = "America/New_York"

    recs = []
    for date, df in frames.items():
        hits = df.index.strftime("%H:%M") == "10:00"
        if not hits.any():
            continue
        ten = df[hits].iloc[0]
        if ten["mr_state"] == "unknown" or not np.isfinite(ten["mr_z"]):
            continue
        path = src / f"{date}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path)
        lo = bars.index.searchsorted(pd.Timestamp(f"{date} 12:00", tz=tz))
        hi = bars.index.searchsorted(pd.Timestamp(f"{date} 16:00", tz=tz))
        if hi - lo < 60:
            continue
        stats = mr._return_stats(bars, False)
        vr, lm_z, _blocks = mr._vr_reference(stats["r"][lo:hi],
                                             stats["valid"][lo:hi], q)
        if not np.isfinite(vr) or vr <= 0:
            continue
        recs.append((date, str(ten["mr_state"]), float(ten["mr_z"]),
                     float(np.log(vr)), float(lm_z)))

    t = pd.DataFrame(recs, columns=["date", "state10", "z10", "aft_lnvr",
                                    "aft_lm_z"]).set_index("date")
    if len(t) < 50:
        print(f"only {len(t)} usable days - not enough to conclude anything")
        return
    # trailing robust z of the afternoon ln VR so different eras compare
    med = t["aft_lnvr"].rolling(120, min_periods=60).median().shift(1)
    mad = (t["aft_lnvr"] - med).abs().rolling(120, min_periods=60).median() \
        * 1.4826
    t["aft_z"] = (t["aft_lnvr"] - med) / mad.replace(0.0, np.nan)
    t = t[np.isfinite(t["aft_z"])]
    print(f"days evaluated: {len(t)}   (afternoon = 12:00-16:00, VR at q={q})")

    print("\nby 10:00 LABEL:")
    for state in REAL_STATES:
        grp = t[t["state10"] == state]
        if not len(grp):
            print(f"  {state:10s} (no days)")
            continue
        print(f"  {state:10s} n={len(grp):4d}  mean afternoon z="
              f"{grp['aft_z'].mean():+.3f}  median={grp['aft_z'].median():+.3f}"
              f"  share above 0: {np.mean(grp['aft_z'] > 0) * 100:.0f}%")
    spread = (t[t["state10"] == "trending"]["aft_z"].median()
              - t[t["state10"] == "reverting"]["aft_z"].median())

    print("\nby 10:00 z BUCKET (continuous, ignores the hysteresis):")
    for label, grp in (("z > +1", t[t["z10"] > 1]),
                       ("|z| <= 1", t[t["z10"].abs() <= 1]),
                       ("z < -1", t[t["z10"] < -1])):
        if not len(grp):
            print(f"  {label:10s} (no days)")
            continue
        print(f"  {label:10s} n={len(grp):4d}  mean afternoon z="
              f"{grp['aft_z'].mean():+.3f}  median={grp['aft_z'].median():+.3f}")
    rho = t["z10"].rank().corr(t["aft_z"].rank())
    print(f"\nrank correlation (10:00 z vs afternoon z): {rho:+.3f}")
    print(f"median afternoon z, trending minus reverting: {spread:+.3f}")
    # ~1/sqrt(n) is the standard error of a rank correlation under the null
    se = 1.0 / np.sqrt(len(t))
    print(f"(rank-correlation standard error at n={len(t)} is ~{se:.3f})")
    if rho < -2 * se or spread < -0.1:
        print("verdict: the relationship runs the WRONG way - investigate the "
              "sign convention before using this anywhere.")
    elif rho > 2 * se and spread > 0.05:
        print("verdict: the morning label carries some forward information - "
              "weak, but the sign is right and the label ordering is monotone."
              " Size any use accordingly; this is not a filter-grade edge.")
    else:
        print("verdict: NO forward information. Do not filter live trades with "
              "this. Retrospective use (does IVB work on trending days) stays "
              "legitimate - that is the plan's expected outcome, not a bug.")


# ── 7. balance ───────────────────────────────────────────────────────────────

def check_balance(frames) -> None:
    section("7. BALANCE - share of days per state")

    def at_1000(df):
        hits = df.index.strftime("%H:%M") == "10:00"
        return df[hits].iloc[0]["mr_state"] if hits.any() else "missing"

    for label, getter in (("final row", lambda df: df["mr_state"].iloc[-1]),
                          ("at 10:00", at_1000)):
        states = pd.Series([getter(df) for df in frames.values()])
        counts = states.value_counts()
        share = states.value_counts(normalize=True) * 100
        line = "  ".join(f"{s}: {counts.get(s, 0)} ({share.get(s, 0):.1f}%)"
                         for s in list(REAL_STATES) + ["unknown"])
        print(f"  {label:10s}  {line}")
    print("\nnote: three states over one tradable year is ~80 days each at "
          "best. Enough to see a large effect, nowhere near enough for "
          "per-regime parameter tuning. Do not use these labels to search.")


if __name__ == "__main__":
    run_dir = find_run_folder()
    frames, meta = load(run_dir)
    check_distribution(frames, meta)
    check_ushape(frames, meta)
    check_agreement(frames, meta)
    check_stability(frames)
    check_persistence(frames)
    check_predictiveness(frames, meta)
    check_balance(frames)
