"""
Strategy Combiner: merge primitive, pool/split/floors, compatibility gate,
greedy+swap+λ, OOS evaluation, persistence (spec §14/§15).
"""

import json
import math

import pandas as pd
import pytest

from optimization.combine.compat import check_compatibility
from optimization.combine.evaluate import evaluate_set
from optimization.combine.io import (list_combine_runs, load_combine_run,
                                     save_combine_run)
from optimization.combine.merge import (merge_streams, merged_total,
                                        no_overlap_walk, trades_to_tuples)
from optimization.combine.pool import (apply_min_trades, build_pool,
                                       discover_entry_runs, list_containers,
                                       load_entry_runs, split_date_boundary,
                                       split_pool)
from optimization.combine.runner import run_combine
from optimization.combine.select import correlation_matrix, greedy_select

TZ = "America/New_York"


def _trades_df(rows, param_name="p"):
    """rows: (date, entry 'HH:MM', exit 'HH:MM', pnl, param_value[, type[, bucket]])"""
    out = []
    for row in rows:
        date, entry, exit_, pnl, pval = row[:5]
        ttype  = row[5] if len(row) > 5 else "alpha"
        bucket = row[6] if len(row) > 6 else "normal"
        out.append({
            "date":       pd.Timestamp(date),
            "entry_time": pd.Timestamp(f"{date} {entry}", tz=TZ),
            "exit_time":  pd.Timestamp(f"{date} {exit_}", tz=TZ),
            "pnl_ticks":  float(pnl),
            param_name:   pval,
            "trade_type": ttype,
            "day_bucket": bucket,
            "pnl_points": float(pnl) / 4,     # present but must never be read
        })
    return pd.DataFrame(out)


def _write_run(root, container, name, trades, param_name="p",
               ticker="ES", tpp=4, start="2026-01-05", end="2026-01-16"):
    run_dir = root / container / name
    run_dir.mkdir(parents=True)
    trades.to_parquet(run_dir / "trades.parquet")
    meta = {"ticker": ticker, "dataset": "Futures/ES/ES_test",
            "ticks_per_point": tpp, "start_date": start, "end_date": end,
            "axes": {"x": {"param": param_name, "values": sorted(
                trades[param_name].unique().tolist())},
                "y": None, "slider": None, "slider2": None}}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


# ── merge primitive ───────────────────────────────────────────────────────────

def _tuples(rows, vid="v"):
    return trades_to_tuples(_trades_df([(*r, 1) for r in rows]), vid)


def test_no_overlap_walk_basics():
    # B overlaps A -> skipped; C starts exactly at A's exit -> kept;
    # D later same day -> kept (multiple non-overlapping same-day trades)
    trades = _tuples([
        ("2026-01-05", "09:00", "10:00", 10),   # A kept
        ("2026-01-05", "09:30", "09:45", 99),   # B overlaps A -> skipped
        ("2026-01-05", "10:00", "11:00", 5),    # C entry == A exit -> kept
        ("2026-01-05", "13:00", "14:00", 3),    # D kept
    ])
    kept = no_overlap_walk(iter(trades))
    assert [t[4] for t in kept] == [10.0, 5.0, 3.0]
    assert merged_total(iter(trades)) == 18.0


def test_tie_break_entry_then_exit_then_key():
    a = _tuples([("2026-01-05", "09:00", "11:00", 1)], vid="zz")
    b = _tuples([("2026-01-05", "09:00", "10:00", 2)], vid="aa")
    kept = no_overlap_walk(merge_streams(a, b))
    assert len(kept) == 1 and kept[0][4] == 2.0    # earlier exit wins the tie
    # equal entry AND exit -> smaller vid wins deterministically
    c = _tuples([("2026-01-05", "09:00", "10:00", 7)], vid="ab")
    kept = no_overlap_walk(merge_streams(b, c))
    assert kept[0][2] == "aa"


def test_incremental_merge_equals_full_remerge():
    s1 = _tuples([("2026-01-05", "09:00", "10:00", 1),
                  ("2026-01-06", "09:00", "10:00", 2)], vid="s1")
    s2 = _tuples([("2026-01-05", "09:30", "11:00", 3),
                  ("2026-01-07", "09:00", "10:00", 4)], vid="s2")
    c  = _tuples([("2026-01-05", "08:55", "09:05", 5)], vid="c")
    incremental = no_overlap_walk(merge_streams(
        list(merge_streams(s1, s2)), c))
    full = no_overlap_walk(iter(sorted(s1 + s2 + c)))
    assert incremental == full


def test_within_variant_overlaps_also_cleaned():
    t = _tuples([("2026-01-05", "09:00", "12:00", 10),
                 ("2026-01-05", "10:00", "11:00", 99)])
    assert merged_total(iter(t)) == 10.0


# ── synthetic container fixture ───────────────────────────────────────────────

@pytest.fixture()
def container(tmp_path):
    """
    Two entry runs over 10 business days 2026-01-05..2026-01-16.

    run_a ('alpha' entries, param p):
      p=1 'C' : 09:00-10:00 daily, +10  (10 trades, the anchor)
      p=2 'A' : 09:30-10:30 daily, +8   (fully collides with C)
      p=3 'B' : 12:00-13:00 daily, +5   (independent)
    run_b ('beta' entries, param q):
      q=1 'D' : 14:00-15:00 daily, +2   (independent, small)
      q=2 'E' : 14:00-15:00 on cpi days only, +50  (day-filterable)
    """
    days = [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range("2026-01-05", "2026-01-16")]
    rows_a = []
    for d in days:
        rows_a += [(d, "09:00", "10:00", 10, 1),
                   (d, "09:30", "10:30", 8, 2),
                   (d, "12:00", "13:00", 5, 3)]
    rows_b = [(d, "14:00", "15:00", 2, 1, "beta") for d in days]
    rows_b += [(d, "14:00", "15:00", 50, 2, "beta", "cpi") for d in days[:2]]

    _write_run(tmp_path, "cont", "run_a", _trades_df(rows_a))
    _write_run(tmp_path, "cont", "run_b",
               _trades_df(rows_b, param_name="q"), param_name="q")

    # decoys: incomplete folder + _combined must both be invisible
    (tmp_path / "cont" / "broken").mkdir()
    (tmp_path / "cont" / "broken" / "meta.json").write_text("{}")
    (tmp_path / "cont" / "_combined" / "old").mkdir(parents=True)
    (tmp_path / "cont" / "_combined" / "old" / "meta.json").write_text("{}")
    (tmp_path / "cont" / "_combined" / "old" / "trades.parquet").write_bytes(b"")
    return tmp_path


def _pool(container, buckets=None, runs=("run_a", "run_b")):
    loaded = load_entry_runs("cont", list(runs), root=container)
    return build_pool(loaded, buckets or {"normal", "cpi"})


# ── discovery / pool / gate ───────────────────────────────────────────────────

def test_discovery_excludes_combined_and_incomplete(container):
    assert discover_entry_runs("cont", container) == ["run_a", "run_b"]
    assert list_containers(container) == ["cont"]


def test_pool_variant_identity(container):
    pool = _pool(container)
    by_type = {}
    for v in pool:
        by_type.setdefault(v.trade_type, []).append(v)
    assert len(by_type["alpha"]) == 3          # p = 1, 2, 3
    assert len(by_type["beta"]) == 2           # q = 1, 2
    assert all(v.params for v in pool)


def test_day_filter_before_everything(container):
    pool = _pool(container, buckets={"normal"})
    # the cpi-only variant disappears entirely
    assert len(pool) == 4
    assert not any(v.params.get("q") == 2 for v in pool)


def test_gate_hard_stop_and_date_intersection(container, tmp_path):
    loaded = load_entry_runs("cont", ["run_a", "run_b"], root=container)
    metas = {n: m for n, (m, _) in loaded.items()}
    assert check_compatibility(metas)["ok"]

    bad = dict(metas["run_a"]); bad["ticker"] = "NQ"
    gate = check_compatibility({"run_a": bad, "run_b": metas["run_b"]})
    assert not gate["ok"] and any("ticker" in e for e in gate["errors"])

    trimmed = dict(metas["run_a"]); trimmed["start_date"] = "2026-01-07"
    gate = check_compatibility({"run_a": trimmed, "run_b": metas["run_b"]})
    assert gate["ok"] and gate["warnings"]
    assert gate["shared_start"] == pd.Timestamp("2026-01-07")


# ── split + floors ────────────────────────────────────────────────────────────

def test_split_partitions_by_date(container):
    pool = _pool(container)
    boundary = split_date_boundary(pool, 0.7)
    split_pool(pool, boundary)
    for v in pool:
        assert all(t[5] <= boundary for t in v.is_tuples)
        assert all(t[5] > boundary for t in v.oos_tuples)
        assert v.n_is + v.n_oos == len(v.is_tuples) + len(v.oos_tuples)
    # 10 business days, 70% -> 7 IS days for the daily variants
    daily = next(v for v in pool if v.trade_type == "alpha")
    assert daily.n_is == 7 and daily.n_oos == 3


def test_min_trades_floor_per_type(container):
    pool = _pool(container)
    split_pool(pool, split_date_boundary(pool, 0.7))
    # cpi variant has 2 IS trades; floor of 3 for beta drops it, alpha stays
    kept = apply_min_trades(pool, {"beta": 3})
    assert not any(v.trade_type == "beta" and v.params.get("q") == 2
                   for v in kept)
    assert sum(v.trade_type == "alpha" for v in kept) == 3


# ── greedy / λ / swap ─────────────────────────────────────────────────────────

def _prepared_pool(container, buckets=None):
    pool = _pool(container, buckets)
    split_pool(pool, split_date_boundary(pool, 0.7))
    return pool


def test_greedy_scores_merged_not_standalone(container):
    """A (+8/day) collides with C (+10/day); B (+5/day) doesn't. The right
    set is {C, B, D}; standalone ranking would pick A second."""
    pool = _prepared_pool(container, buckets={"normal"})
    result = greedy_select(pool, lam=0.0, max_k=10)
    final = [pool[i].vid for i in result["path"][-1]["members"]]
    assert any("p=1" in v for v in final)          # C
    assert any("p=3" in v for v in final)          # B
    assert any("q=1" in v for v in final)          # D
    assert not any("p=2" in v for v in final)      # A adds nothing merged
    # monotone non-decreasing IS path
    ticks = [p["is_ticks"] for p in result["path"] if p["stage"] == "forward"]
    assert ticks == sorted(ticks)


def test_lambda_suppresses_redundant_variant(container, tmp_path):
    """Two near-identical daily streams at different times: λ=0 keeps both,
    a large λ suppresses the copycat."""
    days = [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range("2026-01-05", "2026-01-16")]
    rows = []
    for i, d in enumerate(days):
        pnl = 10 if i % 2 == 0 else 20
        rows.append((d, "09:00", "10:00", pnl, 1))
        rows.append((d, "14:00", "15:00", pnl - 1, 2))   # correlated twin
    _write_run(tmp_path, "twins", "run_t", _trades_df(rows))
    pool = build_pool(load_entry_runs("twins", ["run_t"], root=tmp_path),
                      {"normal"})
    split_pool(pool, split_date_boundary(pool, 0.7))

    free = greedy_select(pool, lam=0.0, max_k=10)
    assert len(free["path"][-1]["members"]) == 2
    strict = greedy_select(pool, lam=1000.0, max_k=10)
    assert len(strict["path"][-1]["members"]) == 1


def test_determinism(container):
    pool1 = _prepared_pool(container)
    pool2 = _prepared_pool(container)
    p1 = greedy_select(pool1, lam=0.5, max_k=10, n_seeds=3)["path"]
    p2 = greedy_select(pool2, lam=0.5, max_k=10, n_seeds=3)["path"]
    assert [(x["k"], x["is_ticks"], x["members"]) for x in p1] \
        == [(x["k"], x["is_ticks"], x["members"]) for x in p2]


def test_correlation_matrix_shape(container):
    pool = _prepared_pool(container)
    corr = correlation_matrix(pool)
    assert corr.shape == (len(pool), len(pool))
    assert all(abs(corr[i][i] - 1.0) < 1e-9 for i in range(len(pool))
               if pool[i].is_daily.std() > 0)


# ── evaluation ────────────────────────────────────────────────────────────────

def test_evaluate_set_metrics():
    t = _tuples([("2026-01-05", "09:00", "10:00", 10),
                 ("2026-01-06", "09:00", "10:00", -5),
                 ("2026-01-07", "09:00", "10:00", 20)])
    m = evaluate_set([t])
    assert m["total_ticks"] == 25.0
    daily = pd.Series([10.0, -5.0, 20.0])
    assert m["sharpe_daily"] == pytest.approx(
        daily.mean() / daily.std(ddof=1) * math.sqrt(252))
    assert m["max_dd_ticks"] == -5.0            # 10 -> 5 dip
    assert m["max_dd_pct"] == pytest.approx(-50.0)


def test_evaluate_empty_oos():
    m = evaluate_set([[]])
    assert m["empty"] and m["total_ticks"] == 0.0
    assert math.isnan(m["sharpe_daily"])


# ── end-to-end runner + persistence ───────────────────────────────────────────

def test_runner_end_to_end_and_roundtrip(container):
    result = run_combine(
        "cont", ["run_a", "run_b"],
        enabled_buckets={"normal", "cpi"}, floors={}, is_fraction=0.7,
        lam=0.0, max_k=10, root=container,
    )
    path_df = result["path_df"]
    assert path_df["is_oos_peak"].sum() == 1
    assert (path_df["oos_ticks"] != 0).any()
    assert "pnl_points" not in path_df.columns    # ticks only, never recomputed

    run_dir = save_combine_run("cont", "combo test", path_df,
                               result["members_df"], result["meta"],
                               root=container)
    assert run_dir.parent.name == "_combined"
    assert list_combine_runs("cont", container) == [run_dir.name, "old"] \
        or run_dir.name in list_combine_runs("cont", container)

    loaded_path, loaded_members, meta = load_combine_run(
        "cont", run_dir.name, root=container)
    pd.testing.assert_frame_equal(path_df, loaded_path)
    assert meta["is_fraction"] == 0.7
    # a chosen set is fully reproducible from members.parquet
    k1 = loaded_members[loaded_members["k"] == 1]
    assert json.loads(k1.iloc[0]["params"])       # params round-trip

    # combine output must stay invisible to entry-run discovery
    assert discover_entry_runs("cont", container) == ["run_a", "run_b"]


def test_runner_empty_pool_message(container):
    with pytest.raises(ValueError, match="min-trades"):
        run_combine("cont", ["run_a"], enabled_buckets={"normal"},
                    floors={"alpha": 99}, is_fraction=0.7, root=container)


def test_single_variant_pool(container):
    pool = _pool(container, buckets={"cpi"})      # only the cpi variant
    split_pool(pool, split_date_boundary(pool, 0.5))
    pool = [v for v in pool if v.n_is > 0]
    result = greedy_select(pool, max_k=10)
    assert len(result["path"]) == 1
