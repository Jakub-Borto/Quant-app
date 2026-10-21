"""
Reports — one implementation of the numbers.

For synthetic 1-, 2-, 3- and 4-parameter Optimizer runs: the report heatmap
of the FULL run and of its COMPACT form equal the Optimizer Explore grid for
every metric, every axis pair and every slice (bit for bit, NaN/inf
included), hatching uses the same rule, and Compress is safe (source
untouched, destination's other files untouched, cancel/failure leaves no
output, compact source refused, pickers limited to reports folders).
"""

import json

import numpy as np
import pandas as pd
import pytest

from _reports_util import (AXES_BY_N, every_layout, explore_arrays, tree_hashes,
                           write_run)
from modules.optimizer.backend import io as opt_io
from modules.optimizer.backend.heatmap_model import color_positions
from modules.optimizer.backend.metrics import METRIC_ORDER
from modules.reports.backend import compact
from modules.reports.backend.heatmap_view import (REPORT_SPLIT_DEFAULTS,
                                                  cell_records, compute_heatmap)


@pytest.fixture()
def reports(tmp_path):
    root = tmp_path / "reports"
    (root / "rep_a" / "data").mkdir(parents=True)
    (root / "rep_a" / "report.py").write_text("import streamlit\n")
    return root


@pytest.mark.parametrize("n_params", [1, 2, 3, 4])
def test_full_and_compact_equal_explore_everywhere(reports, n_params):
    run_dir = write_run(reports / "rep_a" / "data", n_params)
    trades, meta = opt_io.load_run(run_dir.name, root=run_dir.parent)
    out = reports / "rep_a" / "compact"
    out.mkdir()
    compact.compress_run(run_dir, out, [reports])

    full = compact.load_run(run_dir)
    small = compact.load_run(out / run_dir.name)
    assert not full.compact and small.compact
    checked = 0
    for x, y, sliders in every_layout(AXES_BY_N[n_params]):
        ref = explore_arrays(trades, meta, x, y, sliders)
        for run in (full, small):
            hd = compute_heatmap(run.summary, "total_ticks", x, y, sliders,
                                 min_trades=meta["min_trades_default"])
            for m in METRIC_ORDER:
                np.testing.assert_array_equal(hd.arrays[m], ref[m], err_msg=m)
            # hatching: HeatmapChart's rule on the reference counts
            np.testing.assert_array_equal(
                hd.masked, np.nan_to_num(ref["total_trades"], nan=0.0) < 5)
            checked += 1
    assert checked >= 2


def test_compact_meta_and_rows(reports):
    run_dir = write_run(reports / "rep_a" / "data", 3)
    res = compact.compress_run(run_dir, reports / "rep_a", [reports])
    target = res["target"]
    assert target == reports / "rep_a" / run_dir.name
    meta = json.loads((target / "meta.json").read_text())
    src_meta = json.loads((run_dir / "meta.json").read_text())
    block = meta.pop("compact")
    assert meta == src_meta                              # source meta unchanged
    assert block["format_version"] == 1
    assert block["source"] == "rep_a/data/" + run_dir.name
    assert block["n_combinations"] == 3 * 3 * 2 == res["n_combinations"]
    summary = pd.read_parquet(target / "summary.parquet")
    assert list(summary.columns) == ["stop", "target", "side"] + METRIC_ORDER
    assert len(summary) == 18                            # zero-trade combo included
    assert sorted(p.name for p in target.iterdir()) == ["meta.json", "summary.parquet"]


def test_compress_is_much_smaller_on_a_big_run(reports):
    """The output is a small fraction of the source (many trades per cell)."""
    from _reports_util import make_meta
    axes = AXES_BY_N[2]
    rng = np.random.default_rng(3)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=400)
    for s in axes[0]["values"]:
        for t in axes[1]["values"]:
            k = 4000
            rows.append(pd.DataFrame({"stop": s, "target": t,
                                      "pnl_ticks": rng.normal(0, 5, k),
                                      "date": rng.choice(dates, k),
                                      "entry_time": rng.integers(0, 10**12, k),
                                      "notes": ["{\"a\": 1}"] * k}))
    trades = pd.concat(rows, ignore_index=True)
    run_dir = opt_io.save_run(trades, make_meta(axes, trades), "big",
                              root=reports / "rep_a" / "data")
    res = compact.compress_run(run_dir, reports / "rep_a", [reports])
    assert res["result_bytes"] < res["source_bytes"] / 20


def test_source_unchanged_and_destination_untouched(reports):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    dest = reports / "rep_a"
    (dest / "notes.md").write_text("keep me")
    before_src = tree_hashes(run_dir)
    before_dest = {k: v for k, v in tree_hashes(dest).items()
                   if not k.startswith("data/")}
    compact.compress_run(run_dir, dest, [reports])
    assert tree_hashes(run_dir) == before_src
    after = tree_hashes(dest)
    for k, v in before_dest.items():
        assert after[k] == v
    new = {k for k in after if k not in before_dest and not k.startswith("data/")}
    assert new == {f"{run_dir.name}/meta.json", f"{run_dir.name}/summary.parquet"}


def test_cancel_leaves_no_output(reports):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    dest = reports / "rep_a" / "out"
    dest.mkdir()

    class Stop(Exception):
        pass

    def on_progress(cur, total, msg):
        if cur == 3:                                   # after writing, before rename
            raise Stop()

    with pytest.raises(Stop):
        compact.compress_run(run_dir, dest, [reports], on_progress=on_progress)
    assert list(dest.iterdir()) == []


def test_failure_leaves_no_output(reports, monkeypatch):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    dest = reports / "rep_a" / "out"
    dest.mkdir()
    monkeypatch.setattr(compact, "_verify_summary",
                        lambda *a: (_ for _ in ()).throw(compact.CompressError("boom")))
    with pytest.raises(compact.CompressError):
        compact.compress_run(run_dir, dest, [reports])
    assert list(dest.iterdir()) == []


def test_compact_source_refused_and_replace_asks(reports):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    res = compact.compress_run(run_dir, reports / "rep_a", [reports])
    with pytest.raises(compact.CompressError, match="already a compact run"):
        compact.compress_run(res["target"], reports / "rep_a" / "data", [reports])
    with pytest.raises(compact.TargetExists):
        compact.compress_run(run_dir, reports / "rep_a", [reports])
    (res["target"] / "summary.parquet").write_bytes(b"old")
    compact.compress_run(run_dir, reports / "rep_a", [reports], replace=True)
    assert len(pd.read_parquet(res["target"] / "summary.parquet")) == 12
    assert not [p for p in (reports / "rep_a").iterdir() if p.name.startswith(".")]


def test_paths_must_be_inside_reports_folders(reports, tmp_path):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(compact.CompressError, match="inside a reports folder"):
        compact.compress_run(run_dir, outside, [reports])
    stray = write_run(outside, 2)
    with pytest.raises(compact.CompressError, match="inside a reports folder"):
        compact.compress_run(stray, reports / "rep_a", [reports])
    (reports / "_archive").mkdir()
    with pytest.raises(compact.CompressError, match="_archive"):
        compact.compress_run(run_dir, reports / "_archive", [reports])
    with pytest.raises(compact.CompressError, match="inside it"):
        compact.compress_run(run_dir, run_dir, [reports])


def test_find_runs_skips_archive_and_hidden(reports):
    write_run(reports / "rep_a" / "data", 2)
    write_run(reports / "_archive", 2, name="hidden")
    write_run(reports / "rep_a" / ".tmp", 2, name="hidden2")
    found = compact.find_runs([reports])
    assert [(p.name, k) for p, k in found] == [("run_2p", "full")]


def test_load_run_errors_name_the_problem(tmp_path):
    with pytest.raises(compact.RunError, match="not found"):
        compact.load_run(tmp_path / "nope")
    (tmp_path / "x").mkdir()
    with pytest.raises(compact.RunError, match="not an Optimizer run"):
        compact.load_run(tmp_path / "x")
    run_dir = write_run(tmp_path, 2)
    t = pd.read_parquet(run_dir / "trades.parquet").drop(columns=["pnl_ticks"])
    t.to_parquet(run_dir / "trades.parquet")
    with pytest.raises(compact.RunError, match="pnl_ticks"):
        compact.load_run(run_dir)


def test_default_split_never_warms_bad_cells(reports):
    """By default avg trade <= 0, PF <= 1 and Sharpe <= 1 sit at or below the
    middle of the scale (the warm half starts above it) — the same
    colour_positions the Optimizer uses, with the report's split defaults."""
    run_dir = write_run(reports / "rep_a" / "data", 2, seed=7)
    run = compact.load_run(run_dir)
    x, y = AXES_BY_N[2]
    for metric, bound in (("avg_trade", 0.0), ("profit_factor", 1.0),
                          ("sharpe_daily", 1.0), ("sharpe_trade", 1.0),
                          ("total_ticks", 0.0)):
        assert REPORT_SPLIT_DEFAULTS[metric] == bound
        hd = compute_heatmap(run.summary, metric, x, y, {}, 5)
        z = hd.z
        finite = np.isfinite(z)
        assert (hd.norms[finite & (z < bound)] < 0.5).all()
        assert (hd.norms[finite & (z <= bound)] <= 0.5).all()
        # same colours as the Optimizer for the same split
        np.testing.assert_array_equal(
            hd.norms, color_positions(z, hd.zmin, hd.zmax, bound, 0.0, 0.0))


def test_colors_off_neutral_but_hatched(reports):
    run_dir = write_run(reports / "rep_a" / "data", 2)
    run = compact.load_run(run_dir)
    x, y = AXES_BY_N[2]
    hd = compute_heatmap(run.summary, "win_rate", x, y, {}, 5)
    on = cell_records(hd, colors_on=True)
    off = cell_records(hd, colors_on=False)
    assert len({r["fill"] for r in off}) == 1
    assert [r["masked"] for r in off] == [r["masked"] for r in on]
    assert any(r["masked"] for r in off)
    assert [r["text"] for r in off] == [r["text"] for r in on]
    # hover carries every metric + the cell's params
    assert {"stop", "target", "Total Ticks", "Sharpe (daily, zero-filled)",
            "Note"} <= set(on[0]["hover"])
