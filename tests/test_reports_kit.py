"""
Reports kit — the heatmap component in a real Streamlit script run
(streamlit.testing AppTest) + the saved-view file layer + the launch path.

Covers: independent components, saved views surviving a restart, identity by
key (inserting a component above changes nothing), lock (all controls
disabled, persists, per component), stale views (only the stale field
resets, with a notice), corrupt state file (defaults + one warning, kept as
.bad), bad paths (error box, rest of the page renders), duplicate keys, and
that a report launched through the bootstrap imports the kit, the app's
backend and the REAL stdlib inspect.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from _reports_util import write_run  # noqa: E402
from modules.reports.kit import state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _make_report(tmp_path, body: str) -> Path:
    d = tmp_path / "rep"
    (d / "data").mkdir(parents=True, exist_ok=True)
    if not (d / "data" / "run_2p").exists():
        write_run(d / "data", 2)
        write_run(d / "data", 3)
    (d / "report.py").write_text(
        "import streamlit as st\nfrom modules.reports.kit import heatmap\n" + body,
        encoding="utf-8")
    return d


TWO = """
heatmap(key="a", runs={"two": "data/run_2p", "three": "data/run_3p"})
heatmap(key="b", runs={"three": "data/run_3p"}, default_metric="sharpe_daily")
st.write("END")
"""


def _run(d: Path) -> AppTest:
    at = AppTest.from_file(str(d / "report.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _sel(at, key, field):
    return at.selectbox(key=f"rk:{key}:{field}")


def test_two_components_independent_and_saved(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    assert _sel(at, "a", "metric").value == "total_ticks"
    assert _sel(at, "b", "metric").value == "sharpe_daily"

    _sel(at, "a", "metric").select("profit_factor").run()
    _sel(at, "a", "run").select("three").run()
    _sel(at, "a", "slice:side").select("short").run()
    at.slider(key="rk:a:good").set_value(1.25).run()
    at.number_input(key="rk:a:min_trades").set_value(9).run()
    at.toggle(key="rk:a:colors").set_value(False).run()
    assert not at.exception
    assert _sel(at, "b", "metric").value == "sharpe_daily"   # untouched
    assert at.markdown[-1].value == "END"

    saved = state.read_components(d)
    assert saved["a"]["metric"] == "profit_factor"
    assert saved["a"]["slices"] == {"side": "short"}
    assert "b" not in saved                                   # never changed

    # a restart = a fresh session reading the file
    at2 = _run(d)
    assert _sel(at2, "a", "run").value == "three"
    assert _sel(at2, "a", "metric").value == "profit_factor"
    assert _sel(at2, "a", "slice:side").value == "short"
    assert at2.slider(key="rk:a:good").value == 1.25
    assert at2.number_input(key="rk:a:min_trades").value == 9
    assert at2.toggle(key="rk:a:colors").value is False


def test_axis_swap_keeps_x_ne_y(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    assert (_sel(at, "a", "x").value, _sel(at, "a", "y").value) == ("stop", "target")
    _sel(at, "a", "x").select("target").run()
    assert (_sel(at, "a", "x").value, _sel(at, "a", "y").value) == ("target", "stop")


def test_inserting_a_component_above_keeps_views(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    _sel(at, "b", "metric").select("win_rate").run()
    _make_report(tmp_path, 'heatmap(key="new", runs={"two": "data/run_2p"})\n' + TWO)
    at2 = _run(d)
    assert _sel(at2, "b", "metric").value == "win_rate"
    assert _sel(at2, "new", "metric").value == "total_ticks"


def test_lock_disables_and_persists_per_component(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    _sel(at, "a", "metric").select("avg_trade").run()
    at.toggle(key="rk:a:locked").set_value(True).run()
    assert _sel(at, "a", "metric").disabled and _sel(at, "a", "run").disabled
    assert at.slider(key="rk:a:good").disabled
    assert not _sel(at, "b", "metric").disabled
    assert state.read_components(d)["a"]["locked"] is True

    at2 = _run(d)
    for f in ("run", "metric", "x", "y"):
        assert _sel(at2, "a", f).disabled
    assert at2.number_input(key="rk:a:min_trades").disabled
    assert at2.toggle(key="rk:a:colors").disabled
    assert _sel(at2, "a", "metric").value == "avg_trade"
    assert not _sel(at2, "b", "metric").disabled

    at2.toggle(key="rk:a:locked").set_value(False).run()
    assert not _sel(at2, "a", "metric").disabled
    assert state.read_components(d)["a"]["locked"] is False


def test_stale_label_resets_only_that_field(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    _sel(at, "b", "metric").select("win_rate").run()
    _sel(at, "b", "x").select("side").run()
    state.write_entry(d, "b", {**state.read_components(d)["b"], "run": "gone"})
    at2 = _run(d)
    assert _sel(at2, "b", "run").value == "three"           # reset to default
    assert _sel(at2, "b", "metric").value == "win_rate"     # kept
    assert _sel(at2, "b", "x").value == "side"              # kept
    assert any("no longer valid" in c.value for c in at2.caption)


def test_stale_view_on_locked_component_not_written(tmp_path):
    d = _make_report(tmp_path, TWO)
    at = _run(d)
    at.toggle(key="rk:b:locked").set_value(True).run()
    entry = {**state.read_components(d)["b"], "metric": "renamed_metric"}
    state.write_entry(d, "b", entry)
    at2 = _run(d)
    assert _sel(at2, "b", "metric").value == "sharpe_daily"  # default shown
    assert any("no longer valid" in c.value for c in at2.caption)
    assert state.read_components(d)["b"]["metric"] == "renamed_metric"


def test_corrupt_state_file_defaults_warning_and_bad_copy(tmp_path):
    from modules.reports.backend.bootstrap import install_script_start_hook
    install_script_start_hook()
    d = _make_report(tmp_path, TWO)
    (d / ".report_state.json").write_text("{not json", encoding="utf-8")
    at = _run(d)
    assert (d / ".report_state.json.bad").read_text() == "{not json"
    assert len(at.warning) == 1 and "Saved views could not be read" in at.warning[0].value
    assert _sel(at, "a", "metric").value == "total_ticks"
    _sel(at, "a", "metric").select("win_rate").run()
    assert len(at.warning) == 1                              # still exactly one
    assert json.loads((d / ".report_state.json").read_text())["components"]["a"]


def test_bad_path_shows_error_and_page_continues(tmp_path):
    d = _make_report(tmp_path, """
heatmap(key="bad", runs={"missing": "data/nope"})
heatmap(key="ok", runs={"two": "data/run_2p"})
st.write("END")
""")
    at = _run(d)
    assert any("missing" in e.value and "nope" in e.value for e in at.error)
    assert _sel(at, "ok", "metric").value == "total_ticks"
    assert at.markdown[-1].value == "END"


def test_duplicate_key_is_an_error(tmp_path):
    d = _make_report(tmp_path, """
heatmap(key="same", runs={"two": "data/run_2p"})
heatmap(key="same", runs={"two": "data/run_2p"})
""")
    at = _run(d)
    assert any("duplicate key" in e.value for e in at.error)


def test_compact_run_renders_with_compact_caption(tmp_path):
    from modules.reports.backend.compact import compress_run
    d = _make_report(tmp_path, 'heatmap(key="c", runs={"c": "compact/run_2p"})\n')
    compress_run(d / "data" / "run_2p", d / "compact", [tmp_path])
    at = _run(d)
    assert any("compact" in c.value and "pre-cost" in c.value for c in at.caption)


def test_state_file_atomic_and_keeps_unknown_keys(tmp_path):
    state.write_entry(tmp_path, "old_component", {"metric": "x"})
    state.write_entry(tmp_path, "k", {"metric": "y"})
    comps = state.read_components(tmp_path)
    assert comps == {"old_component": {"metric": "x"}, "k": {"metric": "y"}}
    assert [p.name for p in tmp_path.iterdir()] == [".report_state.json"]


def test_launched_report_imports_kit_backend_and_real_inspect(tmp_path):
    """Through the real bootstrap (the launch path) in a separate process,
    cwd = the report folder: the kit and the pure backend import, and
    `inspect` is the standard library even with an `inspect.py` at the repo
    root (the shadowing trap PYTHONPATH would fall into)."""
    d = tmp_path / "rep"
    d.mkdir()
    (d / "helper.py").write_text("VALUE = 42\n")
    (d / "report.py").write_text(
        "import inspect, os, sys\n"
        "import streamlit as st\n"
        "from modules.reports.kit import heatmap, style\n"
        "import modules.optimizer.backend.metrics\n"
        "import helper\n"
        "std = os.path.dirname(os.__file__)\n"
        "assert os.path.dirname(inspect.__file__) == std, inspect.__file__\n"
        "assert helper.VALUE == 42\n"
        "assert 'PySide6' not in sys.modules\n"
        "st.write('ok')\n", encoding="utf-8")
    shadow = REPO / "inspect.py"
    created = not shadow.exists()
    if created:
        shadow.write_text("raise ImportError('shadowed stdlib inspect')\n")
    try:
        from modules.reports.backend.launch import BOOTSTRAP
        res = subprocess.run([sys.executable, "-u", str(BOOTSTRAP), "--apptest",
                              str(d / "report.py")], cwd=d, capture_output=True,
                             text=True, timeout=300,
                             env={**os.environ, "PYTHONPATH": ""})
    finally:
        if created:
            shadow.unlink()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "APPTEST_OK" in res.stdout


def test_launch_args_carry_theme_and_bootstrap(tmp_path):
    from modules.common.backend import palette
    from modules.reports.backend.launch import BOOTSTRAP, launch_args
    args = launch_args(tmp_path / "report.py", 8765)
    assert args[:3] == ["-u", str(BOOTSTRAP), str(tmp_path / "report.py")]
    flags = dict(zip(args[3::2], args[4::2]))
    assert flags["--server.port"] == "8765"
    assert flags["--theme.base"] == "dark"
    assert flags["--theme.backgroundColor"] == palette.BG
    assert flags["--server.runOnSave"] == "true"
