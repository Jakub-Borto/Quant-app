"""
Streamlit smoke test: views/optimizer.py render() runs without exceptions in
both modes. Needs the repo's data/ tree (skipped when absent); widget wiring
only, no backtests run.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _script():
    import streamlit as st
    if "page" not in st.session_state:
        st.session_state.page = "optimizer"
    from views import optimizer
    optimizer.render()


def _run(at: AppTest) -> AppTest:
    at = at.run(timeout=30)
    assert not at.exception, at.exception
    return at


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    if not (REPO_ROOT / "data" / "parquet").exists():
        pytest.skip("repo data/parquet not present")
    return AppTest.from_function(_script, default_timeout=30)


def test_new_run_mode_renders(app):
    at = _run(app)
    # mode radio + the setup selectors exist
    assert at.radio(key="opt_mode").value == "New Run"
    assert at.selectbox(key="opt_strategy").value is not None


def test_sweep_checkbox_shows_range_inputs_and_combo_readout(app):
    at = _run(app)
    options = at.selectbox(key="opt_strategy").options
    if "orb" not in options:
        pytest.skip("orb strategy not present")
    at.selectbox(key="opt_strategy").select("orb")
    at = _run(at)
    # check one sweepable param -> min/max/step inputs + combo readout appear
    at.checkbox(key="opt_orb_sweep_rr").check()
    at = _run(at)
    assert at.number_input(key="opt_orb_lo_rr").value == 2.0   # prefilled default
    at.number_input(key="opt_orb_hi_rr").set_value(4.0)
    at.number_input(key="opt_orb_step_rr").set_value(0.5)
    at = _run(at)
    assert any("5" in str(b.value) and "backtests" in str(b.value)
               for b in at.info)                               # 2.0..4.0 step 0.5


def test_string_param_sweeps_as_value_list(app):
    at = _run(app)
    options = at.selectbox(key="opt_strategy").options
    if "ivb_model_optimized" not in options:
        pytest.skip("ivb_model_optimized not present")
    at.selectbox(key="opt_strategy").select("ivb_model_optimized")
    at = _run(at)
    at.checkbox(key="opt_ivb_model_optimized_sweep_vwap_session").check()
    at = _run(at)
    field = at.text_input(key="opt_ivb_model_optimized_vals_vwap_session")
    assert field.value == "globex"                             # prefilled default
    field.set_value("globex, rth")
    at = _run(at)
    assert any("backtests" in str(b.value) for b in at.info)


def test_explore_mode_renders(app):
    at = _run(app)
    at.radio(key="opt_mode").set_value("Explore")
    at = _run(at)  # either run selector or the "no saved runs" notice — no crash


def test_run_grid_lands_in_explore_unsaved(app):
    """Regression: Run grid must not raise on the opt_mode hand-off
    (StreamlitAPIException) and must land in Explore WITHOUT writing to disk.
    (No further at.run() after the in-run st.rerun(): AppTest's element tree
    then contains stale setup widgets and crashes serializing them — an
    AppTest quirk, not an app behaviour.)"""
    import datetime as dt

    before = set((REPO_ROOT / "data" / "optimizations").glob("*"))

    at = _run(app)
    if "orb" not in at.selectbox(key="opt_strategy").options:
        pytest.skip("orb strategy not present")
    at.selectbox(key="opt_strategy").select("orb")
    at = _run(at)

    # narrow to the dataset's last available day -> the run is one quick day
    t = at.selectbox(key="opt_type").value
    a = at.selectbox(key=f"opt_asset_{t}").value
    d = at.selectbox(key=f"opt_dataset_{t}_{a}").value
    folder = REPO_ROOT / "data" / "parquet" / t / a / d
    last = sorted(f.stem for f in folder.glob("*.parquet")
                  if f.stem[0].isdigit())[-1]
    day = dt.date.fromisoformat(last)
    at.date_input(key=f"opt_start_{t}_{a}_{d}").set_value(day)
    at.date_input(key=f"opt_end_{t}_{a}_{d}").set_value(day)

    at.checkbox(key="opt_orb_sweep_rr").check()   # min=max=default -> 1 combo
    at = _run(at)
    at.button(key="opt_run_btn").click()
    at = _run(at)

    # lands in Explore with the run in memory only — nothing on disk
    assert at.radio(key="opt_mode").value == "Explore"
    assert at.selectbox(key="opt_run_select").value.startswith("●")
    assert any("Not saved yet" in str(s.value) for s in at.success)
    assert at.text_input(key="opt_save_name").value            # prefilled
    assert set((REPO_ROOT / "data" / "optimizations").glob("*")) == before


def _toy_unsaved_run():
    """Minimal in-memory run (trades + meta) as execute_grid_run leaves it."""
    import pandas as pd
    trades = pd.DataFrame({
        "rr":         [1.0, 1.0, 2.0, 2.0],
        "date":       pd.to_datetime(["2026-01-05", "2026-01-06"] * 2),
        "pnl_ticks":  [5.0, -3.0, 8.0, 1.0],
        "day_bucket": ["normal"] * 4,
    })
    meta = {
        "strategy": "orb", "dataset": "Futures/ES/ES_test", "ticker": "ES",
        "tick_size": 0.25, "ticks_per_point": 4,
        "start_date": "2026-01-05", "end_date": "2026-01-06",
        "axes": {"x": {"param": "rr", "values": [1.0, 2.0]},
                 "y": None, "slider": None, "slider2": None},
        "fixed_params": {}, "min_trades_default": 0, "be_band_ticks": 0.0,
        "ff_events_found": True, "split_date": "2026-01-05",
        "n_combos": 2, "n_trades": 4, "created_at": "2026-01-07T00:00:00",
    }
    return trades, meta


def test_save_panel_new_folder(monkeypatch):
    """The optional save panel persists an unsaved run into a user-named
    (new) folder — Data Formatter logic."""
    import shutil

    monkeypatch.chdir(REPO_ROOT)
    if not (REPO_ROOT / "data" / "parquet").exists():
        pytest.skip("repo data/parquet not present")

    def _cleanup():
        for pattern in ("pytest_folder*", "pytest_smoke_run*"):
            for d in (REPO_ROOT / "data" / "optimizations").glob(pattern):
                shutil.rmtree(d)

    _cleanup()
    try:
        trades, meta = _toy_unsaved_run()
        at = AppTest.from_function(_script, default_timeout=30)
        at.session_state["opt_mode"]    = "Explore"
        at.session_state["opt_unsaved"] = True
        at.session_state["opt_trades"]  = trades
        at.session_state["opt_meta"]    = meta
        at = _run(at)

        assert at.selectbox(key="opt_run_select").value.startswith("●")
        at.selectbox(key="opt_save_folder").select("── New folder ──")
        at = _run(at)
        at.text_input(key="opt_save_new_folder").set_value("pytest_folder")
        at.text_input(key="opt_save_name").set_value("pytest_smoke_run")
        at.button(key="opt_save_btn").click()
        at = _run(at)

        saved = (REPO_ROOT / "data" / "optimizations" / "pytest_folder"
                 / "pytest_smoke_run")
        assert (saved / "trades.parquet").exists()
        assert (saved / "meta.json").exists()
        assert any("Saved to" in str(s.value) for s in at.success)
        # cascading selector points at the saved run: folder, then run
        assert at.selectbox(key="opt_explore_folder").value == "pytest_folder"
        assert at.selectbox(key="opt_run_select").value == "pytest_smoke_run"
    finally:
        _cleanup()
