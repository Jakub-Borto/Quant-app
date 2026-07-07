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


def test_run_grid_switches_to_explore(app):
    """Regression: clicking Run grid must not raise on the opt_mode hand-off
    (StreamlitAPIException: widget state set after instantiation)."""
    import datetime as dt
    import shutil

    def _cleanup():
        for d in (REPO_ROOT / "data" / "optimizations").glob("pytest_smoke_run*"):
            shutil.rmtree(d)

    _cleanup()
    try:
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
        at.text_input(key="opt_run_name").set_value("pytest_smoke_run")
        at.button(key="opt_run_btn").click()
        at = _run(at)

        assert at.radio(key="opt_mode").value == "Explore"
        assert any("Saved to" in str(s.value) for s in at.success)
    finally:
        _cleanup()
