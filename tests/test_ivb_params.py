"""ivb_model param contract after the PARAMS_OPTIONS migration: name-keyed
risk registry, zone_sl_risk folded into basic_risk, bool trailing switches,
bit-flag options mirroring FINDER_NAMES. Plus the vwap TP tick-rounding helper
and the per-script min-RR params. Qt-free."""

import math
from importlib import import_module

import numpy as np
import pytest

from modules.optimizer.backend.loader import load_strategy
from modules.optimizer.backend.param_space import is_flags, sweep_kind

from strategies.ivb_model.entries import FINDER_NAMES
from strategies.ivb_model.params import PARAMS, PARAM_SECTIONS, PARAMS_OPTIONS
from strategies.ivb_model.risk import RISK_REGISTRY

# risk/__init__ binds `vwap_tp_risk` etc. to each script's `run`, which shadows the submodule
# for both `from ... import x` AND `import a.b.x as y` — import_module goes to sys.modules
vwap_tp_mod    = import_module("strategies.ivb_model.risk.vwap_tp_risk")
vwap_trail_mod = import_module("strategies.ivb_model.risk.vwap_trailing_risk")


def test_risk_registry_matches_options():
    assert list(RISK_REGISTRY) == PARAMS_OPTIONS["risk_script"]
    assert "zone_sl_risk" not in RISK_REGISTRY
    assert PARAMS["risk_script"] in RISK_REGISTRY


def test_zone_sl_risk_gone():
    assert "zone_rr" not in PARAMS
    assert all("zone_rr" not in keys for keys in PARAM_SECTIONS.values())


def test_dropdown_defaults_are_members():
    for key in ("risk_script", "sl_type", "sl_placement",
                "vwap_session", "vwap_tp_mode"):
        assert PARAMS[key] in PARAMS_OPTIONS[key], key
        assert sweep_kind(PARAMS[key], PARAMS_OPTIONS[key]) == "choice", key


def test_flag_params_mirror_finder_names():
    for key in ("valid_entries", "trailing_entries"):
        assert PARAMS_OPTIONS[key] == list(FINDER_NAMES), key
        assert is_flags(PARAMS[key], PARAMS_OPTIONS[key]), key
        assert sweep_kind(PARAMS[key], PARAMS_OPTIONS[key]) == "flags", key


def test_trailing_switches_are_bools():
    assert PARAMS["trailing_in_profit"] is True
    assert PARAMS["late_trailing"] is False
    assert sweep_kind(PARAMS["trailing_in_profit"]) == "bool"


def test_min_rr_params_are_per_script():
    # each vwap script owns its own trio; neither reads the other's keys
    for switch, threshold, force, section in (
        ("is_over_rr",          "minimal_rr",          "force_trade",          "VWAP Risk"),
        ("trailing_is_over_rr", "trailing_minimal_rr", "trailing_force_trade", "VWAP Trailing Risk"),
    ):
        assert PARAMS[switch] is False
        assert PARAMS[force] is True     # default = the pre-min-RR behaviour (past-3σ trades 1:1)
        assert sweep_kind(PARAMS[switch]) == "bool"
        assert sweep_kind(PARAMS[force]) == "bool"
        assert isinstance(PARAMS[threshold], float)
        assert sweep_kind(PARAMS[threshold]) == "float"
        for key in (switch, threshold, force):
            assert key in PARAM_SECTIONS[section], key


@pytest.mark.parametrize("mod", [vwap_tp_mod, vwap_trail_mod])
def test_tick_tp_rounds_toward_entry(mod):
    # long targets floor to the grid, short targets ceil — never beyond the raw band
    assert mod._tick_tp(5312.37, "long",  0.25) == 5312.25
    assert mod._tick_tp(5312.37, "short", 0.25) == 5312.50
    # already on the grid => no-op in both directions (no float noise either)
    assert mod._tick_tp(5312.25, "long",  0.25) == 5312.25
    assert mod._tick_tp(5312.25, "short", 0.25) == 5312.25
    # non-0.25 grids
    assert mod._tick_tp(2011.07, "long",  0.10) == 2011.00
    assert mod._tick_tp(2011.07, "short", 0.10) == 2011.10
    # NaN passes through
    assert math.isnan(mod._tick_tp(float("nan"), "long", 0.25))


@pytest.mark.parametrize("mod", [vwap_tp_mod, vwap_trail_mod])
def test_tick_tp_array_matches_scalar(mod):
    band = np.array([5312.37, 5312.25, np.nan, 5300.01])
    for direction in ("long", "short"):
        out = mod._tick_tp_array(band, direction, 0.25)
        assert np.isnan(out[2])
        for i in (0, 1, 3):
            assert out[i] == mod._tick_tp(band[i], direction, 0.25)


def test_plugin_loader_exposes_params_options():
    # the exact path the UI uses (repo gotcha: plugins are exec'd, not imported)
    module = load_strategy("ivb_model")
    assert module.PARAMS_OPTIONS == PARAMS_OPTIONS
    assert callable(module.run)
