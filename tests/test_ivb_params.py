"""ivb_model param contract: core params + params OWNED by each entry finder and
each risk script (prefixed, never shared), assembled into PARAMS /
PARAM_SECTIONS / PARAMS_OPTIONS. Plus the vwap TP tick-rounding helper. Qt-free."""

import math
from importlib import import_module

import numpy as np
import pytest

from modules.optimizer.backend.loader import load_strategy
from modules.optimizer.backend.param_space import is_flags, sweep_kind

from strategies.ivb_model.entries import DEFAULT_FLAGS, FINDER_MODULES, FINDER_NAMES
from strategies.ivb_model.params import (CORE_PARAMS, PARAM_SECTIONS, PARAMS,
                                         PARAMS_OPTIONS)
from strategies.ivb_model.risk import RISK_REGISTRY, RISK_SCRIPTS

vwap_tp_mod    = import_module("strategies.ivb_model.risk.vwap_tp_risk")
vwap_trail_mod = import_module("strategies.ivb_model.risk.vwap_trailing_risk")


def test_risk_registry_matches_options():
    assert list(RISK_REGISTRY) == PARAMS_OPTIONS["risk_script"] == list(RISK_SCRIPTS)
    assert PARAMS["risk_script"] in RISK_REGISTRY
    assert PARAMS["risk_script"] == "vwap_trailing_risk"


def test_every_param_has_exactly_one_owner_and_one_section():
    owners = [CORE_PARAMS] + [m.PARAMS for m in FINDER_MODULES] \
        + [m.PARAMS for m in RISK_SCRIPTS.values()]
    keys = [k for d in owners for k in d]
    assert len(keys) == len(set(keys)), "a param name is declared twice"
    assert set(keys) == set(PARAMS)
    in_sections = [k for ks in PARAM_SECTIONS.values() for k in ks]
    assert sorted(in_sections) == sorted(PARAMS), "every param in exactly one section"


def test_risk_scripts_share_no_params():
    """Deleting one risk script must leave the others working: each reads only
    its own prefixed keys (plus core/finder params)."""
    prefixes = {"basic_risk": "basic_", "vwap_tp_risk": "vwap_tp_",
                "vwap_trailing_risk": "trail_"}
    for name, mod in RISK_SCRIPTS.items():
        assert all(k.startswith(prefixes[name]) for k in mod.PARAMS), name
        assert PARAM_SECTIONS[mod.SECTION] == list(mod.PARAMS)
    for gone in ("rr", "sl_type", "sl_placement", "vwap_std", "vwap_session",
                 "is_over_rr", "minimal_rr", "force_trade", "trade_timeout",
                 "trailing_entries", "indicators_folder", "big_trades_folder"):
        assert gone not in PARAMS, gone


def test_default_values_unchanged():
    # the migration renamed params but kept every default
    expected = {
        "basic_rr": 1.0, "basic_sl_type": "VAL/VAH", "basic_trade_timeout": 999,
        "vwap_tp_sl_placement": "zone_logic", "vwap_tp_std": 2, "vwap_tp_session": "globex",
        "vwap_tp_mode": "now", "vwap_tp_is_over_rr": False, "vwap_tp_minimal_rr": 2.0,
        "vwap_tp_force_trade": True, "vwap_tp_trade_timeout": 999,
        "trail_sl_placement": "zone_logic", "trail_vwap_std": 2, "trail_vwap_session": "globex",
        "trail_tp_mode": "now", "trail_trade_timeout": 999, "trail_entries": "1111100",
        "trail_in_profit": True, "trail_late": False, "trail_is_over_rr": False,
        "trail_minimal_rr": 2.0, "trail_force_trade": True,
        "valid_entries": "1111100", "session_start": "09:30", "ib_minutes": 30,
        "tick_size": 0.25,
    }
    for k, v in expected.items():
        assert PARAMS[k] == v and type(PARAMS[k]) is type(v), k


def test_dropdown_defaults_are_members():
    for key in ("risk_script", "basic_sl_type", "vwap_tp_sl_placement", "vwap_tp_session",
                "vwap_tp_mode", "trail_sl_placement", "trail_vwap_session", "trail_tp_mode"):
        assert PARAMS[key] in PARAMS_OPTIONS[key], key
        assert sweep_kind(PARAMS[key], PARAMS_OPTIONS[key]) == "choice", key


def test_flag_params_mirror_finder_names():
    assert DEFAULT_FLAGS == "1111100"
    for key in ("valid_entries", "trail_entries"):
        assert PARAMS_OPTIONS[key] == list(FINDER_NAMES), key
        assert is_flags(PARAMS[key], PARAMS_OPTIONS[key]), key
        assert sweep_kind(PARAMS[key], PARAMS_OPTIONS[key]) == "flags", key


def test_finders_declare_themselves():
    for m in FINDER_MODULES:
        assert isinstance(m.SECTION, str) and m.SECTION in PARAM_SECTIONS
        assert isinstance(m.DEFAULT_ON, bool)
        assert set(m.BASELINES) <= {"rolling", "passive", "cvd"}
        assert callable(m.find_entry)


def test_trailing_switches_are_bools():
    assert PARAMS["trail_in_profit"] is True
    assert PARAMS["trail_late"] is False
    assert sweep_kind(PARAMS["trail_in_profit"]) == "bool"


def test_vwap_band_hook():
    assert RISK_SCRIPTS["basic_risk"].NEEDS_VWAP_BANDS is False
    assert RISK_SCRIPTS["vwap_tp_risk"].NEEDS_VWAP_BANDS is True
    assert RISK_SCRIPTS["vwap_trailing_risk"].NEEDS_VWAP_BANDS is True
    trail = RISK_SCRIPTS["vwap_trailing_risk"]
    assert trail.extra_active_finders({"trail_entries": "101"}, 7) == "1010000"
    assert not hasattr(RISK_SCRIPTS["vwap_tp_risk"], "extra_active_finders")


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


def test_plugin_loader_exposes_the_engine_contract():
    # the exact path the UI uses (repo gotcha: plugins are exec'd, not imported)
    module = load_strategy("ivb_model")
    assert module.PARAMS_OPTIONS == PARAMS_OPTIONS
    assert callable(module.process_day) and callable(module.prepare_day)
    assert set(module.DATA) == {"main", "indicators"}
    assert module.PREPARE_PARAMS == ["session_start"]
