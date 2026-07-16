"""ivb_model param contract after the PARAMS_OPTIONS migration: name-keyed
risk registry, zone_sl_risk folded into basic_risk, bool trailing switches,
bit-flag options mirroring FINDER_NAMES. Qt-free."""

from modules.optimizer.backend.loader import load_strategy
from modules.optimizer.backend.param_space import is_flags, sweep_kind

from strategies.ivb_model.entries import FINDER_NAMES
from strategies.ivb_model.params import PARAMS, PARAM_SECTIONS, PARAMS_OPTIONS
from strategies.ivb_model.risk import RISK_REGISTRY


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


def test_plugin_loader_exposes_params_options():
    # the exact path the UI uses (repo gotcha: plugins are exec'd, not imported)
    module = load_strategy("ivb_model")
    assert module.PARAMS_OPTIONS == PARAMS_OPTIONS
    assert callable(module.run)
