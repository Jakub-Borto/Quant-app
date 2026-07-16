"""Sweep-kind inference, min/max/step ranges, combo enumeration."""

import pytest

from modules.optimizer.backend.param_space import (
    BOOL_SWEEP_VALUES, build_range, combo_count, enumerate_combos,
    parse_flags, parse_values, sweep_kind,
)


# ── sweepability inferred from the PARAMS default's type ─────────────────────

@pytest.mark.parametrize("default,expected", [
    (30,        "int"),
    (0.5,       "float"),
    ("globex",  "categorical"),
    (True,      "bool"),        # bool is an int subclass — must NOT be int
    (False,     "bool"),
    ([1, 2],    None),
    (None,      None),
])
def test_sweep_kind(default, expected):
    assert sweep_kind(default) == expected


FLAG_NAMES = ["a", "b", "c", "d", "e", "f", "g"]


@pytest.mark.parametrize("default,options,expected", [
    ("globex", ["globex", "rth"],       "choice"),       # default in options
    (2, [2, 3],                         "choice"),       # typed (non-str) options
    ("1111100", FLAG_NAMES,             "flags"),        # bitstring, one bit per option
    ("zzz", ["a", "b"],                 "categorical"),  # fits neither rule -> plain str
    (True, [False, True],               "bool"),         # bool wins over options
    (1, [0, 1, "1111100"],              "choice"),       # int default in mixed options
])
def test_sweep_kind_with_options(default, options, expected):
    assert sweep_kind(default, options) == expected


def test_bool_sweep_values():
    assert BOOL_SWEEP_VALUES == [False, True]


# ── flags value lists ─────────────────────────────────────────────────────────

def test_parse_flags():
    assert parse_flags(" 1111100, 1010100 ,1111100 ", 7) == ["1111100", "1010100"]


@pytest.mark.parametrize("text", [
    "111110",           # too short
    "11111000",         # too long
    "1111100, 12345xx", # bad chars
    "",                 # empty
])
def test_parse_flags_errors(text):
    with pytest.raises(ValueError):
        parse_flags(text, 7)


# ── build_range: user-chosen min/max/step -> value list ──────────────────────

def test_build_range_int():
    assert build_range(4, 20, 2, "int") == [4, 6, 8, 10, 12, 14, 16, 18, 20]


def test_build_range_float_dust():
    assert build_range(0.6, 0.8, 0.05, "float") == [0.6, 0.65, 0.7, 0.75, 0.8]


def test_build_range_max_included():
    assert build_range(0.5, 2.0, 0.5, "float") == [0.5, 1.0, 1.5, 2.0]


def test_build_range_single_value():
    assert build_range(30, 30, 1, "int") == [30]


def test_build_range_max_not_on_grid():
    # max is a bound, not forced in: 4,7,10 stops below 12
    assert build_range(4, 12, 3, "int") == [4, 7, 10]


def test_build_range_errors():
    with pytest.raises(ValueError):
        build_range(4, 20, 0, "int")        # step must be > 0
    with pytest.raises(ValueError):
        build_range(20, 4, 2, "int")        # max < min


# ── categorical value lists ───────────────────────────────────────────────────

def test_parse_values():
    assert parse_values(" globex, rth ,globex ") == ["globex", "rth"]  # dedup


@pytest.mark.parametrize("text", ["", " , ,"])
def test_parse_values_empty(text):
    with pytest.raises(ValueError):
        parse_values(text)


# ── combo enumeration ─────────────────────────────────────────────────────────

def test_enumerate_combos_row_major():
    axes = [{"param": "a", "values": [1, 2]},
            {"param": "b", "values": ["x", "y", "z"]}]
    combos = enumerate_combos(axes)
    assert len(combos) == combo_count(axes) == 6
    assert combos[0] == {"a": 1, "b": "x"}
    assert combos[1] == {"a": 1, "b": "y"}      # first axis varies slowest
    assert combos[-1] == {"a": 2, "b": "z"}


def test_enumerate_combos_three_axes_product():
    axes = [{"param": "a", "values": [1, 2]},
            {"param": "b", "values": [1, 2, 3]},
            {"param": "c", "values": [1, 2]}]
    assert len(enumerate_combos(axes)) == 12


def test_enumerate_combos_duplicate_param():
    with pytest.raises(ValueError):
        enumerate_combos([{"param": "a", "values": [1]},
                          {"param": "a", "values": [2]}])


def test_enumerate_combos_empty_axis():
    with pytest.raises(ValueError):
        enumerate_combos([{"param": "a", "values": []}])


def test_combo_count_no_axes():
    assert combo_count([]) == 0
