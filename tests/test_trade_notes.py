"""
The pure trade-notes model: flattening, type sniffing, every operator, and
the grouped AND/OR evaluator.

The invariants worth defending here are the ones that came from real data —
absorption_time being str on some trades and list[str] on others, sparse keys
that only exist on 1 trade in 156, JSON nulls, and strategies that emit no
notes column at all.
"""

import json

import numpy as np
import pandas as pd
import pytest

from modules.common.backend.trade_notes import (KIND_BOOL, KIND_DATETIME,
                                                KIND_EMPTY, KIND_LIST,
                                                KIND_NUMERIC, KIND_TEXT,
                                                KIND_TIME, OPS, ColumnSource,
                                                Condition, Group, Query,
                                                column_kind, display_text,
                                                evaluate, flatten_notes,
                                                infer_kind, notes_fingerprint,
                                                operators_for)


def _trades(notes, **columns) -> pd.DataFrame:
    """A minimal trades frame whose notes column holds `notes` (dicts are
    json.dumps'd, anything else is passed through verbatim)."""
    rows = [json.dumps(n) if isinstance(n, dict) else n for n in notes]
    data = {"ticks": list(range(len(rows))), "notes": rows}
    data.update(columns)
    return pd.DataFrame(data)


def _source(notes, **columns):
    frame = _trades(notes, **columns)
    return ColumnSource(frame, flatten_notes(frame))


def _mask(query, columns):
    mask, _warnings = evaluate(query, columns)
    return mask.tolist()


def _one(column, op, value=None, join="and"):
    return Query((Group((Condition(column, op, value),), join),), join)


# ══ flattening ═══════════════════════════════════════════════════════════════
def test_flatten_without_notes_column_is_empty_not_an_error():
    flat = flatten_notes(pd.DataFrame({"ticks": [1, 2]}))
    assert flat.notes_columns == []
    assert flat.frame.empty or list(flat.frame.columns) == []


def test_flatten_of_none_and_of_an_empty_frame():
    assert flatten_notes(None).notes_columns == []
    assert flatten_notes(_trades([])).notes_columns == []


def test_flatten_sparse_keys_keeps_first_appearance_order():
    flat = flatten_notes(_trades([{"b": 1, "a": 2}, {"c": 3}]))
    assert flat.notes_columns == ["notes.b", "notes.a", "notes.c"]
    # pandas narrows an all-numeric column to float64; the absent row is NaN
    assert flat.frame["notes.c"].isna().tolist() == [True, False]


def test_flatten_tolerates_every_shape_of_broken_notes():
    flat = flatten_notes(_trades([None, "", "not json", "[1, 2]", "7",
                                  {"a": 1}]))
    assert flat.notes_columns == ["notes.a"]
    # only the three genuinely broken payloads are reported
    assert "3 trade(s)" in flat.warnings[0]


def test_flatten_preserves_a_non_range_index():
    frame = _trades([{"a": 1}, {"a": 2}])
    frame.index = pd.Index(["x", "y"])
    assert flatten_notes(frame).frame.index.equals(frame.index)


def test_flatten_keeps_lists_intact_rather_than_exploding_them():
    flat = flatten_notes(_trades([{"absorption_time": ["10:53", "10:54"]}]))
    assert flat.frame["notes.absorption_time"].iloc[0] == ["10:53", "10:54"]
    assert "notes.absorption_time.0" not in flat.frame.columns


def test_flatten_canonicalises_times_but_leaves_dates_alone():
    flat = flatten_notes(_trades([{"t": "2026-01-05T10:22:00", "d": "2026-01-05"}]))
    assert flat.frame["notes.t"].iloc[0] == "10:22"
    assert flat.frame["notes.d"].iloc[0] == "2026-01-05"


def test_flatten_labels_fall_back_to_the_full_id_on_a_collision():
    flat = flatten_notes(_trades([{"ticks": 5, "poc": 1.0}]))
    assert flat.labels["notes.ticks"] == "notes.ticks"
    assert flat.labels["notes.poc"] == "poc"


def test_fingerprint_is_content_based():
    frame = _trades([{"a": 1}, {"a": 2}])
    assert notes_fingerprint(frame) == notes_fingerprint(frame.copy())
    # what a regime join does: same notes, extra columns, new object
    annotated = frame.copy()
    annotated["regime"] = ["trend", "chop"]
    assert notes_fingerprint(annotated) == notes_fingerprint(frame)
    changed = _trades([{"a": 1}, {"a": 3}])
    assert notes_fingerprint(changed) != notes_fingerprint(frame)
    assert notes_fingerprint(None) == (0, None)


# ══ type inference ═══════════════════════════════════════════════════════════
@pytest.mark.parametrize("values, kind", [
    ([None, None], KIND_EMPTY),
    ([1, 2.5, None], KIND_NUMERIC),
    ([True, False, None], KIND_BOOL),          # NOT numeric — bool is an int
    (["10:22", "9:05"], KIND_TIME),
    (["tp_vwap_2", "1:1"], KIND_TEXT),         # "1:1" is a time, "tp_..." isn't
    (["a", ["b", "c"]], KIND_LIST),            # the real absorption_time case
    ([["x"], None], KIND_LIST),
])
def test_infer_kind(values, kind):
    assert infer_kind(pd.Series(values, dtype=object))[0] == kind


def test_infer_kind_reports_numeric_and_time_capability_of_text_columns():
    kind, numeric_ok, time_ok = infer_kind(pd.Series(["1", "2"], dtype=object))
    assert (kind, numeric_ok, time_ok) == (KIND_TEXT, True, False)
    kind, numeric_ok, time_ok = infer_kind(
        pd.Series(["10:22", "n/a"], dtype=object))
    assert (kind, numeric_ok, time_ok) == (KIND_TEXT, False, False)


def test_infer_kind_rejects_impossible_clock_values():
    assert infer_kind(pd.Series(["10:22", "99:99"], dtype=object))[0] == KIND_TEXT


def test_column_kind_uses_dtype_for_real_trade_columns():
    assert column_kind(pd.Series([1, 2]))[0] == KIND_NUMERIC
    assert column_kind(pd.Series([True, False]))[0] == KIND_BOOL
    assert column_kind(pd.to_datetime(["2026-01-05 10:22"]))[0] == KIND_DATETIME


def test_operators_offered_per_kind():
    assert operators_for(KIND_NUMERIC)[:2] == ["gt", "ge"]
    assert operators_for(KIND_BOOL) == ["is_true", "is_false", "is_empty",
                                        "not_empty"]
    assert "one_of" in operators_for(KIND_LIST)
    assert "count_gt" in operators_for(KIND_LIST)
    assert "gt" in operators_for(KIND_TEXT, numeric_ok=True)
    assert "gt" not in operators_for(KIND_TEXT)
    assert "time_gt" in operators_for(KIND_TEXT, time_ok=True)
    assert operators_for(KIND_EMPTY) == ["is_empty", "not_empty"]
    # every advertised operator has a spec
    for kind in (KIND_NUMERIC, KIND_TIME, KIND_BOOL, KIND_TEXT, KIND_LIST,
                 KIND_DATETIME, KIND_EMPTY):
        assert all(op in OPS for op in operators_for(kind, True, True))


# ══ display ══════════════════════════════════════════════════════════════════
def test_display_text_joins_lists_and_blanks_nulls():
    assert display_text(["10:53", "10:54"]) == "10:53, 10:54"
    assert display_text(None) == ""
    assert display_text(float("nan")) == ""
    assert display_text(float("inf")) == "∞"
    assert display_text(5504.75) == "5504.75"


# ══ operators ════════════════════════════════════════════════════════════════
def test_numeric_operators_never_match_a_missing_value():
    columns = _source([{"v": 5}, {"v": 15}, {}])
    assert _mask(_one("notes.v", "gt", "10"), columns) == [False, True, False]
    assert _mask(_one("notes.v", "le", "5"), columns) == [True, False, False]
    assert _mask(_one("notes.v", "between", ("20", "0")), columns) == \
        [True, True, False]                       # bounds auto-swapped
    # the sharp one: NaN != 5 is True in pandas, and must not be here
    assert _mask(_one("notes.v", "ne", "5"), columns) == [False, True, False]


def test_time_operators_compare_as_time_of_day_not_as_text():
    columns = _source([{"t": "9:05"}, {"t": "10:00"}])
    assert "9:05" > "10:00"                       # plain string comparison lies
    assert _mask(_one("notes.t", "time_lt", "10:00"), columns) == [True, False]
    assert _mask(_one("notes.t", "time_ge", "9:05"), columns) == [True, True]
    assert _mask(_one("notes.t", "time_between", ("09:00", "09:30")),
                 columns) == [True, False]


def test_time_operators_work_on_a_real_datetime_column():
    frame = _trades([{}, {}],
                    entry_time=pd.to_datetime(["2026-01-05 09:31",
                                               "2026-01-05 14:05"]))
    columns = ColumnSource(frame, flatten_notes(frame))
    assert _mask(_one("entry_time", "time_gt", "10:30"), columns) == \
        [False, True]


def test_bool_operators_do_not_treat_one_as_true():
    columns = _source([{"e": True}, {"e": False}, {"e": 1}, {}])
    assert _mask(_one("notes.e", "is_true"), columns) == \
        [True, False, False, False]
    assert _mask(_one("notes.e", "is_false"), columns) == \
        [False, True, False, False]


def test_text_operators():
    columns = _source([{"t": "tp_vwap_2"}, {"t": "1:1"}, {}])
    assert _mask(_one("notes.t", "one_of", ("tp_vwap_2",)), columns) == \
        [True, False, False]
    assert _mask(_one("notes.t", "contains", "VWAP"), columns) == \
        [True, False, False]
    assert _mask(_one("notes.t", "matches", r"^\d+:\d+$"), columns) == \
        [False, True, False]


def test_an_invalid_regex_fails_closed_with_a_warning():
    columns = _source([{"t": "a"}])
    mask, warnings = evaluate(_one("notes.t", "matches", "([unclosed"), columns)
    assert mask.tolist() == [False]
    assert warnings and "regex" in warnings[0]


def test_list_operators_including_the_str_or_list_collision():
    columns = _source([{"a": ["10:53", "10:54"]}, {"a": "10:53"}, {}])
    assert _mask(_one("notes.a", "count_gt", "1"), columns) == \
        [True, False, False]
    assert _mask(_one("notes.a", "has_item", "10:54"), columns) == \
        [True, False, False]
    assert _mask(_one("notes.a", "has_item", "10:53"), columns) == \
        [True, True, False]
    # a missing key flattens to [], so "0 items" finds it
    assert _mask(_one("notes.a", "count_eq", "0"), columns) == \
        [False, False, True]
    # text operators run on what the cell displays
    assert _mask(_one("notes.a", "contains", "10:54"), columns) == \
        [True, False, False]


def test_is_empty_covers_nulls_blanks_and_empty_lists():
    columns = _source([{"v": None}, {"v": ""}, {"v": "   "}, {"v": []},
                       {"v": 0}, {}])
    assert _mask(_one("notes.v", "is_empty"), columns) == \
        [True, True, True, True, False, True]
    assert _mask(_one("notes.v", "not_empty"), columns) == \
        [False, False, False, False, True, False]


def test_a_json_null_is_empty_but_never_matches_a_comparison():
    columns = _source([{"vwap_at_entry": None}, {"vwap_at_entry": 18141.9}])
    assert _mask(_one("notes.vwap_at_entry", "is_empty"), columns) == \
        [True, False]
    assert _mask(_one("notes.vwap_at_entry", "gt", "0"), columns) == \
        [False, True]


def test_an_unparseable_operand_drops_the_condition():
    columns = _source([{"v": 1}, {"v": 2}])
    mask, warnings = evaluate(_one("notes.v", "gt", "abc"), columns)
    assert mask.tolist() == [True, True]          # identity, not all-False
    assert warnings


# ══ the evaluator ════════════════════════════════════════════════════════════
def _columns_for_groups():
    return _source([{"flip": 0, "tp": "a"}, {"flip": 3, "tp": "a"},
                    {"flip": 3, "tp": "b"}, {"flip": 0, "tp": "b"}])


def test_group_join_and_top_level_join():
    high = Condition("notes.flip", "gt", "1")
    is_a = Condition("notes.tp", "one_of", ("a",))
    columns = _columns_for_groups()

    assert _mask(Query((Group((high, is_a), "and"),)), columns) == \
        [False, True, False, False]
    assert _mask(Query((Group((high, is_a), "or"),)), columns) == \
        [True, True, True, False]
    assert _mask(Query((Group((high,)), Group((is_a,))), "or"), columns) == \
        [True, True, True, False]
    assert _mask(Query((Group((high,)), Group((is_a,))), "and"), columns) == \
        [False, True, False, False]


@pytest.mark.parametrize("query", [
    None,
    Query(),
    Query((Group(()),)),
    Query((Group(()), Group(())), "or"),
    Query((Group((Condition("", "", None),)),)),
])
def test_an_empty_query_matches_everything(query):
    columns = _columns_for_groups()
    assert _mask(query, columns) == [True] * 4


def test_an_empty_group_is_skipped_rather_than_matching_everything_under_or():
    columns = _columns_for_groups()
    query = Query((Group((Condition("notes.flip", "gt", "1"),)), Group(())), "or")
    assert _mask(query, columns) == [False, True, True, False]


def test_an_unknown_column_is_dropped_not_failed_closed():
    """The Optimizer re-slice case: a cell whose trades never set trail5_stop
    must not silently empty the whole report."""
    columns = _columns_for_groups()
    query = Query((Group((Condition("notes.trail5_stop", "gt", "1"),
                          Condition("notes.flip", "gt", "1")), "and"),))
    mask, warnings = evaluate(query, columns)
    assert mask.tolist() == [False, True, True, False]
    assert any("trail5_stop" in w for w in warnings)


def test_the_mask_is_indexed_like_the_source():
    frame = _trades([{"v": 1}, {"v": 2}])
    frame.index = pd.Index([10, 20])
    columns = ColumnSource(frame, flatten_notes(frame))
    mask, _ = evaluate(_one("notes.v", "gt", "1"), columns)
    assert mask.index.equals(frame.index)


def test_conditions_can_reference_plain_trade_columns():
    columns = _source([{"v": 1}, {"v": 2}, {"v": 3}])   # ticks is 0, 1, 2
    assert _mask(_one("ticks", "ge", "1"), columns) == [False, True, True]


# ══ the one that matters most ════════════════════════════════════════════════
def test_evaluate_never_raises_on_hostile_data():
    junk = [
        {"x": {"nested": "dict"}, "y": [1, [2, 3]], "z": float("inf")},
        {"x": None, "y": [], "z": -float("inf")},
        {"x": "10:22", "y": "not a list", "z": "text"},
        {"x": True, "y": 5, "z": None},
        {},
    ]
    frame = _trades(junk)
    frame["stamp"] = pd.to_datetime(["2026-01-05 10:22", None,
                                     "2026-01-05 14:00", None, None])
    frame["blob"] = [b"bytes", pd.NaT, np.nan, {"a": 1}, ()]
    columns = ColumnSource(frame, flatten_notes(frame))

    targets = ["notes.x", "notes.y", "notes.z", "stamp", "blob", "ticks",
               "nope"]
    values = {0: None, 1: "3", 2: ("1", "9")}
    for column in targets:
        for op, spec in OPS.items():
            condition = Condition(column, op, values[spec.arity])
            mask, _warnings = evaluate(Query((Group((condition,)),)), columns)
            assert len(mask) == len(frame)
            assert mask.dtype == bool
