"""
The Trades & Notes section, and the filter stage it contributes to the shared
report's chain.

The section is unusual: it is a filter AND the display of what it filtered,
and the user can drag it anywhere in the stack. The tests that matter most
are therefore the ones about that duality — one Apply causing exactly one
filter pass, a query matching nothing leaving the section reachable, and
flattened columns never escaping into a saved trades frame.
"""

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                 # noqa: E402

from modules.common.backend.settings import Settings          # noqa: E402
from modules.common.trade_report.backend.trade_stats import DAY_TYPE_ORDER  # noqa: E402
from modules.common.trade_report.ui.notes_table_section import (  # noqa: E402
    TradeNotesSection)


# ── fixtures ─────────────────────────────────────────────────────────────────
NOTES = [
    {"tp_type": "tp_vwap_2", "flip_count": 0, "breakout_time": "10:22",
     "absorption_time": ["10:53", "10:54"], "escalated": False},
    {"tp_type": "tp_vwap_3", "flip_count": 3, "breakout_time": "9:05",
     "absorption_time": "10:53", "escalated": True},
    {"tp_type": "1:1", "flip_count": 3, "breakout_time": "14:31",
     "escalated": False, "trail5_stop": 5480.0},
]


def _host_frame(notes=None, trade_types=("alpha", "beta", "alpha")):
    """A frame shaped like an optimizer run's rows (pnl_ticks + day_bucket,
    which from_optimizer_rows normalises, not ticks + day_type)."""
    notes = NOTES if notes is None else notes
    rows = []
    for n, ttype in enumerate(trade_types):
        day = f"2026-01-{5 + n:02d}"
        rows.append({
            "date": pd.Timestamp(day),
            "entry_time": pd.Timestamp(f"{day} 09:00", tz="America/New_York"),
            "exit_time": pd.Timestamp(f"{day} 10:00", tz="America/New_York"),
            "pnl_ticks": [40.0, -10.0, 120.0][n],
            "direction": "long", "entry_price": 5000.0, "exit_price": 5002.5,
            "exit_reason": "target", "trade_type": ttype,
            "day_bucket": "normal",
        })
    frame = pd.DataFrame(rows)
    if notes:
        frame["notes"] = [json.dumps(n) for n in notes[:len(rows)]]
    return frame


def _plain_frame(notes=None):
    """The backtester's shape: ticks + day_type."""
    frame = _host_frame(notes)
    frame["ticks"] = frame.pop("pnl_ticks")
    frame["day_type"] = frame.pop("day_bucket")
    return frame


@pytest.fixture
def section(qtbot):
    widget = TradeNotesSection()
    qtbot.addWidget(widget)
    return widget


def _condition(row, column_id, op, value=None):
    """Drive one condition row the way a user would."""
    row._column_combo.setCurrentIndex(row._column_combo.findData(column_id))
    index = row._op_combo.findData(op)
    assert index >= 0, f"{op} not offered for {column_id}"
    row._op_combo.setCurrentIndex(index)
    if value is None:
        return
    editor = row._value.currentWidget()
    if hasattr(editor, "set_values"):
        editor.set_values(value if isinstance(value, (list, tuple)) else [value])
    else:
        editor.setText(str(value))


def _first_row(widget):
    return widget._groups[0]._rows[0]


# ══ construction & degradation ═══════════════════════════════════════════════
def test_constructs_with_nothing_to_show(section):
    section.set_source(None)
    section.set_trades(pd.DataFrame())
    assert section.mask_for(None) is None
    assert not section.applies_to_report()


def test_a_strategy_without_notes_still_renders_its_trade_columns(section):
    frame = _plain_frame(notes=[])
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    assert section._notes_boxes == {}
    assert "no trade notes" in section._status.text()
    assert section._table.model().rowCount() == 3


def test_an_empty_frame_does_not_raise(section):
    empty = _plain_frame().iloc[0:0]
    section.set_source(empty)
    section.expand_notes()
    section.set_trades(empty)
    assert section._table.model().rowCount() == 0


# ══ columns ══════════════════════════════════════════════════════════════════
def test_columns_split_into_main_info_and_notes(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    assert "ticks" in section._main_boxes
    assert "notes.tp_type" in section._notes_boxes
    # the raw JSON blob is available but never shown by default
    assert not section._main_boxes["notes"].isChecked()
    assert section._main_boxes["ticks"].isChecked()


def test_show_all_masters_toggle_their_group_only(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    before = section._table.model().columnCount()

    section._notes_all.setChecked(False)
    assert not any(b.isChecked() for b in section._notes_boxes.values())
    assert all(not b.isChecked() for b in section._notes_boxes.values())
    assert section._table.model().columnCount() < before
    assert section._main_boxes["ticks"].isChecked()      # untouched

    section._notes_all.setChecked(True)
    assert section._table.model().columnCount() == before


def test_unchecking_one_box_clears_its_master(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    assert section._notes_all.isChecked()
    section._notes_boxes["notes.tp_type"].setChecked(False)
    assert not section._notes_all.isChecked()


def test_sorting_stays_on_its_column_when_another_is_hidden(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    header = section._table.horizontalHeader()
    model = section._table.model()
    labels = [model.headerData(n, Qt.Horizontal)
              for n in range(model.columnCount())]
    header.setSortIndicator(labels.index("flip_count"), Qt.AscendingOrder)

    section._main_boxes["direction"].setChecked(False)
    model = section._table.model()
    assert model.headerData(header.sortIndicatorSection(),
                            Qt.Horizontal) == "flip_count"


# ══ the query builder ════════════════════════════════════════════════════════
def test_groups_and_conditions_can_be_added_and_removed(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)

    group = section._groups[0]
    group.add_condition()
    assert len(group._rows) == 2
    group._rows[1].removed.emit(group._rows[1])
    assert len(group._rows) == 1

    section._add_group()
    assert len(section._groups) == 2
    section._remove_group(section._groups[1])
    assert len(section._groups) == 1

    # the last group is emptied, never removed — otherwise there is nowhere
    # left to type a condition
    section._remove_group(section._groups[0])
    assert len(section._groups) == 1
    assert len(section._groups[0]._rows) == 1


def test_clear_all_resets_to_one_empty_group(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    _condition(_first_row(section), "notes.flip_count", "ge", "2")
    section._add_group()
    section._commit()
    assert section.is_narrowing()

    section._clear()
    assert len(section._groups) == 1
    assert not section.is_narrowing()
    assert section._table.model().rowCount() == 3


def test_operators_are_offered_per_column_kind(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    row = _first_row(section)

    def ops(column_id):
        row._column_combo.setCurrentIndex(row._column_combo.findData(column_id))
        return [row._op_combo.itemData(n) for n in range(row._op_combo.count())]

    assert "gt" in ops("notes.flip_count")
    assert "time_lt" in ops("notes.breakout_time")
    assert ops("notes.escalated")[:2] == ["is_true", "is_false"]
    assert "count_gt" in ops("notes.absorption_time")
    assert "one_of" in ops("notes.tp_type")
    assert "is_empty" in ops("notes.tp_type")


def test_a_time_query_compares_as_time_of_day(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    _condition(_first_row(section), "notes.breakout_time", "time_lt", "10:00")
    section._commit()
    # only the 9:05 breakout, which sorts AFTER "10:00" as plain text
    assert section._table.model().rowCount() == 1


def test_two_groups_joined_by_or(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    _condition(_first_row(section), "ticks", "gt", "100")
    second = section._add_group()
    _condition(second._rows[0], "notes.flip_count", "ge", "3")
    section._top_join.setCurrentIndex(1)            # any group
    section._commit()
    assert section._table.model().rowCount() == 2   # rows 1 and 2


def test_the_match_readout_counts_against_the_whole_set(section):
    frame = _plain_frame()
    section.set_source(frame)
    section.expand_notes()
    section.set_trades(frame)
    _condition(_first_row(section), "notes.flip_count", "ge", "3")
    section._refresh_preview()
    assert section._match.text() == "2 of 3 trades match"


# ══ scope: table only vs the whole report ════════════════════════════════════
@pytest.fixture
def host(qtbot, tmp_path):
    """The shared report itself — the same widget all three modules use."""
    from modules.common.trade_report.ui import TradeReport
    report = TradeReport(Settings({}, [str(tmp_path)]),
                         track_worker=lambda w: None)
    qtbot.addWidget(report)
    return report


def _load(host, tmp_path, frames=None):
    """Drive the report the way the Combine tab does. Returns the source, so a
    test can switch scope; the notes section is host._notes."""
    from modules.optimizer.combine_detail import CombineReportSource
    frames = frames or {"all": _host_frame()}
    source = CombineReportSource(host)
    source.show_set(resolve=lambda scope: frames[scope], scope="all",
                    header_stem="combined", save_stem=["ES", "run", "k2"],
                    ticker="ES", tick_size=0.25, ticks_per_point=4.0,
                    dataset="", root=tmp_path, regime_start="2026-01-05",
                    regime_end="2026-01-07",
                    # every bucket checked, so the day filter itself is not
                    # narrowing and `filtered` can only come from the query
                    day_bucket_defaults={tag for tag, _ in DAY_TYPE_ORDER})
    host._notes.expand_notes()
    return source


def test_scope_off_narrows_only_the_table(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "ge", "3")
    notes._commit()

    assert notes._table.model().rowCount() == 2
    assert len(host.filtered_trades()) == 3          # the report is untouched
    assert host.is_filtered() is False
    assert host._actions_context()["filtered"] is False


def test_scope_on_narrows_the_whole_report(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "ge", "3")
    notes._commit()
    notes._scope.setChecked(True)

    assert len(host.filtered_trades()) == 2
    assert host.is_filtered() is True
    assert host._actions_context()["filtered"] is True
    assert notes._table.model().rowCount() == 2


def test_a_query_that_matches_everything_is_not_a_filter(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "ge", "0")
    notes._commit()
    notes._scope.setChecked(True)

    assert len(host.filtered_trades()) == 3
    assert notes.is_narrowing() is False
    assert host.is_filtered() is False


def test_one_apply_runs_the_filter_chain_exactly_once(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    notes._scope.setChecked(True)
    calls = []
    original = host._run_filters
    host._run_filters = lambda: (calls.append(1), original())[1]

    _condition(_first_row(notes), "notes.flip_count", "ge", "3")
    notes._commit()
    assert calls == [1]


def test_a_query_matching_nothing_keeps_the_section_reachable(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "gt", "99")
    notes._commit()
    notes._scope.setChecked(True)

    stack = host.panel.sections
    assert stack.frame("metrics").isVisible() is False
    assert stack.frame("trades_table").isVisible() is True
    assert notes._table.model().rowCount() == 0

    notes._clear()                                   # and it can be undone
    assert stack.frame("trades_table").isVisible() is True
    assert len(host.filtered_trades()) == 3


def test_flattened_columns_never_reach_the_saved_trades(host, tmp_path):
    _load(host, tmp_path)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "ge", "3")
    notes._commit()
    notes._scope.setChecked(True)

    saved = host._actions_context()["trades"]
    assert not [c for c in saved.columns if str(c).startswith("notes.")]
    assert "notes" in saved.columns                  # the raw column survives


# ══ re-slicing (the Optimizer's constant) ════════════════════════════════════
def test_a_query_on_a_shared_key_survives_a_scope_switch(host, tmp_path):
    frames = {"all": _host_frame(), "oos": _host_frame(NOTES[:2],
                                                       ("alpha", "beta"))}
    source = _load(host, tmp_path, frames)
    notes = host._notes
    _condition(_first_row(notes), "notes.flip_count", "ge", "3")
    notes._commit()
    notes._scope.setChecked(True)
    assert len(host.filtered_trades()) == 2

    source.set_scope("oos")
    assert len(host.filtered_trades()) == 1
    assert _first_row(notes)._orphan is False


def test_a_query_on_a_vanished_key_degrades_to_identity(host, tmp_path):
    """trail5_stop exists on one trade of the full set and on none of the
    out-of-sample slice. The condition must be flagged and IGNORED, not
    silently empty the whole report."""
    frames = {"all": _host_frame(), "oos": _host_frame(NOTES[:2],
                                                       ("alpha", "beta"))}
    source = _load(host, tmp_path, frames)
    notes = host._notes
    _condition(_first_row(notes), "notes.trail5_stop", "gt", "0")
    notes._commit()
    notes._scope.setChecked(True)
    assert len(host.filtered_trades()) == 1

    source.set_scope("oos")
    assert _first_row(notes)._orphan is True
    assert len(host.filtered_trades()) == 2          # identity, not zero
