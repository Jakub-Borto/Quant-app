"""
The Trades & Notes section — every trade as a row, every key of the
strategies' `notes` JSON as its own column, with column-visibility toggles,
sorting, and a grouped AND/OR query builder.

The section is BOTH a filter and the display of what it filtered, which is
the only genuinely tricky thing about it. Three rules keep that from
oscillating:

  1. STRUCTURE is rebuilt only by set_source(); DATA only by set_trades().
     set_trades renders — it never rebuilds a widget, never re-evaluates the
     query, and never emits.
  2. set_source() NEVER emits. Both hosts call it from inside a handler that
     goes on to run _apply_filters itself (window._on_run_finished,
     report_host.set_source), so an emit there would re-enter the very chain
     that is about to run.
  3. queryChanged fires only on a COMMIT — Apply, Clear all, or the scope
     checkbox — and only when the scope checkbox is on. With the scope off
     the query is a view concern and the section re-filters its own table
     without ever calling the host.

Column-visibility toggles never reach the host at all: they re-slice the
cached frame in place.

Flattening is lazy (showEvent) because the section ships collapsed and the
work is proportional to rows x keys — see FLATTEN_ROW_CAP.
"""

import pandas as pd
from PySide6.QtCore import QRegularExpression, Qt, QTimer, Signal
from PySide6.QtGui import (QDoubleValidator, QIntValidator,
                           QRegularExpressionValidator)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPushButton,
                               QSizePolicy, QStackedWidget, QToolButton,
                               QVBoxLayout, QWidget, QWidgetAction)

from modules.common.backend import trade_notes as tn
from ..dataframe_model import make_table_view, update_table_view
from ..widgets import Banner, Caption, hline, pin_minimum_height
from .sections import ReportSection

# Flattening is rows x keys of json.loads + dict.get. Below the cap it runs
# the moment the section is first shown; above it the user asks for it
# explicitly, because an optimizer run can hand us 100k+ rows (see the note at
# the top of dataframe_model.py) and nobody wants to pay that by accident.
FLATTEN_ROW_CAP = 25_000

# What the plain trades table used to show, in its order — floated to the
# front of the Main info group so the table opens looking familiar.
DEFAULT_LEAD_COLUMNS = ("date", "direction", "entry_time", "exit_time",
                        "entry_price", "exit_price", "exit_reason", "ticks",
                        "trade_type", "day_type", "regime")
# The raw JSON blob: available, but never checked by default — it is 500
# characters of unreadable text now that its contents are their own columns.
NEVER_DEFAULT = ("notes",)

_EDITORS = ("none", "number", "range", "time", "time_range", "text",
            "choices", "count")
_TIME_RE = QRegularExpression(r"^\s*\d{1,2}:[0-5]\d(:[0-5]\d)?\s*$")


def _tool_button(text: str, tooltip: str = "") -> QToolButton:
    button = QToolButton()
    button.setText(text)
    if tooltip:
        button.setToolTip(tooltip)
    return button


def _set_primary(button: QPushButton, primary: bool) -> None:
    """Toggle the accent style (QSS: QPushButton[primary="true"])."""
    if bool(button.property("primary")) == primary:
        return
    button.setProperty("primary", primary)
    button.style().unpolish(button)
    button.style().polish(button)


class _Catalog:
    """What a condition row needs to know about the available columns."""

    def __init__(self, columns=None, lead=()):
        self.columns = columns                      # tn.ColumnSource | None
        self.ids: list[str] = []
        self.labels: dict[str, str] = {}
        self.separator_after: str | None = None
        if columns is None:
            return
        trade_ids = _ordered_trade_columns(columns.trade_columns, lead)
        self.ids = trade_ids + columns.notes_columns
        for cid in self.ids:
            self.labels[cid] = columns.label(cid)
        self.separator_after = trade_ids[-1] if trade_ids else None


def _ordered_trade_columns(columns: list, lead=()) -> list:
    """The lead columns first, in their canonical order, then the rest."""
    ordered = [c for c in lead if c in columns]
    return ordered + [c for c in columns if c not in ordered]


# ══ the "is one of" picker ═══════════════════════════════════════════════════
class _ChoicePicker(QToolButton):
    """Multi-select over a column's distinct display values. A checkable
    QListWidget inside the menu, not checkable QActions — a QMenu closes on
    every trigger, which makes picking three values maddening."""

    valueChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._menu = QMenu(self)
        self._list = QListWidget()
        self._list.setMaximumHeight(240)
        self._list.setMinimumWidth(180)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(self._list)
        self._menu.addAction(action)
        self.setMenu(self._menu)
        self._truncated = 0
        self._list.itemChanged.connect(self._on_item_changed)
        self._update_text()

    def set_choices(self, values, total: int) -> None:
        keep = set(self.values())
        self._truncated = max(0, total - len(values))
        self._list.blockSignals(True)
        self._list.clear()
        for value in values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if value in keep else Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._update_text()

    def values(self) -> tuple:
        return tuple(self._list.item(n).text()
                     for n in range(self._list.count())
                     if self._list.item(n).checkState() == Qt.Checked)

    def set_values(self, values) -> None:
        wanted = set(values or ())
        self._list.blockSignals(True)
        for n in range(self._list.count()):
            item = self._list.item(n)
            item.setCheckState(Qt.Checked if item.text() in wanted
                               else Qt.Unchecked)
        self._list.blockSignals(False)
        self._update_text()

    def _on_item_changed(self, _item) -> None:
        self._update_text()
        self.valueChanged.emit()

    def _update_text(self) -> None:
        chosen = self.values()
        if not chosen:
            self.setText("choose values…")
        elif len(chosen) == 1:
            self.setText(chosen[0][:40])
        else:
            self.setText(f"{len(chosen)} selected")
        if self._truncated:
            self.setToolTip(f"showing the first {self._list.count()} values "
                            f"({self._truncated} more) — use 'contains' for "
                            f"the rest")
        else:
            self.setToolTip("")


# ══ one condition ════════════════════════════════════════════════════════════
class _ConditionRow(QWidget):
    """[column][operator][value][x]. The COLUMN ID is the source of truth, not
    the combo index — the Optimizer re-slices constantly and the id must
    survive a rebuild of the combo's contents."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, catalog: _Catalog, parent=None):
        super().__init__(parent)
        self._catalog = catalog
        self._column_id = None
        self._op = None
        self._orphan = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._column_combo = QComboBox()
        self._column_combo.setMinimumWidth(190)
        self._op_combo = QComboBox()
        self._op_combo.setMinimumWidth(130)

        self._value = QStackedWidget()
        self._pages = {}
        self._build_pages()
        self._value.setMinimumWidth(210)
        self._value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        remove = _tool_button("✕", "remove this condition")
        remove.clicked.connect(lambda: self.removed.emit(self))

        lay.addWidget(self._column_combo)
        lay.addWidget(self._op_combo)
        lay.addWidget(self._value, 1)
        lay.addWidget(remove)

        self._column_combo.currentIndexChanged.connect(self._on_column_changed)
        self._op_combo.currentIndexChanged.connect(self._on_op_changed)
        self.reload_columns()

    # ── pages ─────────────────────────────────────────────────────────────────
    def _build_pages(self) -> None:
        blank = QWidget()
        self._pages["none"] = self._value.addWidget(blank)

        self._number = QLineEdit()
        self._number.setValidator(QDoubleValidator())
        self._number.setPlaceholderText("value")
        self._pages["number"] = self._value.addWidget(self._number)

        self._range_lo, self._range_hi = QLineEdit(), QLineEdit()
        for edit, hint in ((self._range_lo, "from"), (self._range_hi, "to")):
            edit.setValidator(QDoubleValidator())
            edit.setPlaceholderText(hint)
        self._pages["range"] = self._value.addWidget(
            _pair(self._range_lo, self._range_hi))

        self._time = QLineEdit()
        self._time.setValidator(QRegularExpressionValidator(_TIME_RE))
        self._time.setPlaceholderText("HH:MM")
        self._pages["time"] = self._value.addWidget(self._time)

        self._time_lo, self._time_hi = QLineEdit(), QLineEdit()
        for edit, hint in ((self._time_lo, "from HH:MM"),
                           (self._time_hi, "to HH:MM")):
            edit.setValidator(QRegularExpressionValidator(_TIME_RE))
            edit.setPlaceholderText(hint)
        self._pages["time_range"] = self._value.addWidget(
            _pair(self._time_lo, self._time_hi))

        self._text = QLineEdit()
        self._text.setPlaceholderText("text")
        self._pages["text"] = self._value.addWidget(self._text)

        self._choices = _ChoicePicker()
        self._pages["choices"] = self._value.addWidget(self._choices)

        self._count = QLineEdit()
        self._count.setValidator(QIntValidator(0, 10_000))
        self._count.setPlaceholderText("count")
        self._pages["count"] = self._value.addWidget(self._count)

        for edit in (self._number, self._range_lo, self._range_hi, self._time,
                     self._time_lo, self._time_hi, self._text, self._count):
            edit.textChanged.connect(self.changed)
        self._choices.valueChanged.connect(self.changed)

    # ── catalog ───────────────────────────────────────────────────────────────
    def set_catalog(self, catalog: _Catalog) -> None:
        self._catalog = catalog
        self.reload_columns()

    def reload_columns(self) -> None:
        """Repopulate the column combo, keeping this row's chosen column id.
        A column the new slice doesn't have is kept as an ORPHAN — flagged in
        the UI and skipped by the evaluator — rather than silently rewritten
        to whatever happens to sit at index 0."""
        combo = self._column_combo
        combo.blockSignals(True)
        combo.clear()
        for cid in self._catalog.ids:
            combo.addItem(self._catalog.labels.get(cid, cid), cid)
            if cid == self._catalog.separator_after:
                combo.insertSeparator(combo.count())
        if self._column_id is None and self._catalog.ids:
            self._column_id = self._catalog.ids[0]
        index = combo.findData(self._column_id)
        self._orphan = self._column_id is not None and index < 0
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._apply_orphan_style()
        self._reload_ops()

    def _apply_orphan_style(self) -> None:
        if self._orphan:
            self._column_combo.setToolTip(
                f"'{self._column_id}' is not in this trade set — "
                f"this condition is ignored")
            self._column_combo.setStyleSheet("border: 1px solid #c99a35;")
        else:
            self._column_combo.setToolTip("")
            self._column_combo.setStyleSheet("")

    def _reload_ops(self) -> None:
        combo = self._op_combo
        combo.blockSignals(True)
        combo.clear()
        ops = []
        if not self._orphan and self._catalog.columns is not None \
                and self._column_id is not None:
            kind, numeric_ok, time_ok = self._catalog.columns.kind_info(
                self._column_id)
            ops = tn.operators_for(kind, numeric_ok, time_ok)
        elif self._op is not None:
            ops = [self._op]                    # keep an orphan row readable
        for op in ops:
            combo.addItem(tn.OPS[op].label, op)
        index = combo.findData(self._op)
        if index < 0:
            index = 0 if combo.count() else -1
        combo.setCurrentIndex(index)
        self._op = combo.currentData()
        combo.blockSignals(False)
        self._apply_editor()

    def _apply_editor(self) -> None:
        spec = tn.OPS.get(self._op)
        editor = spec.editor if spec else "none"
        self._value.setCurrentIndex(self._pages.get(editor, self._pages["none"]))
        if editor == "choices" and self._catalog.columns is not None \
                and not self._orphan:
            values, total = self._catalog.columns.choices(self._column_id)
            self._choices.set_choices(values, total)

    # ── events ────────────────────────────────────────────────────────────────
    def _on_column_changed(self) -> None:
        data = self._column_combo.currentData()
        if data is None:
            return
        self._column_id = data
        self._orphan = False
        self._apply_orphan_style()
        self._reload_ops()
        self.changed.emit()

    def _on_op_changed(self) -> None:
        data = self._op_combo.currentData()
        if data is None:
            return
        self._op = data
        self._apply_editor()
        self.changed.emit()

    # ── value ─────────────────────────────────────────────────────────────────
    def condition(self) -> tn.Condition:
        return tn.Condition(self._column_id or "", self._op or "",
                            self._value_of())

    def _value_of(self):
        spec = tn.OPS.get(self._op)
        editor = spec.editor if spec else "none"
        if editor == "none":
            return None
        if editor == "number":
            return self._number.text()
        if editor == "range":
            return (self._range_lo.text(), self._range_hi.text())
        if editor == "time":
            return self._time.text()
        if editor == "time_range":
            return (self._time_lo.text(), self._time_hi.text())
        if editor == "text":
            return self._text.text()
        if editor == "count":
            return self._count.text()
        return self._choices.values()


def _pair(left: QWidget, right: QWidget) -> QWidget:
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(left)
    lay.addWidget(right)
    return holder


# ══ one group ════════════════════════════════════════════════════════════════
class _GroupBox(QFrame):
    """A box of conditions joined by ONE connector. Groups never nest — one
    level covers everything realistic, and 'A AND (B OR C)' stays readable."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, catalog: _Catalog, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._catalog = catalog
        self._rows: list[_ConditionRow] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(7)
        pin_minimum_height(self)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._title = Caption("Group 1")
        self._join = QComboBox()
        self._join.addItem("all of", tn.JOIN_AND)
        self._join.addItem("any of", tn.JOIN_OR)
        self._join.setToolTip("how this group's conditions combine")
        self._join.currentIndexChanged.connect(self.changed)
        self._remove = _tool_button("✕", "remove this group")
        self._remove.clicked.connect(lambda: self.removed.emit(self))
        head.addWidget(self._title)
        head.addWidget(self._join)
        head.addStretch()
        head.addWidget(self._remove)
        lay.addLayout(head)

        self._rows_holder = QVBoxLayout()
        self._rows_holder.setSpacing(5)
        lay.addLayout(self._rows_holder)

        add = QPushButton("+ condition")
        add.clicked.connect(lambda: self.add_condition(emit=True))
        foot = QHBoxLayout()
        foot.addWidget(add)
        foot.addStretch()
        lay.addLayout(foot)

    # ── rows ──────────────────────────────────────────────────────────────────
    def add_condition(self, emit: bool = False) -> _ConditionRow:
        row = _ConditionRow(self._catalog)
        row.changed.connect(self.changed)
        row.removed.connect(self._remove_row)
        self._rows.append(row)
        self._rows_holder.addWidget(row)
        if emit:
            self.changed.emit()
        return row

    def _remove_row(self, row: _ConditionRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        if not self._rows:
            self.add_condition()
        self.changed.emit()

    def clear_conditions(self) -> None:
        for row in list(self._rows):
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
        self.add_condition()

    def set_catalog(self, catalog: _Catalog) -> None:
        self._catalog = catalog
        for row in self._rows:
            row.set_catalog(catalog)

    def set_index(self, index: int, removable: bool) -> None:
        self._title.setText(f"Group {index}")
        self._remove.setEnabled(removable)

    def group(self) -> tn.Group:
        return tn.Group(tuple(row.condition() for row in self._rows),
                        self._join.currentData() or tn.JOIN_AND)


# ══ the section ══════════════════════════════════════════════════════════════
class TradeNotesSection(ReportSection):
    """Content for the 'Trades & Notes' section of the shared report."""

    queryChanged = Signal()

    def __init__(self, settings=None, track_worker=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._track_worker = track_worker or (lambda w: None)

        self._source: pd.DataFrame | None = None
        self._flat: tn.FlatNotes | None = None
        self._flat_key = object()               # never equal to a fingerprint
        self._fingerprint = None
        self._columns: tn.ColumnSource | None = None
        self._catalog = _Catalog()
        self._last_trades: pd.DataFrame | None = None
        self._candidate_index = None
        self._lead_columns = list(DEFAULT_LEAD_COLUMNS)

        self._committed = tn.Query()
        self._committed_mask: pd.Series | None = None
        # sticky: once the section has been on screen, later re-slices flatten
        # straight away rather than waiting for another showEvent
        self._flatten_enabled = False
        self._big_frame_ack = False
        self._index_unique = True
        self._suppress = 0

        self._main_boxes: dict = {}
        self._notes_boxes: dict = {}
        self._groups: list[_GroupBox] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        pin_minimum_height(self)

        self._banner = Banner()
        lay.addWidget(self._banner)

        self._status = Caption("")
        lay.addWidget(self._status)

        self._flatten_btn = QPushButton("Flatten trade notes")
        self._flatten_btn.setVisible(False)
        self._flatten_btn.clicked.connect(self._on_flatten_clicked)
        flatten_row = QHBoxLayout()
        flatten_row.addWidget(self._flatten_btn)
        flatten_row.addStretch()
        lay.addLayout(flatten_row)

        lay.addWidget(self._build_columns_card())
        lay.addWidget(self._build_query_card())

        self._table = make_table_view(pd.DataFrame(), height=420)
        header = self._table.horizontalHeader()
        # ResizeToContents measures up to 1000 rows PER COLUMN on every model
        # reset; with 70 columns that is 70k formatted cells per toggle.
        header.setResizeContentsPrecision(64)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self._table)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._add_group()
        self._sync_group_titles()

    # ── construction ──────────────────────────────────────────────────────────
    def _build_columns_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(7)
        pin_minimum_height(card)

        self._main_all = QCheckBox("show all")
        self._main_all.setChecked(True)
        self._main_all.toggled.connect(
            lambda on: self._set_all(self._main_boxes, on))
        lay.addLayout(_group_head(Caption("Main info"), self._main_all))
        self._main_grid = QGridLayout()
        self._main_grid.setHorizontalSpacing(14)
        lay.addLayout(self._main_grid)

        lay.addWidget(hline())

        self._notes_all = QCheckBox("show all")
        self._notes_all.setChecked(True)
        self._notes_all.toggled.connect(
            lambda on: self._set_all(self._notes_boxes, on))
        self._notes_count = Caption("")
        lay.addLayout(_group_head(Caption("Trade notes"), self._notes_all,
                                  self._notes_count))
        self._notes_grid = QGridLayout()
        self._notes_grid.setHorizontalSpacing(14)
        lay.addLayout(self._notes_grid)
        return card

    def _build_query_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(8)
        pin_minimum_height(card)

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(Caption("Show trades matching"))
        self._top_join = QComboBox()
        self._top_join.addItem("all groups", tn.JOIN_AND)
        self._top_join.addItem("any group", tn.JOIN_OR)
        self._top_join.currentIndexChanged.connect(self._on_draft_changed)
        head.addWidget(self._top_join)
        head.addStretch()
        lay.addLayout(head)

        self._groups_holder = QVBoxLayout()
        self._groups_holder.setSpacing(7)
        lay.addLayout(self._groups_holder)

        add_group = QPushButton("+ group")
        add_group.clicked.connect(lambda: self._add_group(emit=True))
        add_row = QHBoxLayout()
        add_row.addWidget(add_group)
        add_row.addStretch()
        lay.addLayout(add_row)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._commit)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self._clear)
        self._scope = QCheckBox("Apply to the whole report")
        self._scope.setToolTip(
            "Also narrow the metrics, equity curve, breakdowns and the "
            "Save / Go to… handoff — not just this table.")
        self._scope.toggled.connect(self._on_scope_toggled)
        self._match = Caption("")
        foot.addWidget(self._apply_btn)
        foot.addWidget(self._clear_btn)
        foot.addWidget(self._scope)
        foot.addStretch()
        foot.addWidget(self._match)
        lay.addLayout(foot)
        return card

    # ══ host API ══════════════════════════════════════════════════════════════
    def set_display_hints(self, *, show_regime_filter: bool = False) -> None:
        lead = list(DEFAULT_LEAD_COLUMNS)
        if show_regime_filter:
            lead.append("regime_filter")
        self._lead_columns = lead

    def set_source(self, source: pd.DataFrame | None) -> None:
        """
        New run / new selection: the ONE entry point that rebuilds structure.
        Never emits — both hosts run _apply_filters right after calling it.
        """
        self._source = source
        self._candidate_index = None
        fingerprint = tn.notes_fingerprint(source)
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._flat = None
            self._flat_key = object()
            self._big_frame_ack = False
        self._index_unique = source is None or source.index.is_unique
        if not self._index_unique and self._scope.isChecked():
            self._scope.blockSignals(True)
            self._scope.setChecked(False)
            self._scope.blockSignals(False)
        self._scope.setEnabled(self._index_unique)
        self._scope.setToolTip(
            "" if self._index_unique else
            "This trade set has a duplicated index — the query can only "
            "narrow this table.")
        self._rebuild(re_evaluate=True)

    def set_trades(self, trades: pd.DataFrame) -> None:
        """Display only — the filtered frame the host just produced."""
        self._suppress += 1
        try:
            self._last_trades = trades
            self._refresh_table()
            self._refresh_match_label()
        finally:
            self._suppress -= 1

    def report_mask(self, frame: pd.DataFrame):
        """
        Called by the host's filter chain. Records what the query is being
        asked about (so the readout says '142 of 3,609', not '142 of 142')
        and returns the mask ONLY when the scope checkbox is on.
        """
        self._candidate_index = None if frame is None else frame.index
        self._refresh_match_label()
        if not self._scope.isChecked():
            return None
        return self.mask_for(frame)

    def mask_for(self, frame: pd.DataFrame):
        """The committed query as a numpy mask over `frame`, or None when the
        query has nothing to say. One evaluation is reused everywhere: the
        committed mask is indexed by the SOURCE, and every stage of the
        report's chain preserves index labels."""
        if frame is None or not self._active():
            return None
        try:
            return self._committed_mask.reindex(
                frame.index, fill_value=False).to_numpy()
        except Exception:      # noqa: BLE001 — never break the filter chain
            return None

    def applies_to_report(self) -> bool:
        return bool(self._scope.isChecked() and self._active())

    def is_narrowing(self) -> bool:
        """A query that matches every trade is not a filter — same rule as
        'all day types checked' not counting as filtered."""
        return bool(self._active() and not self._committed_mask.all())

    # ══ internals ═════════════════════════════════════════════════════════════
    def _active(self) -> bool:
        return (self._committed_mask is not None
                and not self._committed.is_empty())

    def showEvent(self, event):      # noqa: N802 — Qt naming
        super().showEvent(event)
        # the section ships collapsed, so EXPANDING it is what pays for the
        # flatten; nothing else in the report needs the notes columns
        self.expand_notes()

    def expand_notes(self) -> None:
        """Flatten now (what being shown does). Public so a host — or a test —
        can bring the notes columns up without waiting for a showEvent."""
        self._flatten_enabled = True
        if self._source is not None and not self._flat_ready():
            self._rebuild(re_evaluate=True)

    # ── flatten ───────────────────────────────────────────────────────────────
    def _flat_ready(self) -> bool:
        return self._flat is not None and self._flat_key == self._fingerprint

    def _ensure_flat(self) -> None:
        if self._source is None:
            self._flat = tn.FlatNotes.empty(None)
            self._flat_key = self._fingerprint
            return
        if self._flat_ready():
            return
        # deferred until the section has been on screen (or the query is
        # already wired into the report, which needs it either way)
        deferred = not (self._flatten_enabled or self._scope.isChecked())
        oversized = (len(self._source) > FLATTEN_ROW_CAP
                     and not self._big_frame_ack)
        if deferred or oversized:
            self._flat = tn.FlatNotes.empty(self._source.index)
            self._flat_key = object()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._flat = tn.flatten_notes(self._source)
        finally:
            QApplication.restoreOverrideCursor()
        self._flat_key = self._fingerprint

    def _on_flatten_clicked(self) -> None:
        self._big_frame_ack = True
        self._rebuild(re_evaluate=True)

    # ── structural rebuild ────────────────────────────────────────────────────
    def _rebuild(self, *, re_evaluate: bool) -> None:
        self._suppress += 1
        try:
            self._ensure_flat()
            self._columns = (None if self._source is None
                             else tn.ColumnSource(self._source, self._flat))
            self._catalog = _Catalog(self._columns, self._lead_columns)
            self._rebuild_checkboxes()
            for group in self._groups:
                group.set_catalog(self._catalog)
            self._update_status()
            if re_evaluate:
                self._evaluate_committed()
        finally:
            self._suppress -= 1
        self._refresh_table()
        self._refresh_match_label()

    def _update_status(self) -> None:
        self._banner.clear_message()
        if self._source is None or len(self._source) == 0:
            self._status.setText("")
            self._flatten_btn.setVisible(False)
            return
        rows = len(self._source)
        oversized = rows > FLATTEN_ROW_CAP and not self._big_frame_ack
        self._flatten_btn.setVisible(oversized)
        if oversized:
            self._flatten_btn.setText(f"Flatten trade notes for {rows:,} trades")
            self._status.setText(
                f"{rows:,} trades — trade notes not expanded yet.")
            return
        keys = len(self._flat.notes_columns) if self._flat else 0
        if "notes" not in self._source.columns:
            self._status.setText(
                f"{rows:,} trades — this strategy writes no trade notes.")
        else:
            self._status.setText(f"{rows:,} trades · {keys} note key"
                                 f"{'' if keys == 1 else 's'}")
        if self._flat and self._flat.warnings:
            self._banner.show_message("warning", " ".join(self._flat.warnings))

    def _rebuild_checkboxes(self) -> None:
        main_ids = (_ordered_trade_columns(self._catalog.columns.trade_columns,
                                           self._lead_columns)
                    if self._columns is not None else [])
        notes_ids = self._flat.notes_columns if self._flat else []
        self._main_boxes = self._fill_grid(
            self._main_grid, self._main_boxes, main_ids,
            {cid: cid for cid in main_ids}, per_row=6,
            default=lambda cid: cid not in NEVER_DEFAULT)
        self._notes_boxes = self._fill_grid(
            self._notes_grid, self._notes_boxes, notes_ids,
            self._flat.labels if self._flat else {}, per_row=5,
            default=lambda cid: True)
        self._notes_count.setText(f"{len(notes_ids)} key"
                                  f"{'' if len(notes_ids) == 1 else 's'}")
        self._sync_masters()

    def _fill_grid(self, grid: QGridLayout, existing: dict, ids: list,
                   labels: dict, per_row: int, default) -> dict:
        """Rebuild one checkbox group, carrying over what the user already
        decided for a column that survived the re-slice. A column the previous
        slice never offered arrives at its default — same rule the trade-type
        filter uses, so an entry with no trades in one scope doesn't come back
        silently hidden."""
        keep = {cid: box.isChecked() for cid, box in existing.items()}
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # unparent BEFORE deleteLater: deletion is deferred to the
                # event loop, and until then the old box keeps painting on
                # top of the one that replaced it
                widget.setParent(None)
                widget.deleteLater()
        boxes = {}
        for n, cid in enumerate(ids):
            box = QCheckBox(labels.get(cid, cid))
            box.setToolTip(cid)
            box.setChecked(keep.get(cid, default(cid)))
            box.toggled.connect(self._on_column_visibility_changed)
            grid.addWidget(box, n // per_row, n % per_row)
            boxes[cid] = box
        return boxes

    def _set_all(self, boxes: dict, checked: bool) -> None:
        if self._suppress:
            return
        self._suppress += 1
        try:
            for box in boxes.values():
                box.setChecked(checked)
        finally:
            self._suppress -= 1
        self._refresh_table()

    def _sync_masters(self) -> None:
        for master, boxes in ((self._main_all, self._main_boxes),
                              (self._notes_all, self._notes_boxes)):
            master.blockSignals(True)
            master.setChecked(bool(boxes)
                              and all(b.isChecked() for b in boxes.values()))
            master.blockSignals(False)

    def _on_column_visibility_changed(self) -> None:
        if self._suppress:
            return
        self._sync_masters()
        self._refresh_table()

    # ── the query ─────────────────────────────────────────────────────────────
    def _add_group(self, emit: bool = False) -> _GroupBox:
        group = _GroupBox(self._catalog)
        group.changed.connect(self._on_draft_changed)
        group.removed.connect(self._remove_group)
        group.add_condition()
        self._groups.append(group)
        self._groups_holder.addWidget(group)
        self._sync_group_titles()
        if emit:
            self._on_draft_changed()
        return group

    def _remove_group(self, group: _GroupBox) -> None:
        if len(self._groups) <= 1:
            group.clear_conditions()            # the last group is cleared,
            return                              # never removed
        self._groups.remove(group)
        group.setParent(None)
        group.deleteLater()
        self._sync_group_titles()
        self._on_draft_changed()

    def _sync_group_titles(self) -> None:
        removable = len(self._groups) > 1
        for n, group in enumerate(self._groups, start=1):
            group.set_index(n, removable)

    def _draft(self) -> tn.Query:
        return tn.Query(tuple(g.group() for g in self._groups),
                        self._top_join.currentData() or tn.JOIN_AND)

    def _on_draft_changed(self) -> None:
        if self._suppress:
            return
        _set_primary(self._apply_btn, self._draft() != self._committed)
        self._preview_timer.start()

    def _commit(self) -> None:
        self._committed = self._draft()
        _set_primary(self._apply_btn, False)
        self._evaluate_committed()
        if self._scope.isChecked():
            self.queryChanged.emit()            # the host re-runs the chain
        else:
            self._refresh_table()
            self._refresh_match_label()

    def _clear(self) -> None:
        self._suppress += 1
        try:
            for group in list(self._groups[1:]):
                self._groups.remove(group)
                group.setParent(None)
                group.deleteLater()
            self._groups[0].clear_conditions()
            self._top_join.setCurrentIndex(0)
            self._sync_group_titles()
        finally:
            self._suppress -= 1
        self._commit()

    def _on_scope_toggled(self, _checked: bool) -> None:
        if self._suppress:
            return
        self._refresh_table()
        self.queryChanged.emit()

    def _evaluate_committed(self) -> None:
        if self._columns is None:
            self._committed_mask = None
            return
        mask, warnings = tn.evaluate(self._committed, self._columns)
        self._committed_mask = mask
        if warnings:
            self._banner.show_message("warning", " · ".join(warnings[:3]))
        elif self._flat and self._flat.warnings:
            self._banner.show_message("warning", " ".join(self._flat.warnings))
        else:
            self._banner.clear_message()

    def _refresh_preview(self) -> None:
        if self._columns is None:
            self._match.setText("")
            return
        mask, _warnings = tn.evaluate(self._draft(), self._columns)
        self._show_match(mask)

    def _refresh_match_label(self) -> None:
        if self._columns is None or self._committed_mask is None:
            self._match.setText("")
            return
        self._show_match(self._committed_mask)

    def _show_match(self, mask: pd.Series) -> None:
        index = self._candidate_index
        if index is None:
            index = (self._last_trades.index if self._last_trades is not None
                     else mask.index)
        try:
            selected = mask.reindex(index, fill_value=False)
        except Exception:      # noqa: BLE001 — duplicated index
            selected = mask
            index = mask.index
        self._match.setText(f"{int(selected.sum()):,} of {len(index):,} "
                            f"trades match")

    # ── the table ─────────────────────────────────────────────────────────────
    def _visible_ids(self) -> list:
        return ([cid for cid, box in self._main_boxes.items() if box.isChecked()]
                + [cid for cid, box in self._notes_boxes.items()
                   if box.isChecked()])

    def _refresh_table(self) -> None:
        frame = self._last_trades
        if frame is None:
            update_table_view(self._table, pd.DataFrame())
            return
        # with the scope off the query is this table's business alone
        if not self._scope.isChecked() and self._active():
            mask = self.mask_for(frame)
            if mask is not None:
                frame = frame[mask]

        sorted_id = self._current_sort_id()
        data, labels = {}, []
        for cid in self._visible_ids():
            series = self._series_for(cid, frame)
            if series is None:
                continue
            label = self._catalog.labels.get(cid, cid)
            data[label] = series
            labels.append(label)
        table = (pd.DataFrame(data, index=frame.index, columns=labels)
                 if labels else pd.DataFrame(index=frame.index))
        # a pure date reads better than a midnight timestamp (what the plain
        # trades table did before this section replaced it)
        if "date" in table.columns:
            try:
                table["date"] = pd.to_datetime(table["date"]).dt.date
            except Exception:      # noqa: BLE001 — leave an odd date alone
                pass

        header = self._table.horizontalHeader()
        # ResizeToContents is unusable past a couple of dozen columns
        header.setSectionResizeMode(QHeaderView.Interactive if len(labels) > 15
                                    else QHeaderView.ResizeToContents)
        if len(labels) > 15:
            header.setDefaultSectionSize(110)
        self._restore_sort(sorted_id, labels)
        update_table_view(self._table, table)

    def _series_for(self, cid: str, frame: pd.DataFrame):
        if self._flat is not None and cid in self._flat.frame.columns:
            return self._flat.frame[cid].reindex(frame.index)
        if cid in frame.columns:
            return frame[cid]
        return None

    def _current_sort_id(self):
        header = self._table.horizontalHeader()
        section = header.sortIndicatorSection()
        model = self._table.model()
        if section < 0 or model is None or section >= model.columnCount():
            return None
        return model.headerData(section, Qt.Horizontal)

    def _restore_sort(self, label, labels: list) -> None:
        """Re-anchor the sort by COLUMN LABEL. update_table_view re-applies
        the indicator by section INDEX, so without this a hidden column
        silently hands its sort to whatever slid into that slot."""
        header = self._table.horizontalHeader()
        order = header.sortIndicatorOrder()
        if label in labels:
            header.setSortIndicator(labels.index(label), order)
        else:
            header.setSortIndicator(-1, Qt.AscendingOrder)


def _group_head(*widgets) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(10)
    for widget in widgets:
        lay.addWidget(widget)
    lay.addStretch()
    return lay
