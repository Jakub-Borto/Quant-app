"""
Report layout dialog — reorder the trade report's sections and choose how each
one shows (always visible / dropdown / hidden).

Reordering mirrors settings_dialog._FolderList._move (takeItem/insertItem with
Up/Down buttons): the app has no drag-drop anywhere, so this stays consistent
with the one existing precedent.

Saving writes ui_prefs[UI_PREF_TRADE_REPORT] and pings LAYOUT_BUS, so every
open report window restacks immediately — including the Optimizer's cell
drill-down, which shares the layout.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout)

from modules.common.backend.settings import SETTINGS_PATH, UI_PREF_TRADE_REPORT
from .. import theme
from .sections import (DEFAULT_ORDER, LAYOUT_BUS, MODE_LABELS, MODES,
                       SPEC_BY_KEY, resolve_layout)


class ReportLayoutDialog(QDialog):
    """Order + per-section mode for the shared trade report."""

    def __init__(self, settings, settings_path=SETTINGS_PATH, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._settings_path = settings_path
        self.setWindowTitle("Report layout")
        self.setMinimumWidth(560)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Order the report's sections and choose how each one shows. "
            "This layout is shared by the Backtester and the Optimizer's cell "
            "detail.\n\nOrder is a display choice only — it never changes "
            "which trades a section was computed from.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        lay.addWidget(intro)

        row = QHBoxLayout()
        self._list = QListWidget()
        self._list.setMinimumHeight(340)
        row.addWidget(self._list, 1)

        buttons = QVBoxLayout()
        up = QPushButton("Up")
        down = QPushButton("Down")
        reset = QPushButton("Reset to defaults")
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(+1))
        reset.clicked.connect(self._on_reset)
        buttons.addWidget(up)
        buttons.addWidget(down)
        buttons.addSpacing(12)
        buttons.addWidget(QLabel("Show as"))
        self._mode = QComboBox()
        for mode in MODES:
            self._mode.addItem(MODE_LABELS[mode], mode)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        buttons.addWidget(self._mode)
        buttons.addStretch()
        buttons.addWidget(reset)
        row.addLayout(buttons)
        lay.addLayout(row)

        note = QLabel("🔒 sections are always visible — they hold controls you "
                      "would not want to lose access to. They can still be "
                      "reordered.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(note)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self._modes: dict[str, str] = {}
        self._load(resolve_layout(settings.ui_pref(UI_PREF_TRADE_REPORT)))
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.setCurrentRow(0)

    # ── population ────────────────────────────────────────────────────────────
    def _load(self, layout) -> None:
        order, modes = layout
        self._modes = dict(modes)
        self._list.clear()
        for key in order:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, key)
            self._list.addItem(item)
            self._refresh_row(self._list.count() - 1)

    def _refresh_row(self, row: int) -> None:
        item = self._list.item(row)
        key = item.data(Qt.UserRole)
        spec = SPEC_BY_KEY[key]
        lock = "🔒  " if spec.lockable else ""
        item.setText(f"{lock}{spec.title}   ·   {MODE_LABELS[self._modes[key]]}")

    def _current_key(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    # ── interactions ──────────────────────────────────────────────────────────
    def _move(self, delta: int) -> None:
        row = self._list.currentRow()
        new = row + delta
        if row < 0 or new < 0 or new >= self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(new, item)
        self._list.setCurrentRow(new)

    def _on_row_changed(self, _row: int) -> None:
        key = self._current_key()
        if key is None:
            return
        spec = SPEC_BY_KEY[key]
        self._mode.blockSignals(True)
        self._mode.setCurrentIndex(self._mode.findData(self._modes[key]))
        self._mode.blockSignals(False)
        self._mode.setEnabled(not spec.lockable)

    def _on_mode_changed(self, _index: int) -> None:
        key = self._current_key()
        if key is None or SPEC_BY_KEY[key].lockable:
            return
        self._modes[key] = self._mode.currentData()
        self._refresh_row(self._list.currentRow())

    def _on_reset(self) -> None:
        self._load(resolve_layout(None))
        self._list.setCurrentRow(0)

    def _on_ok(self) -> None:
        order = [self._list.item(i).data(Qt.UserRole)
                 for i in range(self._list.count())]
        modes = {k: self._modes[k] for k in order
                 if self._modes[k] != SPEC_BY_KEY[k].default_mode}
        self._settings.set_ui_pref(UI_PREF_TRADE_REPORT,
                                   {"order": order, "modes": modes})
        self._settings.save(self._settings_path)
        LAYOUT_BUS.changed.emit()
        self.accept()


DEFAULT_SECTION_ORDER = DEFAULT_ORDER   # re-export for convenience
