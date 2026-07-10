"""
Settings dialog — folder lists for the three plugin categories + data roots.

Rules mirrored from modules/common/backend/settings.py:
- each plugin category shows its locked in-repo default as row 0 (lock glyph,
  not selectable/removable), followed by the user's extra folders;
- data roots are fully editable (add / remove / reorder), minimum one;
- nonexistent paths render red as a warning but are kept (the folder might be
  a disconnected drive).

OK writes settings.json and accepts; Cancel discards edits.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout)
from PySide6.QtGui import QColor

from modules.common.backend.settings import (CATEGORY_LABELS,
                                             PLUGIN_CATEGORIES, Settings,
                                             _resolve)
from . import theme


class _FolderList(QGroupBox):
    """One category's folder list + Add/Remove (+ optional Up/Down) buttons."""

    def __init__(self, title: str, entries: list[str],
                 locked_first: Path | None = None, orderable: bool = False,
                 min_rows: int = 0, parent=None):
        super().__init__(title, parent)
        self._locked_first = locked_first
        self._min_rows = min_rows

        self._list = QListWidget()
        self._list.setMinimumHeight(84)
        if locked_first is not None:
            item = QListWidgetItem(f"🔒  {locked_first}   (default, always first)")
            item.setFlags(Qt.ItemIsEnabled)   # visible, not selectable
            item.setForeground(QColor(theme.TEXT_MUTED))
            self._list.addItem(item)
        for entry in entries:
            self._add_row(entry)

        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._on_add)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._on_remove)
        btns = QVBoxLayout()
        btns.addWidget(add_btn)
        btns.addWidget(rm_btn)
        if orderable:
            up = QPushButton("Up")
            down = QPushButton("Down")
            up.clicked.connect(lambda: self._move(-1))
            down.clicked.connect(lambda: self._move(+1))
            btns.addWidget(up)
            btns.addWidget(down)
        btns.addStretch()

        lay = QHBoxLayout(self)
        lay.addWidget(self._list, stretch=1)
        lay.addLayout(btns)

    # ── rows ──────────────────────────────────────────────────────────────────
    def _add_row(self, entry: str) -> None:
        item = QListWidgetItem(entry)
        item.setData(Qt.UserRole, entry)
        if not _resolve(entry).exists():
            item.setForeground(QColor(theme.BAD))
            item.setToolTip("Folder does not exist")
        elif not Path(entry).is_absolute():
            item.setToolTip("Relative to the project folder")
        self._list.addItem(item)

    def _first_editable(self) -> int:
        return 1 if self._locked_first is not None else 0

    def _on_add(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder")
        if folder:
            self._add_row(folder)

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        lo = self._first_editable()
        if row < lo:
            return
        if (self._list.count() - lo) <= self._min_rows:
            return   # keep at least min_rows editable entries (data roots: 1)
        self._list.takeItem(row)

    def _move(self, delta: int) -> None:
        row = self._list.currentRow()
        new = row + delta
        lo = self._first_editable()
        if row < lo or new < lo or new >= self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(new, item)
        self._list.setCurrentRow(new)

    def entries(self) -> list[str]:
        return [self._list.item(i).data(Qt.UserRole)
                for i in range(self._first_editable(), self._list.count())]


class SettingsDialog(QDialog):
    """Edit + persist the app settings. exec() == Accepted means saved."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — folders")
        self.setMinimumWidth(680)
        self._settings = settings

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Plugin folders are searched in order; the in-repo default is "
            "always first. Each data root is a full tree: raw_dbn/, parquet/, "
            "trades/, optimizations/, news_and_holidays/.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        lay.addWidget(intro)

        self._plugin_lists: dict[str, _FolderList] = {}
        for category in PLUGIN_CATEGORIES:
            fl = _FolderList(CATEGORY_LABELS[category],
                             settings.extra_plugin_dirs[category],
                             locked_first=settings.default_plugin_dir(category))
            self._plugin_lists[category] = fl
            lay.addWidget(fl)

        self._roots_list = _FolderList("Data root folders",
                                       settings.data_roots_raw,
                                       orderable=True, min_rows=1)
        lay.addWidget(self._roots_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _on_ok(self) -> None:
        for category, fl in self._plugin_lists.items():
            self._settings.extra_plugin_dirs[category] = fl.entries()
        roots = self._roots_list.entries()
        self._settings.data_roots_raw = roots or ["data"]
        self._settings.save()
        self.accept()
