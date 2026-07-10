"""
MainMenuWindow — the launcher.

Header (app title + settings gear), a 2-column grid of module cards, and a
footer summarizing the configured data roots. Clicking a card spawns a FRESH
module window every time (multi-instance by design); the menu keeps strong
Python references in self._open_windows so windows aren't garbage-collected,
and prunes them on destroy.

Close policy: closing the menu with open module windows asks for confirmation
("running jobs will be cancelled"), then closes every tracked window; the app
quits when the last window is gone.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QToolButton, QVBoxLayout, QWidget)

from modules.common.backend.settings import Settings
from modules.common.ui import theme
from modules.common.ui.settings_dialog import SettingsDialog
from .cards import ModuleCard
from .registry import MODULES


class MainMenuWindow(QWidget):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._open_windows: list[QWidget] = []
        self._instance_counters: dict[str, int] = {}

        self.setWindowTitle("Quant Research Platform")
        self.setObjectName("menuRoot")   # gradient backdrop (theme.py)
        # custom QWidget subclasses only paint QSS backgrounds with this set
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.resize(920, 680)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 18)
        lay.setSpacing(16)

        # ── header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title = QLabel("Research Engine")
        title.setStyleSheet("font-size: 29px; font-weight: 700; "
                            "background: transparent;")
        subtitle = QLabel("ES · NQ · Intraday Futures")
        subtitle.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 13px; "
                               f"background: transparent;")
        accent = QFrame()
        accent.setObjectName("accentBar")
        accent.setFixedSize(46, 3)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_box.addWidget(accent)
        header.addLayout(title_box)
        header.addStretch()

        gear = QToolButton()
        gear.setText("⚙")
        gear.setToolTip("Settings — plugin folders && data roots")
        gear.setStyleSheet(
            f"QToolButton {{ font-size: 20px; padding: 6px 10px; "
            f"background: {theme.SURFACE}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 8px; }} "
            f"QToolButton:hover {{ border-color: {theme.ACCENT}; }}")
        gear.clicked.connect(self._open_settings)
        header.addWidget(gear, alignment=Qt.AlignTop)
        lay.addLayout(header)

        # ── card grid (2 columns) ─────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        for n, spec in enumerate(MODULES):
            card = ModuleCard(spec.number, spec.title, spec.blurb)
            card.clicked.connect(lambda s=spec: self._open_module(s))
            grid.addWidget(card, n // 2, n % 2)
        lay.addLayout(grid)
        lay.addStretch()

        # ── footer ────────────────────────────────────────────────────────────
        self._footer = QLabel()
        self._footer.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(self._footer)
        self._refresh_footer()

    # ── footer / settings ─────────────────────────────────────────────────────
    def _refresh_footer(self) -> None:
        roots = " · ".join(str(r) for r in self._settings.data_roots)
        self._footer.setText(f"Data roots: {roots}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, parent=self)
        if dlg.exec():
            self._refresh_footer()
            # Already-open windows keep their state; new windows (and any
            # window's own Refresh) read the updated settings.

    # ── module spawning ───────────────────────────────────────────────────────
    def _open_module(self, spec) -> None:
        try:
            window = spec.create_window(self._settings)
        except Exception as e:  # noqa: BLE001 — a broken module shouldn't kill the menu
            QMessageBox.critical(self, spec.title,
                                 f"Could not open {spec.title}:\n{e}")
            return

        count = self._instance_counters.get(spec.key, 0) + 1
        self._instance_counters[spec.key] = count
        suffix = f" ({count})" if count > 1 else ""
        window.setWindowTitle(f"{spec.title}{suffix} — Quant Research")
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.destroyed.connect(
            lambda _=None, w=window: self._forget_window(w))
        self._open_windows.append(window)
        window.showMaximized()

    def _forget_window(self, window) -> None:
        self._open_windows = [w for w in self._open_windows if w is not window]

    # ── close policy ──────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        open_windows = list(self._open_windows)
        if open_windows:
            answer = QMessageBox.question(
                self, "Close all?",
                f"{len(open_windows)} module window(s) are open. Close "
                f"everything? Running jobs will be cancelled.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for w in open_windows:
                w.close()
        event.accept()
