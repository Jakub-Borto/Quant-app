"""
InstanceLogPanel — the shared console for every running script instance.

One chip per instance (script name + instance number + port), one console.
Selecting a chip re-renders that instance's full buffered log (the ring
buffer in ScriptInstance is the source of truth), so history — including a
traceback from an instance that crashed while another was selected — is
always available. Live output appends only for the selected instance.

The console never closes itself: a script finishing, crashing or being
killed only recolors the chip. The Kill button ends the SELECTED instance's
process deliberately; removal of a chip + log happens exclusively through
the window's explicit Dismiss control, and only once the process is dead.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QTextCursor
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from modules.common.ui import theme
from .process_manager import ScriptInstance

STATE_COLORS = {
    ScriptInstance.STARTING: "#c99a35",   # amber
    ScriptInstance.RUNNING:  "#2f9e53",   # green
    ScriptInstance.EXITED:   theme.TEXT_MUTED,
    ScriptInstance.CRASHED:  "#c94444",   # red
}


def _dot_icon(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pm)


class InstanceLogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._chips: dict[ScriptInstance, QPushButton] = {}
        self._current: ScriptInstance | None = None

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self._chips_lay = QHBoxLayout()
        self._chips_lay.setSpacing(6)

        self._kill = QPushButton("Kill")
        self._kill.setToolTip("Kill the selected instance's process "
                              "(its log stays until dismissed)")
        self._kill.setEnabled(False)
        self._kill.clicked.connect(self._on_kill)

        top = QHBoxLayout()
        top.addLayout(self._chips_lay)
        top.addStretch()
        top.addWidget(self._kill)

        self._console = QPlainTextEdit()
        self._console.setObjectName("console")
        self._console.setReadOnly(True)
        self._console.setFixedHeight(280)
        self._console.setMaximumBlockCount(4000)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addLayout(top)
        lay.addWidget(self._console)

    # ── public API ────────────────────────────────────────────────────────────
    def add_instance(self, inst: ScriptInstance) -> None:
        chip = QPushButton(self._chip_text(inst))
        chip.setCheckable(True)
        chip.setIcon(_dot_icon(STATE_COLORS[inst.state]))
        chip.setToolTip(str(inst.script_path))
        chip.clicked.connect(lambda _=False, i=inst: self.select_instance(i))
        self._group.addButton(chip)
        self._chips_lay.addWidget(chip)
        self._chips[inst] = chip

        inst.output.connect(lambda text, i=inst: self._on_live_output(i, text))
        inst.state_changed.connect(lambda i=inst: self._on_state_changed(i))

        self.select_instance(inst)   # a fresh launch grabs the console

    def remove_instance(self, inst: ScriptInstance) -> None:
        chip = self._chips.pop(inst, None)
        if chip is None:
            return
        self._group.removeButton(chip)
        self._chips_lay.removeWidget(chip)
        chip.deleteLater()
        if inst is self._current:
            self._current = None
            remaining = list(self._chips)
            if remaining:
                self.select_instance(remaining[-1])
            else:
                self._console.setPlainText("")
                self._kill.setEnabled(False)

    def select_instance(self, inst: ScriptInstance) -> None:
        if inst not in self._chips:
            return
        self._current = inst
        self._chips[inst].setChecked(True)
        self._console.setPlainText(inst.log_text())
        self._scroll_to_bottom()
        self._kill.setEnabled(inst.is_alive())

    def current_instance(self) -> ScriptInstance | None:
        return self._current

    # ── internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _chip_text(inst: ScriptInstance) -> str:
        return f"{inst.label}  :{inst.port}" if inst.port else inst.label

    def _on_live_output(self, inst: ScriptInstance, text: str) -> None:
        if inst is not self._current:
            return
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self._scroll_to_bottom()

    def _on_state_changed(self, inst: ScriptInstance) -> None:
        chip = self._chips.get(inst)
        if chip is not None:
            chip.setIcon(_dot_icon(STATE_COLORS[inst.state]))
        if inst is self._current:
            self._kill.setEnabled(inst.is_alive())

    def _on_kill(self) -> None:
        if self._current is not None and self._current.is_alive():
            self._current.stop()

    def _scroll_to_bottom(self) -> None:
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())
