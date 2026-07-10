"""
Chart View Settings — controls for the single-trade candlestick window.

settings() returns the EXACT dict shape of the old render_chart_view_controls:
{"view_mode": "Fixed session time" | "Candles before entry",
 "candles_before": int | None, "session_start_time": datetime.time | None,
 "candles_after": int} — consumed unchanged by
 modules.common.backend.chart_window.resolve_chart_window.
"""

from PySide6.QtCore import QTime, Signal
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QRadioButton,
                               QSpinBox, QTimeEdit, QVBoxLayout, QWidget)

from ..widgets import SectionHeader

MODE_FIXED   = "Fixed session time"
MODE_CANDLES = "Candles before entry"


class ChartViewControls(QWidget):
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(SectionHeader("Chart View Settings"))

        row = QHBoxLayout()

        # mode radio (default = first option, like st.radio)
        self._fixed = QRadioButton(MODE_FIXED)
        self._candles = QRadioButton(MODE_CANDLES)
        self._fixed.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._fixed)
        group.addButton(self._candles)
        row.addWidget(QLabel("Chart start from:"))
        row.addWidget(self._fixed)
        row.addWidget(self._candles)
        row.addSpacing(20)

        self._before_label = QLabel("Candles before entry")
        self._before = QSpinBox()
        self._before.setRange(1, 390)
        self._before.setValue(30)
        row.addWidget(self._before_label)
        row.addWidget(self._before)

        self._session_label = QLabel("Session start time (NY)")
        self._session = QTimeEdit(QTime(9, 30))
        self._session.setDisplayFormat("HH:mm")
        row.addWidget(self._session_label)
        row.addWidget(self._session)
        row.addSpacing(20)

        row.addWidget(QLabel("Candles after exit"))
        self._after = QSpinBox()
        self._after.setRange(0, 390)
        self._after.setValue(10)
        row.addWidget(self._after)
        row.addStretch()
        outer.addLayout(row)

        self._fixed.toggled.connect(self._sync_visibility)
        for w in (self._fixed, self._candles):
            w.toggled.connect(lambda _=None: self.settingsChanged.emit())
        self._before.valueChanged.connect(lambda _=None: self.settingsChanged.emit())
        self._session.timeChanged.connect(lambda _=None: self.settingsChanged.emit())
        self._after.valueChanged.connect(lambda _=None: self.settingsChanged.emit())
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        fixed = self._fixed.isChecked()
        self._session_label.setVisible(fixed)
        self._session.setVisible(fixed)
        self._before_label.setVisible(not fixed)
        self._before.setVisible(not fixed)

    def settings(self) -> dict:
        if self._fixed.isChecked():
            return {
                "view_mode":          MODE_FIXED,
                "candles_before":     None,
                "session_start_time": self._session.time().toPython(),
                "candles_after":      int(self._after.value()),
            }
        return {
            "view_mode":          MODE_CANDLES,
            "candles_before":     int(self._before.value()),
            "session_start_time": None,
            "candles_after":      int(self._after.value()),
        }
