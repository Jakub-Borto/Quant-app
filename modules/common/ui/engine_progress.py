"""
EngineProgressPanel — live view of a modules.engine.run_strategy() run.

The engine calls on_progress(current_day, total_days, text) at EVERY step of
every day (see modules/engine/runner.py::_Progress): the text's first line is
what the engine is doing right now ("Day 12 of 370 · 2025-05-06 — preparing
the day (prepare_day)"), the next lines are the read-ahead, the counters so
far and the cache / elapsed / time-left line. That can be thousands of calls a
second on a warm run, so the panel only STORES the latest call and a timer
paints it ~16 times a second (the last state is always painted on finish).

After the run, finish() shows the engine's one-paragraph summary and fills the
collapsible "Engine timing" table (RunResult.timing).
"""

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget

from . import theme
from .widgets import CollapsibleSection

PAINT_MS = 60


class EngineProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(True)
        self._now = QLabel()
        self._now.setWordWrap(True)
        self._now.setStyleSheet("font-weight: 600;")
        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")

        self._timing = CollapsibleSection("Engine timing of the last run")
        self._timing_text = QPlainTextEdit()
        self._timing_text.setObjectName("console")
        self._timing_text.setReadOnly(True)
        self._timing_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._timing_text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._timing_text.setMinimumHeight(260)
        self._timing.add_widget(self._timing_text)
        self._timing.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self._bar)
        lay.addWidget(self._now)
        lay.addWidget(self._detail)
        lay.addWidget(self._timing)

        self._latest: tuple | None = None
        self._painted: tuple | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(PAINT_MS)
        self._timer.timeout.connect(self._paint)
        self._bar.setVisible(False)
        self._now.setVisible(False)
        self._detail.setVisible(False)

    # ── run lifecycle ────────────────────────────────────────────────────────
    def start(self, headline: str) -> None:
        self._latest = self._painted = None
        self._bar.setValue(0)
        self._bar.setFormat("starting…")
        self._now.setText(headline)
        self._detail.setText("")
        for w in (self._bar, self._now, self._detail):
            w.setVisible(True)
        self._timer.start()

    def on_progress(self, current: int, total: int, text: str = "") -> None:
        """Slot for FunctionWorker.signals.progress — store only (see module doc)."""
        self._latest = (current, total, text)

    def finish(self, summary: str, timing: str = "") -> None:
        self._timer.stop()
        self._bar.setValue(1000)
        self._bar.setFormat("done")
        self._now.setText("Finished")
        self._detail.setText(summary)
        self._timing_text.setPlainText(timing or "(nothing was timed)")
        self._timing.setVisible(bool(timing))

    def fail(self, headline: str) -> None:
        """Error or cancel: keep the last engine state visible under `headline`."""
        self._timer.stop()
        self._paint()
        self._bar.setFormat("stopped")
        self._now.setText(headline)

    def hide_progress(self) -> None:
        self._timer.stop()
        for w in (self._bar, self._now, self._detail):
            w.setVisible(False)

    # ── painting ─────────────────────────────────────────────────────────────
    def _paint(self) -> None:
        latest = self._latest
        if latest is None or latest is self._painted:
            return
        self._painted = latest
        current, total, text = latest
        if total > 0:
            self._bar.setValue(int(1000 * min(current, total) / total))
            self._bar.setFormat(f"day {min(current, total)} / {total}  (%p%)")
        lines = text.splitlines() if text else []
        if lines:
            self._now.setText(lines[0])
            self._detail.setText("\n".join(lines[1:]))

    def current_text(self) -> tuple[str, str, str]:
        """(bar text, now line, detail) as painted — for tests."""
        return self._bar.format(), self._now.text(), self._detail.text()
