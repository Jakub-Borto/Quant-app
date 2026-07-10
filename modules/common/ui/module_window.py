"""
ModuleWindowBase — shared scaffold for every module window.

Provides:
- a scrollable content area (module pages are long, like the old Streamlit
  pages) with a title + caption header;
- background-worker tracking: track_worker() keeps a strong reference (a
  GC'd QRunnable dies silently) and powers has_running_jobs();
- the close policy: closing a window with a running job asks first, then
  cancels cancellable workers (non-cancellable ones finish in background —
  they're pool threads, harmless at window level).

Every module window subclasses this and builds its page inside self.content
(a QVBoxLayout).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QLabel, QMessageBox, QScrollArea,
                               QVBoxLayout, QWidget)

from .workers import FunctionWorker, start_worker


class ModuleWindowBase(QWidget):
    def __init__(self, settings, title: str, caption: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._active_workers: set[FunctionWorker] = set()

        self.resize(1240, 860)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page = QWidget()
        scroll.setWidget(page)
        outer.addWidget(scroll)

        self.content = QVBoxLayout(page)
        # generous side margins — module windows open maximized, and content
        # glued to the screen edges reads badly
        self.content.setContentsMargins(48, 22, 48, 30)
        self.content.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("pageTitle")
        caption_lbl = QLabel(caption)
        caption_lbl.setObjectName("pageCaption")
        accent = QFrame()
        accent.setObjectName("accentBar")
        accent.setFixedSize(46, 3)
        self.content.addWidget(title_lbl)
        self.content.addWidget(caption_lbl)
        self.content.addWidget(accent)
        self.content.addSpacing(8)

    # ── workers ───────────────────────────────────────────────────────────────
    def track_worker(self, worker: FunctionWorker) -> None:
        """Keep the worker alive, auto-forget on any terminal signal, start it."""
        self._active_workers.add(worker)
        for sig in (worker.signals.finished, worker.signals.error,
                    worker.signals.cancelled):
            sig.connect(lambda *_a, w=worker: self._active_workers.discard(w))
        start_worker(worker)

    def has_running_jobs(self) -> bool:
        return bool(self._active_workers)

    def cancel_all_jobs(self) -> None:
        for w in list(self._active_workers):
            w.cancel()

    # ── close policy ──────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        if self.has_running_jobs():
            answer = QMessageBox.question(
                self, "Job running",
                "A job is still running. Cancel it and close this window?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel_all_jobs()
        event.accept()
