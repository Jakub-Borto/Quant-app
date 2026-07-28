"""
ModuleWindowBase — shared scaffold for every module window.

Provides:
- a scrollable content area (module pages are long, like the old Streamlit
  pages) with a title + caption header;
- add_header_action(widget): a right-aligned strip pinned ABOVE the scroll
  area, so a control like the report-layout gear stays reachable however far
  the page is scrolled. Created lazily — a module that never calls it gets no
  extra chrome;
- background-worker tracking: track_worker() keeps a strong reference (a
  GC'd QRunnable dies silently) and powers has_running_jobs();
- the close policy: closing a window with a running job asks first, then
  cancels cancellable workers (non-cancellable ones finish in background —
  they're pool threads, harmless at window level).

Every module window subclasses this and builds its page inside self.content
(a QVBoxLayout).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMessageBox,
                               QScrollArea, QVBoxLayout, QWidget)

from .widgets import pin_minimum_height
from .workers import FunctionWorker, start_worker


class ModuleWindowBase(QWidget):
    def __init__(self, settings, title: str, caption: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._active_workers: set[FunctionWorker] = set()

        self.resize(1240, 860)

        self._outer = outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._header_actions = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Vertical bar ALWAYS on: with AsNeeded, expanding a section grows the
        # page, the bar appears, the viewport narrows, every word-wrapped
        # label re-wraps taller, and the layout oscillates — the visible
        # "jiggle" when opening a dropdown. A permanent bar keeps the content
        # width constant, so a toggle is a single clean relayout.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        page = QWidget()
        # The page is resized every time a section expands/collapses, and by
        # default Qt treats a resize as "the whole widget is now invalid" —
        # every child repaints, including the params form far above the change.
        # WA_StaticContents keeps existing content valid and repaints only the
        # newly exposed strip. Measured on a 1600x1000 Backtester: 438 paint
        # events per dropdown toggle without it, 72 with it.
        page.setAttribute(Qt.WA_StaticContents, True)
        scroll.setWidget(page)
        outer.addWidget(scroll)

        self.content = QVBoxLayout(page)
        # generous side margins — module windows open maximized, and content
        # glued to the screen edges reads badly
        self.content.setContentsMargins(48, 22, 48, 30)
        self.content.setSpacing(10)
        # Part of the "everything squashes for a frame" fix: the page's
        # minimumHeight is 0 by default, so nothing stops it being laid out
        # shorter than its content while the scroll area catches up — and a
        # QVBoxLayout squeezed like that crushes every child past its own
        # minimum. See pin_minimum_height for the traced mechanism.
        pin_minimum_height(page)

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

    # ── pinned header actions ─────────────────────────────────────────────────
    def add_header_action(self, widget: QWidget) -> None:
        """Pin a control to a right-aligned strip above the scroll area (so it
        does not scroll away with the page)."""
        if self._header_actions is None:
            self._header_actions = QHBoxLayout()
            self._header_actions.setContentsMargins(48, 14, 48, 0)
            self._header_actions.addStretch()
            self._outer.insertLayout(0, self._header_actions)
        self._header_actions.addWidget(widget, alignment=Qt.AlignTop)

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
