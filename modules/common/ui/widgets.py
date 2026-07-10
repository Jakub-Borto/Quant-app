"""
Small shared widgets used across every module window:

  MetricTile          the st.metric analog — caption over a big value
  SectionHeader       an st.subheader analog
  Caption             muted small-print label (st.caption)
  Banner              inline colored info/success/warning/error strip
  CollapsibleSection  the st.expander analog
  ProgressLogPanel    progress bar + rolling log (the transforms/optimizer
                      progress pattern, incl. the 0.25 s repaint throttle and
                      the deque(200) log cap — final call always paints)
"""

import time
from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from . import theme


class MetricTile(QFrame):
    """Caption on top, large value below — the st.metric analog."""

    def __init__(self, label: str, value: str = "—", help_text: str = "",
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: {theme.SURFACE}; border: 1px solid "
            f"{theme.BORDER}; border-radius: 8px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(2)

        self._caption = QLabel(label)
        self._caption.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 11px; border: none;")
        self._value = QLabel(str(value))
        self._value.setStyleSheet(
            "font-size: 17px; font-weight: 600; border: none;")
        self._value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self._caption)
        lay.addWidget(self._value)
        if help_text:
            self.setToolTip(help_text)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: str) -> None:
        self._value.setText(str(value))


class SectionHeader(QLabel):
    """st.subheader analog."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "font-size: 17px; font-weight: 600; margin-top: 10px;")


class Caption(QLabel):
    """st.caption analog — small muted text, wraps."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")


class Banner(QLabel):
    """Inline message strip — st.info/success/warning/error analog."""

    _COLORS = {
        "info":    ("#173a5e", "#8ecbff"),
        "success": ("#14401f", "#7fe39a"),
        "warning": ("#4a3a12", "#ffd479"),
        "error":   ("#4a1717", "#ff9d9d"),
    }

    def __init__(self, kind: str = "info", text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_kind(kind)
        self.setVisible(bool(text))

    def set_kind(self, kind: str) -> None:
        bg, fg = self._COLORS.get(kind, self._COLORS["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; "
            f"padding: 8px 12px;")

    def show_message(self, kind: str, text: str) -> None:
        self.set_kind(kind)
        self.setText(text)
        self.setVisible(True)

    def clear_message(self) -> None:
        self.setText("")
        self.setVisible(False)


class CollapsibleSection(QWidget):
    """st.expander analog: a toggle header + a hideable content area."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._button.setStyleSheet(
            f"QToolButton {{ background: {theme.SURFACE}; border: 1px solid "
            f"{theme.BORDER}; border-radius: 6px; padding: 7px 10px; "
            f"text-align: left; }}")
        self._button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button.toggled.connect(self._on_toggled)

        self.content = QWidget()
        self.content.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 8, 10, 8)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._button)
        lay.addWidget(self.content)

    def _on_toggled(self, checked: bool) -> None:
        self._button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

    def add_widget(self, w: QWidget) -> None:
        self.content_layout.addWidget(w)

    def set_expanded(self, expanded: bool) -> None:
        self._button.setChecked(expanded)


class ProgressLogPanel(QWidget):
    """
    QProgressBar + rolling log console. Connect a FunctionWorker's `progress`
    signal to on_progress(). Keeps the old optimizer view's behavior: log
    capped at the most recent 200 lines, repaints throttled to ~4/s, and the
    final call (current >= total) always paints.
    """

    def __init__(self, log_height: int = 260, parent=None):
        super().__init__(parent)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(log_height)
        self._logs: deque[str] = deque(maxlen=200)
        self._last_paint = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._bar)
        lay.addWidget(self._log)

    def reset(self) -> None:
        self._logs.clear()
        self._log.setPlainText("")
        self._bar.setValue(0)
        self._last_paint = 0.0

    def on_progress(self, current: int, total: int, message: str = "") -> None:
        if message:
            self._logs.append(message)
        now = time.monotonic()
        if current < total and now - self._last_paint < 0.25:
            return
        self._last_paint = now
        self._bar.setValue(int(1000 * (current / total)) if total else 1000)
        self._log.setPlainText("\n".join(self._logs))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def log_line(self, message: str) -> None:
        """Message-only entry point (the combiner's log(msg) callback)."""
        self.on_progress(0, 1, message)


def hline() -> QFrame:
    """Thin horizontal separator (st.divider analog)."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {theme.BORDER};")
    return line
