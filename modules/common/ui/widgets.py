"""
Small shared widgets used across every module window — the app's design
system (their looks live in theme.py, keyed by objectName):

  Card / wrap_card    rounded surface panel that groups related controls —
                      the main thing separating "designed" from "raw"
  MetricTile          the st.metric analog — caption over a big value
  SectionHeader       accent-barred section title (st.subheader analog)
  Caption             muted small-print label (st.caption)
  Banner              inline colored info/success/warning/error strip
  CollapsibleSection  the st.expander analog (header + card-styled body)
  ProgressLogPanel    progress bar + rolling log (the transforms/optimizer
                      progress pattern, incl. the 0.25 s repaint throttle and
                      the deque(200) log cap — final call always paints)
"""

import time
from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout,
                               QPlainTextEdit, QProgressBar, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from . import theme


class Card(QFrame):
    """Rounded surface panel (QSS: QFrame#card). Add content via .body."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 14, 18, 14)
        self.body.setSpacing(10)


def wrap_card(content) -> Card:
    """Wrap a QLayout or QWidget in a Card."""
    card = Card()
    if isinstance(content, QLayout):
        card.body.addLayout(content)
    else:
        card.body.addWidget(content)
    return card


class MetricTile(QFrame):
    """Caption on top, large value below — the st.metric analog."""

    def __init__(self, label: str, value: str = "—", help_text: str = "",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("metricTile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 9)
        lay.setSpacing(2)

        self._caption = QLabel(label)
        self._caption.setObjectName("metricCaption")
        self._value = QLabel(str(value))
        self._value.setObjectName("metricValue")
        self._value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self._caption)
        lay.addWidget(self._value)
        if help_text:
            self.setToolTip(help_text)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: str) -> None:
        self._value.setText(str(value))


class SectionHeader(QLabel):
    """Section title with the accent bar on the left (QSS: #sectionHeader)."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("sectionHeader")


class Caption(QLabel):
    """st.caption analog — small muted text, wraps."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")


class Banner(QLabel):
    """Inline message strip — st.info/success/warning/error analog.
    (bg tint, readable fg, and a solid edge stripe on the left)."""

    _COLORS = {
        "info":    ("#14283f", "#8ecbff", "#3f83c9"),
        "success": ("#12301b", "#7fe39a", "#2f9e53"),
        "warning": ("#382d10", "#ffd479", "#c99a35"),
        "error":   ("#3b1515", "#ff9d9d", "#c94444"),
    }

    def __init__(self, kind: str = "info", text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_kind(kind)
        self.setVisible(bool(text))

    def set_kind(self, kind: str) -> None:
        bg, fg, edge = self._COLORS.get(kind, self._COLORS["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 7px; "
            f"border-left: 3px solid {edge}; padding: 9px 13px;")

    def show_message(self, kind: str, text: str) -> None:
        self.set_kind(kind)
        self.setText(text)
        self.setVisible(True)

    def clear_message(self) -> None:
        self.setText("")
        self.setVisible(False)


class CollapsibleSection(QWidget):
    """st.expander analog: a toggle header + a card-styled body."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._button.setStyleSheet(
            f"QToolButton {{ background: {theme.SURFACE_2}; border: 1px solid "
            f"{theme.BORDER}; border-radius: 8px; padding: 8px 12px; "
            f"text-align: left; font-weight: 600; }} "
            f"QToolButton:hover {{ border-color: {theme.BORDER_LIGHT}; }} "
            f"QToolButton:checked {{ border-bottom-left-radius: 0; "
            f"border-bottom-right-radius: 0; }}")
        self._button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button.toggled.connect(self._on_toggled)

        # the body reads as the lower half of one card (QSS: #cardBody)
        self.content = QFrame()
        self.content.setObjectName("cardBody")
        self.content.setStyleSheet(
            "QFrame#cardBody { border-top: none; "
            "border-top-left-radius: 0; border-top-right-radius: 0; }")
        self.content.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(14, 12, 14, 12)

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
        self._log.setObjectName("console")
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
