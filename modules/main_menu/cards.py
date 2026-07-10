"""
ModuleCard — one clickable launcher card: muted index number, title, blurb,
accent "Open →" affordance, hover glow. Whole card is clickable and emits
clicked.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QLabel, QVBoxLayout

from modules.common.ui import theme


class ModuleCard(QFrame):
    clicked = Signal()

    _BASE = (f"QFrame#card {{ background: {theme.SURFACE}; "
             f"border: 1px solid {theme.BORDER}; border-radius: 12px; }}")
    _HOVER = (f"QFrame#card {{ background: {theme.SURFACE_2}; "
              f"border: 1px solid {theme.ACCENT}; border-radius: 12px; }}")

    def __init__(self, number: str, title: str, blurb: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(self._BASE)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMinimumHeight(150)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)

        num = QLabel(number)
        num.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 26px; font-weight: 700; "
            f"border: none; background: transparent;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-size: 17px; font-weight: 600; border: none; "
            "background: transparent;")
        blurb_lbl = QLabel(blurb)
        blurb_lbl.setWordWrap(True)
        blurb_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 12px; border: none; "
            f"background: transparent;")
        open_lbl = QLabel("Open →")
        open_lbl.setStyleSheet(
            f"color: {theme.ACCENT_HOVER}; font-size: 13px; font-weight: 600; "
            f"border: none; background: transparent;")

        lay.addWidget(num)
        lay.addWidget(title_lbl)
        lay.addWidget(blurb_lbl)
        lay.addStretch()
        lay.addWidget(open_lbl)

        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(24)
        self._shadow.setOffset(0, 4)
        self._shadow.setColor(Qt.transparent)
        self.setGraphicsEffect(self._shadow)

    # ── interactions ──────────────────────────────────────────────────────────
    def enterEvent(self, event) -> None:
        self.setStyleSheet(self._HOVER)
        from PySide6.QtGui import QColor
        glow = QColor(theme.ACCENT)
        glow.setAlpha(90)
        self._shadow.setColor(glow)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.setStyleSheet(self._BASE)
        self._shadow.setColor(Qt.transparent)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)
