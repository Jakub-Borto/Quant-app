"""
Application-wide wheel guard.

Problem: on a scrollable page, QComboBox / spin boxes / sliders / date edits
swallow the mouse wheel — scrolling the page silently CHANGES VALUES the
cursor happens to pass over. (QComboBox and QAbstractSpinBox react to the
wheel even without focus.)

Fix (one filter installed on the QApplication):
- QEvent.Polish: any such widget created with the default WheelFocus policy
  is downgraded to StrongFocus, so the wheel can never *give* it focus.
- QEvent.Wheel on an UNFOCUSED such widget: the event is redirected to the
  enclosing scroll area's viewport, so the page scrolls like the user
  expected. (Sent directly — Qt only propagates *spontaneous* wheel events up
  the parent chain, so a plain re-send to the parent would die in the first
  container.) A widget the user explicitly clicked (focused) still responds
  to the wheel normally.

QTabBar is included so hovering the Optimizer's tab strip doesn't switch
tabs while scrolling.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox,
                               QApplication, QComboBox, QSlider, QTabBar)

_WHEELABLE = (QAbstractSpinBox, QComboBox, QSlider, QTabBar)


class WheelGuard(QObject):
    def eventFilter(self, obj, event) -> bool:
        etype = event.type()
        if etype == QEvent.Polish and isinstance(obj, _WHEELABLE):
            if obj.focusPolicy() == Qt.WheelFocus:
                obj.setFocusPolicy(Qt.StrongFocus)
            return False
        if etype == QEvent.Wheel and isinstance(obj, _WHEELABLE) \
                and not obj.hasFocus():
            scroller = obj.parentWidget()
            while scroller is not None \
                    and not isinstance(scroller, QAbstractScrollArea):
                scroller = scroller.parentWidget()
            if scroller is not None:
                QApplication.sendEvent(scroller.viewport(), event)
            return True
        return False


def install_wheel_guard(app: QApplication) -> None:
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    app._wheel_guard = guard   # keep a strong reference
