"""
Checkbox filter rows: trade-type filter (dynamic values) and day-type filter
(driven by DAY_TYPE_ORDER) — the Qt analog of the old checkbox rows in the
backtester view and the optimizer cell detail.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QGridLayout, QWidget

from modules.common.backend.trade_stats import DAY_TYPE_ORDER


class CheckboxFilterRow(QWidget):
    """A row of labeled checkboxes. items = [(tag, label), ...]."""

    selectionChanged = Signal()

    def __init__(self, items: list[tuple[str, str]],
                 checked_tags: set | None = None, per_row: int = 7,
                 parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        self._boxes: dict[str, QCheckBox] = {}
        for n, (tag, label) in enumerate(items):
            box = QCheckBox(label)
            box.setChecked(checked_tags is None or tag in checked_tags)
            box.toggled.connect(self.selectionChanged)
            grid.addWidget(box, n // per_row, n % per_row)
            self._boxes[tag] = box

    def selected(self) -> list[str]:
        """Checked tags, in the items' original order."""
        return [tag for tag, box in self._boxes.items() if box.isChecked()]

    def all_selected(self) -> bool:
        return all(box.isChecked() for box in self._boxes.values())


def make_trade_type_filter(trade_types: list[str],
                           per_row: int = 6) -> CheckboxFilterRow:
    """Filter row over the distinct trade_type values (all checked)."""
    return CheckboxFilterRow([(t, t) for t in trade_types], per_row=per_row)


def make_day_type_filter(checked_tags: set | None = None) -> CheckboxFilterRow:
    """Filter row over DAY_TYPE_ORDER (all checked unless told otherwise)."""
    return CheckboxFilterRow(list(DAY_TYPE_ORDER), checked_tags=checked_tags,
                             per_row=len(DAY_TYPE_ORDER))
