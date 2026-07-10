"""
Auto-generated parameter form from a plugin's PARAMS dict — the Qt analog of
the old views' param-widget rendering.

Type dispatch (verbatim ORDER from views/backtester._render_param_widget —
bool is checked BEFORE int because bool is an int subclass):

    bool  -> QCheckBox
    float -> QDoubleSpinBox(step 0.1, 2 decimals)
    int   -> QSpinBox(step 1)
    str   -> QLineEdit
    other -> warning label; the default value is passed through unchanged

Layout mirrors the old views: with PARAM_SECTIONS, one captioned row per
section (params not in any section land under "Other"); without sections,
rows of up to 10. `numeric_only=True` reproduces Monte Carlo's
_param_widgets: ONLY int/float params get widgets and only those keys appear
in values().

IMPORTANT: spin boxes get explicit wide ranges — Qt's defaults (0..99 /
0..99.99) would silently clamp real values like account_size=100000, which
would be a logic change.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QGridLayout, QLabel,
                               QLineEdit, QSpinBox, QVBoxLayout, QWidget)

from . import theme

INT_RANGE = (-1_000_000_000, 1_000_000_000)
FLOAT_RANGE = (-1e12, 1e12)


def make_param_widget(default):
    """One widget for one param default. Returns (widget, getter) — getter()
    yields the current value with the same type family as the default."""
    if isinstance(default, bool):
        w = QCheckBox()
        w.setChecked(default)
        return w, w.isChecked
    elif isinstance(default, float):
        w = QDoubleSpinBox()
        w.setRange(*FLOAT_RANGE)
        w.setSingleStep(0.1)
        w.setDecimals(2)
        w.setValue(default)
        return w, w.value
    elif isinstance(default, int):
        w = QSpinBox()
        w.setRange(*INT_RANGE)
        w.setSingleStep(1)
        w.setValue(default)
        return w, w.value
    elif isinstance(default, str):
        w = QLineEdit(default)
        return w, w.text
    return None, None   # unsupported type — caller shows a warning


class ParamsForm(QWidget):
    """The whole form. values() returns {param: current value}."""

    def __init__(self, params: dict, sections: dict | None = None,
                 hidden=frozenset(), excluded=frozenset(),
                 numeric_only: bool = False, per_row: int = 10, parent=None):
        super().__init__(parent)
        self._getters: dict[str, callable] = {}
        self._passthrough: dict[str, object] = {}

        visible = {k: v for k, v in params.items()
                   if k not in hidden and k not in excluded}
        if numeric_only:
            # MC semantics (_param_widgets): only int/float defaults get a
            # widget and only those keys appear in values(). NOTE: the old
            # chain checked isinstance(default, int) FIRST, so a bool default
            # rendered as a 0/1 number input — bool is an int subclass. We
            # keep that quirk (int(default)) for exact parity.
            visible = {k: (int(v) if isinstance(v, bool) else v)
                       for k, v in visible.items()
                       if isinstance(v, (int, float))}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        def add_row_grid(keys: list[str], caption: str | None):
            if caption:
                cap = QLabel(caption)
                cap.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")
                outer.addWidget(cap)
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            for col, key in enumerate(keys):
                default = visible[key]
                widget, getter = make_param_widget(default)
                name = QLabel(key)
                name.setStyleSheet("font-size: 12px;")
                grid.addWidget(name, 0, col, alignment=Qt.AlignBottom)
                if widget is None:
                    warn = QLabel(f"unsupported type: {type(default).__name__}")
                    warn.setStyleSheet(f"color: {theme.WARN}; font-size: 11px;")
                    grid.addWidget(warn, 1, col)
                    self._passthrough[key] = default
                else:
                    grid.addWidget(widget, 1, col)
                    self._getters[key] = getter
            outer.addLayout(grid)

        if sections and not numeric_only:
            rendered = set()
            for section_label, keys in sections.items():
                section_keys = [k for k in keys if k in visible]
                if not section_keys:
                    continue
                add_row_grid(section_keys, section_label)
                rendered.update(section_keys)
            unassigned = [k for k in visible if k not in rendered]
            if unassigned:
                add_row_grid(unassigned, "Other")
        else:
            items = list(visible.keys())
            for i in range(0, len(items), per_row):
                add_row_grid(items[i:i + per_row], None)

    def values(self) -> dict:
        out = {k: g() for k, g in self._getters.items()}
        out.update(self._passthrough)
        return out
