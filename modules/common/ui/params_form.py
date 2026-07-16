"""
Auto-generated parameter form from a plugin's PARAMS dict.

═══════════════════════════════════════════════════════════════════════════
 HOW TO DECLARE PLUGIN PARAMS (strategies, data transforms, position sizers)
═══════════════════════════════════════════════════════════════════════════

Every plugin exposes a module-level  PARAMS = {name: default}  dict. The
DEFAULT VALUE'S TYPE decides both the widget rendered here AND how the
Optimizer can sweep the param (the sweep twin of these rules lives in
modules/optimizer/backend/param_space.py — keep the two in sync):

    bool   -> checkbox.               Optimizer: swept over [False, True].
              Use real True/False for on/off switches — NOT 0/1 ints.
    float  -> decimal spinbox.        Optimizer: min/max/step range.
    int    -> integer spinbox.        Optimizer: min/max/step range.
    str    -> free-text box.          Optimizer: comma-separated value list.
              Times like "09:30" are plain str params — there is no special
              time widget; parse/validate them inside the plugin.
    other  -> "unsupported type" warning label; the default is passed
              through values() unchanged and cannot be edited or swept.

Optionally declare  PARAMS_OPTIONS = {name: [choice, ...]}  next to PARAMS
to upgrade a param to one of two richer widgets:

  DROPDOWN — when the default IS one of the declared choices:

      PARAMS         = {"vwap_session": "globex"}
      PARAMS_OPTIONS = {"vwap_session": ["globex", "rth"]}

    Choices may be str, int or float; values() returns the chosen option
    with its ORIGINAL type (picking 2 from [2, 3] yields the int 2).
    Optimizer: swept over a user-checked subset of the choices.

  BIT-FLAG GROUP — when the default is a '0'/'1' string exactly as long as
  the options list (one character per option, same order):

      PARAMS         = {"valid_entries": "1111100"}
      PARAMS_OPTIONS = {"valid_entries": ["absorption_delta", ...7 names]}

    Renders one named checkbox per option; values() returns the bitstring
    back ("1010100"), so plugin code that parses flag strings needs no
    change. Optimizer: swept over a comma-separated list of bitstrings.

  Precedence: a bool default ALWAYS renders as a checkbox (its options
  entry is ignored — True == 1 would false-match int choice lists); the
  dropdown rule wins over the bit-flag rule (a default like "1" with
  options ["0", "1"] is a dropdown); a PARAMS_OPTIONS entry whose param
  fits neither rule is ignored and the plain type widget is used.

  PARAMS_OPTIONS constrains the UI only — it does NOT validate what run()
  receives. Keep the choice lists in sync with what the plugin code
  actually dispatches on.

═══════════════════════════════════════════════════════════════════════════

Layout: with PARAM_SECTIONS every section is a collapsible drop-down box
titled with the section label (params not in any section land under
"Other"); without sections, plain rows of up to 10. Collapsed sections still
report their values — the widgets exist either way. `numeric_only=True`
reproduces Monte Carlo's
_param_widgets: ONLY int/float params get widgets and only those keys appear
in values(); a bool default is coerced to a 0/1 number input (legacy parity
quirk) and PARAMS_OPTIONS is ignored entirely in this mode.

IMPORTANT: spin boxes get explicit wide ranges — Qt's defaults (0..99 /
0..99.99) would silently clamp real values like account_size=100000, which
would be a logic change.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QLabel, QLineEdit, QSpinBox,
                               QVBoxLayout, QWidget)

from . import theme
from .widgets import CollapsibleSection

INT_RANGE = (-1_000_000_000, 1_000_000_000)
FLOAT_RANGE = (-1e12, 1e12)


def _is_flags(default, options) -> bool:
    """True when (default, options) declare a bit-flag param (see the module
    docstring). Qt-free twin: param_space.is_flags — keep in sync."""
    return (isinstance(default, str) and bool(options)
            and len(default) == len(options) and set(default) <= {"0", "1"})


class FlagsGroup(QWidget):
    """Named checkbox group for a bit-flag string param: one checkbox per
    option name, seeded from the default bitstring; value() returns the
    bitstring back ('1' per checked box, in declared option order)."""

    # named `toggled` so callers wiring change signals via
    # hasattr(widget, "toggled") (sweep_panel._ParamCell) catch it like a
    # plain QCheckBox
    toggled = Signal()

    def __init__(self, names: list, bits: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._boxes: list[QCheckBox] = []
        for name, bit in zip(names, bits):
            cb = QCheckBox(str(name))
            cb.setChecked(bit == "1")
            cb.toggled.connect(self.toggled)
            lay.addWidget(cb)
            self._boxes.append(cb)

    def value(self) -> str:
        return "".join("1" if cb.isChecked() else "0" for cb in self._boxes)


def make_param_widget(default, options: list | None = None):
    """One widget for one param default. Returns (widget, getter) — getter()
    yields the current value with the same type family as the default.
    `options` is the param's PARAMS_OPTIONS entry (None when undeclared);
    see the module docstring for the dropdown / bit-flag rules."""
    if isinstance(default, bool):
        w = QCheckBox()
        w.setChecked(default)
        return w, w.isChecked
    elif options and default in options:
        w = QComboBox()
        for opt in options:
            w.addItem(str(opt), opt)    # userData keeps the ORIGINAL typed value
        w.setCurrentIndex(options.index(default))
        return w, w.currentData
    elif _is_flags(default, options):
        w = FlagsGroup(options, default)
        return w, w.value
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
                 numeric_only: bool = False, per_row: int = 10,
                 options: dict | None = None, parent=None):
        super().__init__(parent)
        self._getters: dict[str, callable] = {}
        self._passthrough: dict[str, object] = {}
        self._options = options or {}   # the plugin's PARAMS_OPTIONS dict

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

        def build_grid(keys: list[str]) -> QWidget:
            """One widget holding the param name/editor grid (rows of per_row)."""
            box = QWidget()
            grid = QGridLayout(box)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            for n, key in enumerate(keys):
                row, col = (n // per_row) * 2, n % per_row
                default = visible[key]
                widget, getter = make_param_widget(default, self._options.get(key))
                name = QLabel(key)
                name.setStyleSheet("font-size: 12px;")
                grid.addWidget(name, row, col, alignment=Qt.AlignBottom)
                if widget is None:
                    warn = QLabel(f"unsupported type: {type(default).__name__}")
                    warn.setStyleSheet(f"color: {theme.WARN}; font-size: 11px;")
                    grid.addWidget(warn, row + 1, col)
                    self._passthrough[key] = default
                else:
                    grid.addWidget(widget, row + 1, col)
                    self._getters[key] = getter
            return box

        if sections and not numeric_only:
            # each PARAM_SECTIONS section is a collapsible drop-down box;
            # collapsed by default when there are several (a strategy like ivb
            # has ~18 sections — showing all at once floods the page), expanded
            # when the strategy only has one
            rendered = set()
            section_specs = []
            for section_label, keys in sections.items():
                section_keys = [k for k in keys if k in visible]
                if section_keys:
                    section_specs.append((section_label, section_keys))
                    rendered.update(section_keys)
            unassigned = [k for k in visible if k not in rendered]
            if unassigned:
                section_specs.append(("Other", unassigned))

            expanded = len(section_specs) == 1
            for section_label, keys in section_specs:
                box = CollapsibleSection(section_label, expanded=expanded)
                box.add_widget(build_grid(keys))
                outer.addWidget(box)
        else:
            items = list(visible.keys())
            if items:
                outer.addWidget(build_grid(items))

    def values(self) -> dict:
        out = {k: g() for k, g in self._getters.items()}
        out.update(self._passthrough)
        return out
