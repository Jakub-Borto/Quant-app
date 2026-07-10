"""
SweepPanel — the Optimizer's parameter panel.

The PySide6 port of render_param_panel / _render_sweep_inputs /
render_role_assignment from legacy_streamlit/views/optimizer.py:

- every visible strategy param gets either a sweep checkbox (when its default
  type is sweepable per param_space.sweep_kind) or a plain fixed-value widget;
- a checked param shows its sweep editor — min/max/step for int/float
  (values via build_range), a comma-separated list for str (parse_values) —
  plus the live values-preview caption / inline error;
- at most MAX_SWEPT params can be checked (others grey out at the cap);
- sweep order = selection order (survivors keep their rank);
- the role row assigns X / Y / Slider 1 / Slider 2 with distinct-role
  validation.

Emits changed on any edit so the tab can refresh the live combo readout.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QSpinBox, QVBoxLayout, QWidget)

from modules.common.backend.asset_info import HIDDEN_PARAMS
from modules.common.ui import theme
from modules.common.ui.params_form import make_param_widget
from modules.common.ui.widgets import Caption
from modules.optimizer.backend.param_space import (MAX_SWEPT, ROLE_LABELS,
                                                   ROLES, build_range,
                                                   parse_values, sweep_kind)


def _values_preview(values: list) -> str:
    fmt = [f"{v:g}" if isinstance(v, float) else str(v) for v in values]
    if len(fmt) > 8:
        fmt = fmt[:4] + ["…"] + fmt[-2:]
    return f"{len(values)} value{'s' if len(values) != 1 else ''}: " + ", ".join(fmt)


class _SweepEditor(QWidget):
    """min/max/step (numeric) or comma-list (categorical) editor for one param.
    values() -> list, or None on invalid input (error caption shown inline)."""

    edited = Signal()

    def __init__(self, param: str, default, kind: str, parent=None):
        super().__init__(parent)
        self._kind = kind
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        if kind == "categorical":
            self._text = QLineEdit(str(default))
            self._text.setToolTip("comma-separated values to test")
            self._text.textChanged.connect(self._refresh)
            lay.addWidget(self._text)
        else:
            row = QHBoxLayout()
            if kind == "int":
                self._lo = QSpinBox(); self._hi = QSpinBox(); self._step = QSpinBox()
                for w in (self._lo, self._hi):
                    w.setRange(-1_000_000_000, 1_000_000_000)
                self._step.setRange(1, 1_000_000_000)
                self._lo.setValue(int(default)); self._hi.setValue(int(default))
                self._step.setValue(1)
            else:
                self._lo = QDoubleSpinBox(); self._hi = QDoubleSpinBox()
                self._step = QDoubleSpinBox()
                for w in (self._lo, self._hi):
                    w.setRange(-1e12, 1e12); w.setDecimals(6); w.setSingleStep(0.1)
                self._step.setRange(0.0, 1e12); self._step.setDecimals(6)
                self._step.setSingleStep(0.05)
                self._lo.setValue(float(default)); self._hi.setValue(float(default))
                self._step.setValue(0.1)
            for label, w in (("min", self._lo), ("max", self._hi),
                             ("step", self._step)):
                box = QVBoxLayout(); box.setSpacing(0)
                cap = QLabel(label)
                cap.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px;")
                box.addWidget(cap); box.addWidget(w)
                row.addLayout(box)
                w.valueChanged.connect(self._refresh)
            lay.addLayout(row)

        self._caption = Caption("")
        lay.addWidget(self._caption)
        self._refresh()

    def values(self) -> list | None:
        try:
            if self._kind == "categorical":
                return parse_values(self._text.text())
            return build_range(self._lo.value(), self._hi.value(),
                               self._step.value(), self._kind)
        except ValueError:
            return None

    def _refresh(self) -> None:
        try:
            if self._kind == "categorical":
                values = parse_values(self._text.text())
            else:
                values = build_range(self._lo.value(), self._hi.value(),
                                     self._step.value(), self._kind)
        except ValueError as e:
            self._caption.setText(f"⚠ {e}")
            self._caption.setStyleSheet(f"color: {theme.WARN}; font-size: 12px;")
        else:
            self._caption.setText(_values_preview(values))
            self._caption.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")
        self.edited.emit()


class _ParamCell(QWidget):
    """One param's cell: sweep checkbox (if sweepable) + either the sweep
    editor or the fixed-value widget."""

    toggled = Signal()
    edited = Signal()

    def __init__(self, param: str, default, kind: str | None, parent=None):
        super().__init__(parent)
        self.param = param
        self.kind = kind
        self._default = default

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        if kind is not None:
            self.check = QCheckBox(param)
            self.check.setToolTip("sweep this param")
            f = self.check.font(); f.setBold(True); self.check.setFont(f)
            self.check.toggled.connect(self._on_toggled)
            lay.addWidget(self.check)
        else:
            self.check = None
            lay.addWidget(QLabel(param))

        self._holder = QVBoxLayout()
        lay.addLayout(self._holder)
        self.sweep_editor: _SweepEditor | None = None
        self._fixed_getter = None
        self._show_fixed()

    # ── swap between fixed widget and sweep editor ────────────────────────────
    def _clear_holder(self) -> None:
        while self._holder.count():
            item = self._holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _show_fixed(self) -> None:
        self._clear_holder()
        self.sweep_editor = None
        widget, getter = make_param_widget(self._default)
        if widget is None:
            warn = QLabel(f"unsupported type: {type(self._default).__name__}")
            warn.setStyleSheet(f"color: {theme.WARN}; font-size: 11px;")
            self._holder.addWidget(warn)
            self._fixed_getter = lambda: self._default
        else:
            self._holder.addWidget(widget)
            self._fixed_getter = getter
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(lambda _=None: self.edited.emit())
            elif hasattr(widget, "textChanged"):
                widget.textChanged.connect(lambda _=None: self.edited.emit())
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(lambda _=None: self.edited.emit())

    def _show_sweep(self) -> None:
        self._clear_holder()
        self._fixed_getter = None
        self.sweep_editor = _SweepEditor(self.param, self._default, self.kind)
        self.sweep_editor.edited.connect(self.edited)
        self._holder.addWidget(self.sweep_editor)

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            self._show_sweep()
        else:
            self._show_fixed()
        self.toggled.emit()

    # ── state ─────────────────────────────────────────────────────────────────
    @property
    def is_swept(self) -> bool:
        return self.check is not None and self.check.isChecked()

    def fixed_value(self):
        return self._fixed_getter() if self._fixed_getter else self._default

    def swept_values(self) -> list | None:
        return self.sweep_editor.values() if self.sweep_editor else None


class SweepPanel(QWidget):
    changed = Signal()

    def __init__(self, strategy, parent=None):
        super().__init__(parent)
        visible = {k: v for k, v in getattr(strategy, "PARAMS", {}).items()
                   if k not in HIDDEN_PARAMS}
        self._cells: dict[str, _ParamCell] = {}
        self._sweep_order: list[str] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(Caption(
            f"Check up to {MAX_SWEPT} params to sweep — numeric params take "
            f"min/max/step, text params a comma-separated value list; "
            f"everything else is held at the value shown."))

        # sections honoring PARAM_SECTIONS (the old _param_layout), rows of 3
        sections = self._param_layout(strategy, visible)
        for section_label, keys in sections:
            lay.addWidget(Caption(section_label))
            grid = QGridLayout()
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(10)
            for n, param in enumerate(keys):
                cell = _ParamCell(param, visible[param],
                                  sweep_kind(visible[param]))
                cell.toggled.connect(lambda p=param: self._on_cell_toggled(p))
                cell.edited.connect(self.changed)
                self._cells[param] = cell
                grid.addWidget(cell, n // 3, n % 3)
            for c in range(3):
                grid.setColumnStretch(c, 1)
            lay.addLayout(grid)

        # role assignment row
        lay.addWidget(Caption("Axis roles"))
        self._roles_row = QHBoxLayout()
        lay.addLayout(self._roles_row)
        self._role_combos: dict[str, QComboBox] = {}
        self._role_error = Caption("")
        lay.addWidget(self._role_error)
        self._rebuild_roles()

    @staticmethod
    def _param_layout(strategy, visible: dict) -> list:
        """[(section_label, [param, ...]), ...] honoring PARAM_SECTIONS."""
        if not hasattr(strategy, "PARAM_SECTIONS"):
            return [("Parameters", list(visible.keys()))]
        sections, rendered = [], set()
        for label, keys in strategy.PARAM_SECTIONS.items():
            keys = [k for k in keys if k in visible]
            if keys:
                sections.append((label, keys))
                rendered.update(keys)
        unassigned = [k for k in visible if k not in rendered]
        if unassigned:
            sections.append(("Other", unassigned))
        return sections

    # ── sweep-order + cap bookkeeping ─────────────────────────────────────────
    def _on_cell_toggled(self, param: str) -> None:
        checked = {p for p, c in self._cells.items() if c.is_swept}
        # selection order: survivors keep their old rank, new checks append
        self._sweep_order = [p for p in self._sweep_order if p in checked] \
            + [p for p in checked if p not in self._sweep_order]
        cap_reached = len(checked) >= MAX_SWEPT
        for p, cell in self._cells.items():
            if cell.check is not None and p not in checked:
                cell.check.setEnabled(not cap_reached)
        self._rebuild_roles()
        self.changed.emit()

    def _rebuild_roles(self) -> None:
        while self._roles_row.count():
            item = self._roles_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._role_combos = {}
        order = self._sweep_order
        available = ROLES[:max(len(order), 1)]
        for i, param in enumerate(order):
            box = QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(QLabel(param))
            combo = QComboBox()
            for role in available:
                combo.addItem(ROLE_LABELS.get(role, role), role)
            combo.setCurrentIndex(min(i, len(available) - 1))
            combo.currentIndexChanged.connect(lambda _=None: self.changed.emit())
            box.addWidget(combo)
            holder = QWidget()
            holder.setLayout(box)
            self._roles_row.addWidget(holder)
            self._role_combos[param] = combo
        self._roles_row.addStretch()

    # ── outputs (mirror the old return values) ────────────────────────────────
    def sweep_order(self) -> list[str]:
        return list(self._sweep_order)

    def fixed_params(self) -> dict:
        return {p: cell.fixed_value() for p, cell in self._cells.items()
                if not cell.is_swept}

    def swept_values(self) -> dict:
        return {p: self._cells[p].swept_values() for p in self._sweep_order}

    def roles_by_param(self) -> dict | None:
        """{param: role}; None when roles collide (error caption shown)."""
        roles = {p: combo.currentData()
                 for p, combo in self._role_combos.items()}
        if roles and len(set(roles.values())) != len(roles):
            self._role_error.setText("Each swept param needs a distinct role "
                                     "(X / Y / Slider 1 / Slider 2).")
            self._role_error.setStyleSheet(f"color: {theme.BAD}; font-size: 12px;")
            return None
        self._role_error.setText("")
        return roles
