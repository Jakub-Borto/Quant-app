"""
InstanceEditor — one collapsible "Instance N" configuration box:
override-trades-file checkbox (+ file combo when on / default caption when
off), sizer combo, the sizer's param form (shared keys excluded — they come
from the window's shared defaults), and the editable label with the
{sizer}_{n} auto-default that regenerates on sizer change unless hand-edited
(same behavior as the old widget-key trick).
"""

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QLabel,
                               QLineEdit, QVBoxLayout, QWidget)

from modules.analytics.backend.costs import SHARED_SIZER_KEYS
from modules.common.backend.data_roots import TradesRef
from modules.common.backend.plugins import PluginRef, load_module
from modules.common.ui.params_form import ParamsForm
from modules.common.ui.widgets import Banner, Caption, CollapsibleSection


class InstanceEditor(CollapsibleSection):
    def __init__(self, index: int, trades_refs: list[TradesRef],
                 sizer_refs: list[PluginRef], parent=None):
        super().__init__(f"Instance {index + 1}", expanded=(index == 0),
                         parent=parent)
        self._index = index
        self._trades_refs = trades_refs
        self._sizer_refs = sizer_refs
        self._default_ref: TradesRef | None = None
        self._sizer_module = None
        self._label_edited = False

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        self._override = QCheckBox("Override trades file")
        self._file = QComboBox()
        for ref in trades_refs:
            self._file.addItem(ref.label, ref)
        self._file.setVisible(False)
        self._default_caption = Caption("")
        grid.addWidget(self._override, 0, 0)
        grid.addWidget(self._file, 1, 0)
        grid.addWidget(self._default_caption, 1, 0)
        grid.addWidget(QLabel("sizer"), 0, 1)
        self._sizer = QComboBox()
        for ref in sizer_refs:
            self._sizer.addItem(ref.label, ref)
        grid.addWidget(self._sizer, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        self._banner = Banner()
        lay.addWidget(self._banner)
        self._params_holder = QVBoxLayout()
        lay.addLayout(self._params_holder)
        self._params_form: ParamsForm | None = None

        label_row = QGridLayout()
        label_row.addWidget(QLabel("label"), 0, 0)
        self._label = QLineEdit()
        self._label.textEdited.connect(self._on_label_edited)
        label_row.addWidget(self._label, 1, 0)
        lay.addLayout(label_row)

        self.add_widget(body)

        self._override.toggled.connect(self._sync_file_visibility)
        self._sizer.currentIndexChanged.connect(self._on_sizer_changed)
        self._on_sizer_changed()

    # ── wiring ────────────────────────────────────────────────────────────────
    def set_default_file(self, ref: TradesRef | None) -> None:
        self._default_ref = ref
        if ref is not None:
            self._default_caption.setText(f"Using default: {ref.label}")
            # override combo starts at the shared default, like the old index=
            i = self._file.findText(ref.label)
            if i >= 0 and not self._override.isChecked():
                self._file.setCurrentIndex(i)
        self._sync_file_visibility()

    def _sync_file_visibility(self) -> None:
        override = self._override.isChecked()
        self._file.setVisible(override)
        self._default_caption.setVisible(not override)

    def _on_label_edited(self, _text: str) -> None:
        self._label_edited = True

    def _auto_label(self) -> str:
        sizer_ref = self._sizer.currentData()
        name = sizer_ref.name if sizer_ref is not None else "sizer"
        return f"{name}_{self._index + 1}"

    def _on_sizer_changed(self) -> None:
        # rebuild the params form for the newly selected sizer
        while self._params_holder.count():
            item = self._params_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._params_form = None
        self._sizer_module = None
        self._banner.clear_message()

        ref: PluginRef | None = self._sizer.currentData()
        if ref is None:
            return
        try:
            self._sizer_module = load_module(ref)
        except Exception as e:  # noqa: BLE001
            self._banner.show_message("error", f"Failed to load sizer `{ref.name}`: {e}")
            return
        params = getattr(self._sizer_module, "PARAMS", {})
        user_keys = [k for k in params if k not in SHARED_SIZER_KEYS]
        if user_keys:
            self._params_form = ParamsForm(
                {k: params[k] for k in user_keys},
                options=getattr(self._sizer_module, "PARAMS_OPTIONS", None))
            self._params_holder.addWidget(self._params_form)

        # sizer change regenerates the auto label unless the user typed one
        if not self._label_edited:
            self._label.setText(self._auto_label())

    # ── output ────────────────────────────────────────────────────────────────
    def config(self, account_size: float) -> dict | None:
        """The instance config dict (None when the sizer failed to load)."""
        if self._sizer_module is None:
            return None
        if self._override.isChecked():
            trades_ref = self._file.currentData()
        else:
            trades_ref = self._default_ref
        if trades_ref is None:
            return None
        params = self._params_form.values() if self._params_form else {}
        params["account_size"] = account_size
        # dollars_per_tick intentionally NOT injected here — derived from the
        # filename at execution so it's always correct for the selected asset.
        sizer_ref: PluginRef = self._sizer.currentData()
        return {
            "label": self._label.text() or self._auto_label(),
            "trades_ref": trades_ref,
            "sizer": sizer_ref.name,
            "sizer_module": self._sizer_module,
            "params": params,
        }
