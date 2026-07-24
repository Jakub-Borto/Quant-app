"""
Regime Detector — Run tab. Pick a parquet dataset (union of the type/asset/
dataset tree, any type), pick a detector script from the configured
regime_detectors folders, set params, name the run, Run/Cancel with live
progress. Output goes to {input root}/regimes/{ASSET}/{run_name}/ — always
the root the input came from.

Rerunning into an existing run name is a RESUME: skip_existing fills only
missing days. If the existing meta.json was produced by a different script/
version/params, a warning banner says so before the run proceeds (the meta
is rewritten).
"""

from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from modules.common.backend.data_roots import regimes_dir, scan_structure
from modules.common.backend.plugins import PluginRef, list_plugins, load_module
from modules.common.ui.params_form import ParamsForm
from modules.common.ui.widgets import (Banner, Caption, ProgressLogPanel,
                                       SectionHeader)
from modules.common.ui.widgets import wrap_card
from modules.common.ui.workers import FunctionWorker
from modules.regime_detector.backend import io as rio
from modules.regime_detector.backend import schema


class RunTab(QWidget):
    def __init__(self, settings, track_worker, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._track_worker = track_worker
        self._structure: dict = {}
        self._detectors: list[PluginRef] = []
        self._module = None
        self._worker: FunctionWorker | None = None
        self._name_edited = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)

        # ── left column: input pickers ────────────────────────────────────────
        self._root = QComboBox()
        self._type = QComboBox()
        self._asset = QComboBox()
        self._dataset = QComboBox()
        self._detector = QComboBox()
        left_rows = [("Data root", self._root), ("Type", self._type),
                     ("Asset", self._asset), ("Input dataset", self._dataset),
                     ("Detector script", self._detector)]
        for r, (label, widget) in enumerate(left_rows):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(widget, r, 1)

        # ── right column: output run name ─────────────────────────────────────
        self._output_hint = Caption("")
        self._run_name = QLineEdit()
        grid.addWidget(self._output_hint, 0, 2, 1, 2)
        grid.addWidget(QLabel("Run name"), 1, 2)
        grid.addWidget(self._run_name, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        outer.addWidget(wrap_card(grid))

        # detector params form
        self._params_header = SectionHeader("Detector parameters")
        self._params_header.setVisible(False)
        outer.addWidget(self._params_header)
        self._params_holder = QVBoxLayout()
        outer.addLayout(self._params_holder)
        self._params_form: ParamsForm | None = None

        self._skip = QCheckBox("Skip days that already have output "
                               "(resume the run)")
        self._skip.setChecked(True)
        outer.addWidget(self._skip)

        # ── run / cancel ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(180)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        refresh_btn = QPushButton("Refresh folders")
        refresh_btn.clicked.connect(self._rescan)
        btn_row.addStretch()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        self._banner = Banner()
        outer.addWidget(self._banner)
        self._progress = ProgressLogPanel(log_height=360)
        outer.addWidget(self._progress)
        outer.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._root.currentIndexChanged.connect(self._rescan_structure)
        self._type.currentIndexChanged.connect(self._on_type_changed)
        self._asset.currentIndexChanged.connect(self._on_asset_changed)
        self._detector.currentIndexChanged.connect(self._on_detector_changed)
        self._run_name.textEdited.connect(self._on_name_edited)
        self._rescan()

    # ── scanning / cascading pickers ──────────────────────────────────────────
    def _rescan(self) -> None:
        self._detectors = list_plugins(
            self.settings.plugin_dirs("regime_detectors"))
        self._detector.blockSignals(True)
        self._detector.clear()
        self._detector.addItems([d.label for d in self._detectors])
        self._detector.blockSignals(False)

        self._root.blockSignals(True)
        self._root.clear()
        for root in self.settings.data_roots:
            self._root.addItem(str(root), root)
        self._root.blockSignals(False)
        self._root.setVisible(self._root.count() > 1)
        self._rescan_structure()
        self._on_detector_changed()

    def _current_root(self) -> Path:
        return Path(self._root.currentData()) if self._root.count() \
            else Path(self.settings.data_roots[0])

    def _rescan_structure(self) -> None:
        self._structure = scan_structure([self._current_root()],
                                         source="parquet")
        self._type.blockSignals(True)
        self._type.clear()
        self._type.addItems(list(self._structure.keys()))
        self._type.blockSignals(False)
        self._on_type_changed()

    def _on_type_changed(self) -> None:
        assets = list(self._structure.get(self._type.currentText(), {}).keys())
        self._asset.blockSignals(True)
        self._asset.clear()
        self._asset.addItems(assets)
        self._asset.blockSignals(False)
        self._on_asset_changed()

    def _on_asset_changed(self) -> None:
        refs = self._structure.get(self._type.currentText(), {}) \
                              .get(self._asset.currentText(), [])
        self._dataset.clear()
        self._dataset.addItems([r.dataset for r in refs])
        asset = self._asset.currentText()
        self._output_hint.setText(
            f"Output path: {regimes_dir(self._current_root(), asset)}/"
            if asset else "")
        self._refresh_proposed_name()

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def _refresh_proposed_name(self) -> None:
        """Prefill {SYMBOL}_{script_stem}; stop once the user typed a name."""
        if self._name_edited:
            return
        asset = self._asset.currentText()
        if asset and 0 <= self._detector.currentIndex() < len(self._detectors):
            stem = self._detectors[self._detector.currentIndex()].name
            self._run_name.setText(rio.propose_run_name(asset, stem))

    def _on_detector_changed(self) -> None:
        """Load the selected script, validate its contract, rebuild params."""
        while self._params_holder.count():
            item = self._params_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._params_form = None
        self._params_header.setVisible(False)
        self._module = None
        self._banner.clear_message()
        if not (0 <= self._detector.currentIndex() < len(self._detectors)):
            return
        try:
            module = load_module(self._detectors[self._detector.currentIndex()])
        except Exception as e:  # noqa: BLE001 — broken plugin must not kill the window
            self._banner.show_message("error", f"Could not load detector: {e}")
            return
        errors = schema.validate_detector(module)
        if errors:
            self._banner.show_message(
                "error", "Detector contract errors: " + "; ".join(errors))
            return
        self._module = module
        self._params_form = ParamsForm(
            module.PARAMS, sections=getattr(module, "PARAM_SECTIONS", None),
            options=getattr(module, "PARAMS_OPTIONS", None))
        self._params_holder.addWidget(self._params_form)
        self._params_header.setVisible(True)
        self._refresh_proposed_name()

    # ── run flow ──────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        if self._module is None:
            self._banner.show_message("error", "No valid detector selected "
                                               "(check the regime_detectors "
                                               "folders in Settings).")
            return
        if not self._dataset.currentText():
            self._banner.show_message("error", "No input dataset selected.")
            return
        try:
            run_name = rio.safe_name(self._run_name.text())
        except ValueError:
            self._banner.show_message("error", "Please enter a run name.")
            return

        root = self._current_root()
        input_path = (root / "parquet" / self._type.currentText()
                      / self._asset.currentText() / self._dataset.currentText())
        output_path = regimes_dir(root, self._asset.currentText()) / run_name
        params = self._params_form.values()

        self._warn_on_mismatch(output_path, params)

        self._progress.reset()
        self._output_path_str = str(output_path)
        worker = FunctionWorker(self._module.run_all,
                                input_folder=str(input_path),
                                output_folder=str(output_path),
                                skip_existing=self._skip.isChecked(),
                                params=params, needs_progress=True)
        worker.signals.progress.connect(self._progress.on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        worker.signals.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._track_worker(worker)

    def _warn_on_mismatch(self, output_path: Path, params: dict) -> None:
        """Resuming an existing run with a different script/params is legal
        but suspicious — say so out loud before proceeding."""
        try:
            prev = rio.read_meta(output_path)
        except (OSError, ValueError):
            return
        diffs = []
        current_stem = self._detectors[self._detector.currentIndex()].name
        if prev.get("script_name") != current_stem:
            diffs.append(f"script ({prev.get('script_name')} → {current_stem})")
        if prev.get("script_version") != self._module.SCRIPT_VERSION:
            diffs.append(f"script_version ({prev.get('script_version')} → "
                         f"{self._module.SCRIPT_VERSION})")
        prev_params = prev.get("params", {})
        changed = [k for k in set(prev_params) | set(params)
                   if prev_params.get(k) != params.get(k)]
        if changed:
            diffs.append("params: " + ", ".join(sorted(changed)))
        if diffs:
            self._banner.show_message(
                "warning", "Resuming into an existing run whose meta.json "
                "differs — " + "; ".join(diffs) + ". Existing daily files are "
                "kept as-is; meta.json will be rewritten.")

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def _reset_buttons(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

    def _on_finished(self, _result) -> None:
        self._banner.show_message("success",
                                  f"Done. Saved to {self._output_path_str}")
        self._reset_buttons()

    def _on_error(self, message: str, _tb: str) -> None:
        self._banner.show_message("error", message)
        self._reset_buttons()

    def _on_cancelled(self) -> None:
        self._banner.show_message(
            "warning", "Cancelled. Days already written stay on disk — "
                       "rerunning with skip enabled resumes from there.")
        self._reset_buttons()
