"""
Data Formatter window — raw DBN -> enriched parquet.

The PySide6 port of legacy_streamlit/views/data_formatter.py. Same flow:
pick a data root / input source / type / asset / dataset and a transform,
choose (or create) an output folder under <root>/parquet/<type>/<asset>/,
then Run — the transform's run_all() executes on a worker thread with live
progress + log, and Cancel aborts via the on_progress callback.

Output always lands in the SAME data root the input came from.
"""

import time
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton)

from modules.common.backend.data_roots import scan_structure
from modules.common.backend.plugins import PluginRef, list_plugins, load_module
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.widgets import Banner, Caption, ProgressLogPanel
from modules.common.ui.workers import FunctionWorker
from modules.data_formatter.backend.scan import get_output_folders

NEW_FOLDER = "── New folder ──"


class DataFormatterWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Data Formatter",
                         "Convert raw DBN files into candles and save as Parquet.",
                         parent)
        self._structure: dict = {}
        self._transforms: list[PluginRef] = []
        self._worker: FunctionWorker | None = None
        self._start_time = 0.0

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)

        # ── left column: input pickers ────────────────────────────────────────
        self._root = QComboBox()
        self._source = QComboBox()
        self._source.addItems(["raw_dbn", "parquet"])
        self._type = QComboBox()
        self._asset = QComboBox()
        self._dataset = QComboBox()
        self._transform = QComboBox()

        left_rows = [("Data root", self._root), ("Input source", self._source),
                     ("Type", self._type), ("Asset", self._asset),
                     ("Input dataset", self._dataset),
                     ("Transform", self._transform)]
        for r, (label, widget) in enumerate(left_rows):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(widget, r, 1)

        # ── right column: output picker ───────────────────────────────────────
        self._output_hint = Caption("")
        self._output = QComboBox()
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText("e.g. ES_indicators")
        grid.addWidget(self._output_hint, 0, 2, 1, 2)
        grid.addWidget(QLabel("Output folder"), 1, 2)
        grid.addWidget(self._output, 1, 3)
        self._new_name_label = QLabel("New folder name")
        grid.addWidget(self._new_name_label, 2, 2)
        grid.addWidget(self._new_name, 2, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.content.addLayout(grid)

        self._skip = QCheckBox("Skip already processed files")
        self._skip.setChecked(True)
        self.content.addWidget(self._skip)

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
        self.content.addLayout(btn_row)

        self._banner = Banner()
        self.content.addWidget(self._banner)
        self._progress = ProgressLogPanel(log_height=420)
        self.content.addWidget(self._progress)
        self.content.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._root.currentIndexChanged.connect(self._rescan_structure)
        self._source.currentIndexChanged.connect(self._rescan_structure)
        self._type.currentIndexChanged.connect(self._on_type_changed)
        self._asset.currentIndexChanged.connect(self._on_asset_changed)
        self._output.currentIndexChanged.connect(self._on_output_changed)
        self._rescan()

    # ── scanning / cascading pickers ──────────────────────────────────────────
    def _rescan(self) -> None:
        """Re-read settings-driven folders (data roots + transform plugins)."""
        self._transforms = list_plugins(
            self.settings.plugin_dirs("data_transforms"))
        self._transform.clear()
        self._transform.addItems([t.label for t in self._transforms])

        self._root.blockSignals(True)
        self._root.clear()
        for root in self.settings.data_roots:
            self._root.addItem(str(root), root)
        self._root.blockSignals(False)
        self._root.setVisible(self._root.count() > 1)
        self._rescan_structure()

    def _current_root(self) -> Path:
        return Path(self._root.currentData()) if self._root.count() \
            else Path(self.settings.data_roots[0])

    def _rescan_structure(self) -> None:
        source = self._source.currentText()
        merged = scan_structure([self._current_root()], source=source)
        # single-root scan -> {type: {asset: [DatasetRef]}}
        self._structure = merged
        self._type.blockSignals(True)
        self._type.clear()
        self._type.addItems(list(merged.keys()))
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
        self._refresh_output_options()

    def _refresh_output_options(self) -> None:
        root, atype, asset = (self._current_root(), self._type.currentText(),
                              self._asset.currentText())
        self._output_hint.setText(
            f"Output path: {root / 'parquet' / atype / asset}/")
        self._output.blockSignals(True)
        self._output.clear()
        self._output.addItems([NEW_FOLDER] + get_output_folders(root, atype, asset))
        self._output.blockSignals(False)
        self._on_output_changed()

    def _on_output_changed(self) -> None:
        is_new = self._output.currentText() == NEW_FOLDER
        self._new_name.setVisible(is_new)
        self._new_name_label.setVisible(is_new)

    # ── run flow ──────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        if self._transform.currentIndex() < 0:
            self._banner.show_message("error", "No transform scripts found in "
                                               "the configured data_transforms folders.")
            return
        if not self._dataset.currentText():
            self._banner.show_message("error", "No input dataset selected.")
            return
        if self._output.currentText() == NEW_FOLDER:
            output_folder_name = self._new_name.text().strip()
        else:
            output_folder_name = self._output.currentText()
        if not output_folder_name:
            self._banner.show_message("error", "Please enter an output folder name.")
            return

        root = self._current_root()
        atype, asset = self._type.currentText(), self._asset.currentText()
        input_path = str(root / self._source.currentText() / atype / asset
                         / self._dataset.currentText())
        output_path = str(root / "parquet" / atype / asset / output_folder_name)

        transform = load_module(self._transforms[self._transform.currentIndex()])

        self._progress.reset()
        self._start_time = time.time()
        self._output_path_str = output_path

        worker = FunctionWorker(transform.run_all, input_folder=input_path,
                                output_folder=output_path,
                                skip_existing=self._skip.isChecked(),
                                needs_progress=True)
        worker.signals.progress.connect(self._progress.on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        worker.signals.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self.track_worker(worker)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def _reset_buttons(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

    def _on_finished(self, _result) -> None:
        elapsed = time.time() - self._start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        took = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        self._banner.show_message(
            "success", f"Done in {took}. Saved to {self._output_path_str}")
        self._reset_buttons()

    def _on_error(self, message: str, _tb: str) -> None:
        self._banner.show_message("error", message)
        self._reset_buttons()

    def _on_cancelled(self) -> None:
        self._banner.show_message("warning", "Cancelled. Files already written "
                                             "stay on disk (skip-existing will "
                                             "pick up where this stopped).")
        self._reset_buttons()
