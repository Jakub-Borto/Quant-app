"""
Optimizer "Combine" tab — the Strategy Combiner (no-overlap merge of saved
runs' variants, greedy IS selection + sealed OOS path).

The PySide6 port of render_combine / _render_combine_results /
_render_path_chart: container + entry-run pickers with the load cache and the
compatibility gate, day-bucket checkboxes, per-trade-type min-trades floors,
selection settings (IS/OOS split, redundancy penalty λ, max set size, greedy
seeds — same defaults and tooltips), Run on a worker with the throttled log,
save + the saved-combine-run viewer (path chart, rounded table, member
inspector). Nothing is ever re-backtested.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

import pandas as pd

from modules.common.backend.data_roots import optimizations_root
from modules.common.ui.charts.path_chart import CombinePathChart
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.trade_report.filters import CheckboxFilterRow
from modules.common.ui.widgets import (Banner, Caption, CollapsibleSection,
                                       ProgressLogPanel, SectionHeader,
                                       hline, wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.optimizer.backend.buckets import BUCKET_ORDER
from modules.optimizer.backend.combine import io as cmb_io
from modules.optimizer.backend.combine.compat import check_compatibility
from modules.optimizer.backend.combine.pool import (discover_entry_runs,
                                                    list_containers,
                                                    load_entry_runs)
from modules.optimizer.backend.combine.runner import run_combine

_SELECTION_TOOLTIP = """\
Redundancy penalty λ (ticks per unit correlation): greedy scores each
candidate as `marginal merged ticks − λ × corr`, where corr is the daily-P&L
correlation with the MOST similar already-selected member. λ=0 accepts any
positive gain — near-duplicate param nudges (corr ≈ 1, tiny gains) pile up.
λ=30 means a perfect clone must beat an uncorrelated alternative by 30+ ticks.

Max set size: a ceiling on the greedy path, not a target — greedy already
stops when no candidate adds positive score.

Greedy seeds: insurance against first-pick lock-in — the forward pass re-runs
N times, forcing the first pick to each of the top-N standalone in-sample
variants; the best-ending path wins. 1 = plain greedy; 3–5 is a cheap
robustness check (runtime × N)."""


class CombineTab(QWidget):
    def __init__(self, settings, track_worker, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._track_worker = track_worker
        self._loaded: dict | None = None          # opt_cmb_loaded
        self._cache_key = None                    # opt_cmb_cache_key
        self._name_edited = False

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # ── container + entry runs ────────────────────────────────────────────
        sel = QGridLayout()
        self._root = QComboBox()
        sel.addWidget(QLabel("Data root"), 0, 0)
        sel.addWidget(self._root, 1, 0)
        self._container = QComboBox()
        sel.addWidget(QLabel("Container folder"), 0, 1)
        sel.addWidget(self._container, 1, 1)
        sel.setColumnStretch(1, 2)
        lay.addWidget(wrap_card(sel))
        self._info = Banner()
        lay.addWidget(self._info)

        runs_label = QLabel("Entry runs to pool")
        runs_label.setToolTip("every child folder holding meta.json + "
                              "trades.parquet; _combined/ and incomplete "
                              "folders are excluded")
        lay.addWidget(runs_label)
        self._runs_list = QListWidget()
        self._runs_list.setMaximumHeight(150)
        lay.addWidget(self._runs_list)

        self._gate = Banner()
        lay.addWidget(self._gate)
        self._gate_warnings = Banner()
        lay.addWidget(self._gate_warnings)

        # ── pre-run controls ──────────────────────────────────────────────────
        lay.addWidget(Caption("Day types included (dropped from the pool "
                              "before the split)"))
        self._buckets = CheckboxFilterRow(list(BUCKET_ORDER),
                                          per_row=len(BUCKET_ORDER))
        lay.addWidget(self._buckets)

        lay.addWidget(Caption("Per-entry min-trades floor (in-sample trade count)"))
        self._floors_holder = QGridLayout()
        lay.addLayout(self._floors_holder)
        self._floor_spins: dict[str, QSpinBox] = {}

        settings_label = QLabel("Selection settings ⓘ")
        settings_label.setToolTip(_SELECTION_TOOLTIP)
        lay.addWidget(settings_label)
        srow = QGridLayout()
        self._split = QComboBox()
        self._split.addItems(["50/50", "60/40", "70/30", "80/20"])
        self._split.setCurrentIndex(2)
        self._split.setToolTip("chronological: train = earliest X% of trading dates")
        self._lam = QDoubleSpinBox()
        self._lam.setRange(0.0, 1e9)
        self._lam.setSingleStep(5.0)
        self._lam.setValue(0.0)
        self._lam.setToolTip("ticks subtracted per unit of daily-P&L correlation "
                             "with the closest selected member; 0 = off, ~20-40 "
                             "suppresses near-duplicate param nudges")
        self._max_k = QSpinBox()
        self._max_k.setRange(1, 200)
        self._max_k.setSingleStep(5)
        self._max_k.setValue(30)
        self._max_k.setToolTip("ceiling on the greedy path, not a target — "
                               "greedy stops on its own when nothing adds "
                               "positive score")
        self._seeds = QSpinBox()
        self._seeds.setRange(1, 10)
        self._seeds.setValue(1)
        self._seeds.setToolTip("re-run forward selection forcing the first pick "
                               "to each of the top-N standalone variants; best "
                               "path wins (runtime × N)")
        for c, (label, w) in enumerate([("IS/OOS split", self._split),
                                        ("Redundancy penalty λ", self._lam),
                                        ("Max set size", self._max_k),
                                        ("Greedy seeds", self._seeds)]):
            srow.addWidget(QLabel(label), 0, c)
            srow.addWidget(w, 1, c)
        lay.addWidget(wrap_card(srow))

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Combine run name"))
        self._name = QLineEdit()
        self._name.textEdited.connect(lambda _=None: setattr(self, "_name_edited", True))
        name_row.addWidget(self._name)
        lay.addLayout(name_row)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run combine")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addStretch()
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        self._progress = ProgressLogPanel(log_height=200)
        lay.addWidget(self._progress)
        self._run_banner = Banner()
        lay.addWidget(self._run_banner)

        # ── saved combine runs viewer ─────────────────────────────────────────
        self._results_box = QWidget()
        self._results_box.setVisible(False)
        rlay = QVBoxLayout(self._results_box)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(10)
        rlay.addWidget(hline())
        rlay.addWidget(SectionHeader("Saved combine runs"))
        rlay.addWidget(Caption("These are snapshots — the controls above only "
                               "shape the NEXT “Run combine”. Check which run "
                               "you're viewing here."))
        self._result_select = QComboBox()
        rlay.addWidget(self._result_select)
        self._result_caption = Caption("")
        rlay.addWidget(self._result_caption)
        self._inputs_section = CollapsibleSection(
            "This run's inputs (entry runs · floors · day types)")
        self._inputs_label = QLabel()
        self._inputs_label.setTextFormat(Qt.MarkdownText)
        self._inputs_label.setWordWrap(True)
        self._inputs_section.add_widget(self._inputs_label)
        rlay.addWidget(self._inputs_section)
        self._path_chart = CombinePathChart()
        rlay.addWidget(self._path_chart)
        self._peak_caption = Caption("")
        rlay.addWidget(self._peak_caption)
        self._path_table = make_table_view(pd.DataFrame(), height=300)
        rlay.addWidget(self._path_table)
        self._oos_warning = Banner()
        rlay.addWidget(self._oos_warning)
        inspect_row = QHBoxLayout()
        inspect_row.addWidget(QLabel("Inspect member variants of…"))
        self._inspect = QComboBox()
        inspect_row.addWidget(self._inspect)
        inspect_row.addStretch()
        rlay.addLayout(inspect_row)
        self._members_table = make_table_view(pd.DataFrame(), height=260)
        rlay.addWidget(self._members_table)
        lay.addWidget(self._results_box)
        lay.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._root.currentIndexChanged.connect(self._refresh_containers)
        self._container.currentIndexChanged.connect(self._on_container_changed)
        self._runs_list.itemChanged.connect(self._on_runs_changed)
        self._split.currentIndexChanged.connect(self._sync_default_name)
        self._lam.valueChanged.connect(self._sync_default_name)
        self._result_select.currentIndexChanged.connect(self._show_selected_result)
        self._inspect.currentIndexChanged.connect(self._show_members)
        self.rescan()

    # ══ scanning ═══════════════════════════════════════════════════════════════
    def rescan(self) -> None:
        self._root.blockSignals(True)
        self._root.clear()
        for root in self.settings.data_roots:
            self._root.addItem(str(root), root)
        self._root.blockSignals(False)
        self._root.setVisible(self._root.count() > 1)
        self._refresh_containers()

    def _runs_root(self):
        root = self._root.currentData() if self._root.count() \
            else self.settings.data_roots[0]
        return optimizations_root(root)

    def _refresh_containers(self) -> None:
        containers = list_containers(root=self._runs_root())
        self._info.clear_message()
        self._container.blockSignals(True)
        self._container.clear()
        self._container.addItems(containers)
        self._container.blockSignals(False)
        if not containers:
            self._info.show_message(
                "info", "No containers with entry runs under this data root's "
                        "optimizations/ — save some optimizer runs into a "
                        "folder first.")
            self._results_box.setVisible(False)
            return
        self._on_container_changed()

    def _on_container_changed(self) -> None:
        container = self._container.currentText()
        if not container:
            return
        entry_runs = discover_entry_runs(container, root=self._runs_root())
        self._runs_list.blockSignals(True)
        self._runs_list.clear()
        for name in entry_runs:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)      # default = all selected
            self._runs_list.addItem(item)
        self._runs_list.blockSignals(False)
        self._sync_default_name()
        self._on_runs_changed()
        self._refresh_results()

    def _selected_runs(self) -> list[str]:
        return [self._runs_list.item(i).text()
                for i in range(self._runs_list.count())
                if self._runs_list.item(i).checkState() == Qt.Checked]

    def _on_runs_changed(self, _item=None) -> None:
        """(Re)load the ticked runs (cached) and run the compatibility gate."""
        self._gate.clear_message()
        self._gate_warnings.clear_message()
        container = self._container.currentText()
        selected = self._selected_runs()
        if not container or not selected:
            self._gate.show_message("info", "Tick at least one entry run.")
            self._loaded = None
            self._run_btn.setEnabled(False)
            return

        cache_key = (str(self._runs_root()), container, tuple(sorted(selected)))
        if self._cache_key != cache_key:
            self._loaded = load_entry_runs(container, selected,
                                           root=self._runs_root())
            self._cache_key = cache_key

        gate = check_compatibility({n: meta for n, (meta, _t) in self._loaded.items()})
        if not gate["ok"]:
            self._gate.show_message(
                "error", "Incompatible runs — cannot combine:\n"
                         + "\n".join(f"- {e}" for e in gate["errors"]))
            self._run_btn.setEnabled(False)
            return
        self._gate.show_message(
            "success",
            f"Compatible: {gate['ticker']} on {gate['dataset']} · ticks/point "
            f"{gate['ticks_per_point']:g} · shared window "
            f"{gate['shared_start'].date()} → {gate['shared_end'].date()}")
        if gate["warnings"]:
            self._gate_warnings.show_message("warning", "\n".join(gate["warnings"]))
        self._run_btn.setEnabled(True)
        self._rebuild_floors()

    def _rebuild_floors(self) -> None:
        while self._floors_holder.count():
            item = self._floors_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._floor_spins = {}
        if not self._loaded:
            return
        trade_types = sorted({
            str(t) for _, (meta, trades) in self._loaded.items()
            for t in (trades["trade_type"].dropna().unique()
                      if "trade_type" in trades.columns else ["unknown"])
        })
        for i, tt in enumerate(trade_types):
            spin = QSpinBox()
            spin.setRange(0, 1_000_000)
            spin.setSingleStep(5)
            spin.setValue(10)
            self._floor_spins[tt] = spin
            self._floors_holder.addWidget(QLabel(tt), (i // 6) * 2, i % 6)
            self._floors_holder.addWidget(spin, (i // 6) * 2 + 1, i % 6)

    def _sync_default_name(self, _=None) -> None:
        if not self._name_edited:
            split_label = self._split.currentText()
            self._name.setText(
                f"combine_{split_label.replace('/', '-')}_lam{self._lam.value():g}")

    # ══ run ═══════════════════════════════════════════════════════════════════
    def _on_run(self) -> None:
        self._run_banner.clear_message()
        container = self._container.currentText()
        selected = self._selected_runs()
        enabled_buckets = set(self._buckets.selected())
        if not enabled_buckets:
            self._run_banner.show_message("warning", "No day types selected.")
            return
        run_name = self._name.text()
        if not run_name.strip():
            self._run_banner.show_message("error", "Enter a run name.")
            return
        floors = {tt: int(spin.value()) for tt, spin in self._floor_spins.items()}
        is_fraction = int(self._split.currentText().split("/")[0]) / 100
        root = self._runs_root()

        self._progress.reset()
        self._run_btn.setEnabled(False)
        self._pending = {"container": container, "run_name": run_name,
                         "root": root}

        def job(on_progress=None):
            # run_combine's log(msg) callback maps onto the progress log
            def log(msg):
                on_progress(0, 1, str(msg))
            return run_combine(
                container, selected, enabled_buckets=enabled_buckets,
                floors=floors, is_fraction=is_fraction,
                lam=float(self._lam.value()), max_k=int(self._max_k.value()),
                n_seeds=int(self._seeds.value()), log=log, root=root)

        worker = FunctionWorker(job, needs_progress=True)
        worker.signals.progress.connect(self._progress.on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._track_worker(worker)

    def _on_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_banner.show_message("error", message)

    def _on_finished(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        run_dir = cmb_io.save_combine_run(
            self._pending["container"], self._pending["run_name"],
            result["path_df"], result["members_df"], result["meta"],
            root=self._pending["root"])
        self._run_banner.show_message("success", f"Saved to {run_dir}")
        self._refresh_results(select=run_dir.name)

    # ══ saved-run viewer ════════════════════════════════════════════════════════
    def _refresh_results(self, select: str | None = None) -> None:
        container = self._container.currentText()
        saved = cmb_io.list_combine_runs(container, root=self._runs_root()) \
            if container else []
        if not saved:
            self._results_box.setVisible(False)
            return
        self._results_box.setVisible(True)
        self._result_select.blockSignals(True)
        self._result_select.clear()
        self._result_select.addItems(saved)
        if select and select in saved:
            self._result_select.setCurrentIndex(saved.index(select))
        self._result_select.blockSignals(False)
        self._show_selected_result()

    def _show_selected_result(self) -> None:
        container = self._container.currentText()
        name = self._result_select.currentText()
        if not container or not name:
            return
        path_df, members_df, meta = cmb_io.load_combine_run(
            container, name, root=self._runs_root())
        self._path_df, self._members_df = path_df, members_df

        self._result_caption.setText(
            f"{meta.get('ticker')} · {len(meta.get('runs', []))} entry runs · "
            f"{meta.get('shared_start')} → {meta.get('shared_end')} · "
            f"IS through {meta.get('split_boundary')} "
            f"({meta.get('is_fraction', 0):.0%}) · λ={meta.get('lambda', 0):g} · "
            f"pool {meta.get('pool_size')} variants")
        floors_used = meta.get("min_trades_floors", {})
        self._inputs_label.setText(
            "**Entry runs pooled:**\n"
            + "\n".join(f"- {r}" for r in meta.get("runs", []))
            + "\n\n**Min-trades floors (in-sample):** "
            + (", ".join(f"{k}={v}" for k, v in floors_used.items())
               if floors_used else "none")
            + "\n\n**Day types included:** "
            + ", ".join(meta.get("enabled_day_buckets", [])))

        self._path_chart.set_path(path_df)
        peak_k = path_df.loc[path_df["is_oos_peak"], "k"]
        self._peak_caption.setText(
            f"OOS peak at k = {int(peak_k.iloc[0]) if len(peak_k) else '—'} — "
            "read it as “look around here”: the exact argmax is one noisy "
            "realization, prefer the plateau. Selection saw only the in-sample "
            "slice; judge sets by the orange line.")

        table = path_df[["k", "stage", "is_ticks", "oos_ticks", "is_sharpe",
                         "oos_sharpe", "is_max_dd", "oos_max_dd",
                         "n_trades_is", "n_trades_oos", "is_oos_peak"]].copy()
        for col in ("is_ticks", "oos_ticks", "is_max_dd", "oos_max_dd"):
            table[col] = table[col].round(0)
        for col in ("is_sharpe", "oos_sharpe"):
            table[col] = table[col].round(2)
        update_table_view(self._path_table, table)
        if path_df["oos_empty"].any():
            self._oos_warning.show_message(
                "warning", "Some sets have an empty out-of-sample slice (all "
                           "member trades fall in-sample) — their OOS ticks "
                           "read 0 by convention.")
        else:
            self._oos_warning.clear_message()

        # member inspector — default = the OOS-peak row; nothing auto-chosen
        options = [f"k={row.k} ({row.stage})" for row in path_df.itertuples()]
        self._inspect.blockSignals(True)
        self._inspect.clear()
        self._inspect.addItems(options)
        self._inspect.setCurrentIndex(int(path_df["is_oos_peak"].idxmax()))
        self._inspect.blockSignals(False)
        self._show_members()

    def _show_members(self) -> None:
        if self._inspect.currentIndex() < 0:
            return
        row = self._path_df.iloc[self._inspect.currentIndex()]
        members = self._members_df[(self._members_df["k"] == row["k"])
                                   & (self._members_df["stage"] == row["stage"])]
        update_table_view(self._members_table,
                          members[["trade_type", "params", "run", "n_is", "n_oos"]])
