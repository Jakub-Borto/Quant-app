"""
ReportsWindow — research reports (Streamlit apps) with archive and compress
management.

Tabs:
  Reports   every report folder (a direct subfolder with report.py) across
            the configured report folders: run / stop / restart / open in the
            dedicated scripts browser / console (the Scripts module's process
            + log machinery, reused), plus Open folder, Archive and Delete
            (both disabled while the report runs). Refreshes on tab focus.
  Archive   the zips under every `<reports folder>/_archive/`: Restore,
            Delete, Open archive folder.
  Compress  a full Optimizer run inside a reports folder -> a compact
            per-combination run in a destination inside a reports folder.

Archive / restore / delete / compress run on worker threads (the window never
freezes); every delete goes through the typed-code DeleteConfirmDialog.
Closing the window stops every report process it started (after asking).
"""

import time
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.widgets import (Banner, Caption, Card, SectionHeader,
                                       hline, pin_minimum_height, wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.scripts.backend.ports import find_free_port
from modules.scripts.log_panel import STATE_COLORS, InstanceLogPanel
from modules.scripts.process_manager import ScriptInstance, open_in_scripts_browser

from .backend import archive as arch
from .backend import compact
from .backend.launch import launch_args
from .backend.scan import (ARCHIVE_DIR, human_size, list_archives, list_reports,
                           owning_reports_folder, is_in_archive)
from .delete_dialog import DeleteConfirmDialog

REFRESH_MIN_INTERVAL = 1.0     # s — focus-driven refreshes are throttled


class ReportInstance(ScriptInstance):
    """One report's Streamlit process: launched through the reports
    bootstrap (repo root APPENDED to sys.path, the shared theme, wide
    layout) with cwd = the report folder."""

    def __init__(self, report_ref, instance_no: int, port: int, parent=None):
        shim = SimpleNamespace(name=report_ref.name, path=report_ref.entry)
        super().__init__(shim, "streamlit", instance_no, port=port, parent=parent)
        self.report_path = Path(report_ref.path)

    def launch_args(self) -> list[str]:
        return launch_args(self.script_path, self.port)


def _open_folder(path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _danger(btn: QPushButton) -> QPushButton:
    btn.setStyleSheet("QPushButton:enabled { color: #ff9d9d; border-color: #6b2f2f; } "
                      "QPushButton:enabled:hover { border-color: #c94444; }")
    return btn


class _JobStrip(QWidget):
    """Busy line for a background job: message + progress bar + Cancel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(260)
        self.cancel = QPushButton("Cancel")
        lay.addWidget(self.label, 1)
        lay.addWidget(self.bar)
        lay.addWidget(self.cancel)
        self.setVisible(False)
        self._worker = None
        self.cancel.clicked.connect(self._on_cancel)

    def start(self, text: str, worker: FunctionWorker) -> None:
        self._worker = worker
        self.label.setText(text)
        self.bar.setValue(0)
        self.cancel.setEnabled(True)
        self.setVisible(True)
        worker.signals.progress.connect(self.on_progress)
        for sig in (worker.signals.finished, worker.signals.error,
                    worker.signals.cancelled):
            sig.connect(lambda *_a: self.stop())

    def on_progress(self, current: int, total: int, message: str) -> None:
        self.bar.setValue(int(1000 * current / total) if total else 0)
        if message:
            self.label.setText(message)

    def stop(self) -> None:
        self._worker = None
        self.setVisible(False)

    def busy(self) -> bool:
        return self._worker is not None

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel.setEnabled(False)
            self.label.setText("Cancelling…")


class ReportsWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(
            settings, "Reports",
            "Research reports — each one a Streamlit app with prose, math and "
            "live displays of saved results. Archive, restore, and compress "
            "Optimizer runs for them.",
            parent)
        self._instances: dict[Path, ReportInstance] = {}
        self._counters: dict[Path, int] = {}
        self._busy: set[Path] = set()          # report folders / zips with a job
        self._last_refresh = 0.0

        self._banner = Banner()
        self.content.addWidget(self._banner)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_reports_tab(), "Reports")
        self.tabs.addTab(self._build_archive_tab(), "Archive")
        self.tabs.addTab(self._build_compress_tab(), "Compress")
        self.tabs.currentChanged.connect(lambda _i: self.refresh(force=True))
        self.content.addWidget(self.tabs)
        self.content.addStretch()

        self.refresh(force=True)

    # ══ shared ═════════════════════════════════════════════════════════════════
    def folders(self) -> list[Path]:
        return [Path(d) for d in self.settings.plugin_dirs("reports")]

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < REFRESH_MIN_INTERVAL:
            return
        self._last_refresh = now
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._refresh_reports()
        elif idx == 1:
            self._refresh_archives()
        else:
            self._refresh_sources()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self.refresh()

    def _run_job(self, strip: _JobStrip, text: str, fn, *args, busy_key=None,
                 on_done=None, **kwargs) -> None:
        worker = FunctionWorker(fn, *args, needs_progress=True, **kwargs)
        if busy_key is not None:
            self._busy.add(busy_key)

        def finish(*_a):
            strip.stop()                       # before the refresh reads busy()
            self._busy.discard(busy_key)
            self.refresh(force=True)

        def ok(result):
            if on_done is not None:
                on_done(result)
            finish()

        def failed(message, _tb):
            self._banner.show_message("error", message)
            finish()

        def cancelled():
            self._banner.show_message("info", f"{text} — cancelled; nothing changed.")
            finish()

        worker.signals.finished.connect(ok)
        worker.signals.error.connect(failed)
        worker.signals.cancelled.connect(cancelled)
        strip.start(text, worker)
        self._banner.clear_message()
        self.track_worker(worker)
        self.refresh(force=True)

    # ══ Reports tab ════════════════════════════════════════════════════════════
    def _build_reports_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        pin_minimum_height(page)
        head = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda: self.refresh(force=True))
        head.addWidget(refresh)
        self._where = Caption("")
        head.addWidget(self._where, 1)
        lay.addLayout(head)
        self._reports_job = _JobStrip()
        lay.addWidget(self._reports_job)
        self._report_rows = QVBoxLayout()
        self._report_rows.setSpacing(10)
        lay.addLayout(self._report_rows)
        lay.addWidget(hline())
        lay.addWidget(SectionHeader("Console"))
        self._panel = InstanceLogPanel()
        lay.addWidget(self._panel)
        lay.addStretch()
        return page

    @staticmethod
    def _clear(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()

    def _refresh_reports(self) -> None:
        folders = self.folders()
        self._where.setText("Report folders: " + " · ".join(str(f) for f in folders))
        refs = list_reports(folders)
        self._clear(self._report_rows)
        listed = set()
        for ref in refs:
            listed.add(Path(ref.path))
            self._report_rows.addWidget(self._report_card(ref))
        if not refs:
            self._report_rows.addWidget(Caption(
                "No reports yet — create a folder with a report.py inside a "
                "report folder (see reports/README.md) and hit Refresh."))
        # a report deleted / renamed on disk while running stays controllable
        orphans = [i for p, i in self._instances.items() if p not in listed]
        if orphans:
            card = Card()
            title = QLabel("Running, no longer on disk")
            title.setStyleSheet("font-weight: 600;")
            card.body.addWidget(title)
            for inst in orphans:
                card.body.addLayout(self._instance_line(inst))
            self._report_rows.addWidget(card)

    def _report_card(self, ref) -> Card:
        card = Card()
        inst = self._instances.get(Path(ref.path))
        alive = inst is not None and inst.is_alive()
        busy = Path(ref.path) in self._busy

        head = QHBoxLayout()
        head.setSpacing(8)
        name = QLabel(ref.name)
        name.setStyleSheet("font-weight: 600; font-size: 14px;")
        head.addWidget(name)
        if alive:
            badge = QLabel(f"● running  :{inst.port}")
            badge.setStyleSheet(f"color: {STATE_COLORS[inst.state]}; font-size: 12px;")
            head.addWidget(badge)
        head.addStretch()

        run = QPushButton("Run")
        run.setProperty("primary", True)
        run.setEnabled(not alive and not busy)
        run.setToolTip("Already running — use Open / Restart" if alive else
                       "Start the report and open it in the scripts browser")
        run.clicked.connect(lambda _=False, r=ref: self._run(r))
        head.addWidget(run)

        folder_btn = QPushButton("Open folder")
        folder_btn.clicked.connect(lambda _=False, p=ref.path: _open_folder(p))
        head.addWidget(folder_btn)

        why = ("Stop the report first" if alive else
               "A job is running on this report" if busy else "")
        archive_btn = QPushButton("Archive")
        archive_btn.setEnabled(not alive and not busy)
        archive_btn.setToolTip(why or "Zip the report into _archive/ and remove "
                                      "the folder (restorable from the Archive tab)")
        archive_btn.clicked.connect(lambda _=False, r=ref: self._archive(r))
        head.addWidget(archive_btn)

        delete_btn = _danger(QPushButton("Delete"))
        delete_btn.setEnabled(not alive and not busy)
        delete_btn.setToolTip(why or "Permanently delete the report folder")
        delete_btn.clicked.connect(lambda _=False, r=ref: self._delete_report(r))
        head.addWidget(delete_btn)
        card.body.addLayout(head)

        card.body.addWidget(Caption(
            f"{ref.path}  —  modified {ref.modified:%Y-%m-%d %H:%M}"))
        if inst is not None:
            card.body.addLayout(self._instance_line(inst))
        return card

    def _instance_line(self, inst: ReportInstance) -> QHBoxLayout:
        line = QHBoxLayout()
        line.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {STATE_COLORS[inst.state]};")
        lbl = QLabel(f"{inst.label}   :{inst.port}   ({inst.state})")
        line.addWidget(dot)
        line.addWidget(lbl)
        line.addStretch()
        log_btn = QPushButton("Log")
        log_btn.clicked.connect(lambda _=False, i=inst: self._panel.select_instance(i))
        line.addWidget(log_btn)
        restart = QPushButton("Restart")
        restart.setToolTip("Relaunch on the same port — the open tab reconnects")
        restart.clicked.connect(lambda _=False, i=inst: i.restart())
        line.addWidget(restart)
        open_btn = QPushButton("Open")
        open_btn.setEnabled(inst.is_alive())
        open_btn.clicked.connect(lambda _=False, i=inst: open_in_scripts_browser(i.url))
        line.addWidget(open_btn)
        if inst.is_alive():
            stop = QPushButton("Stop")
            stop.clicked.connect(lambda _=False, i=inst: i.stop())
            line.addWidget(stop)
        else:
            dismiss = QPushButton("Dismiss")
            dismiss.clicked.connect(lambda _=False, i=inst: self._dismiss(i))
            line.addWidget(dismiss)
        return line

    def _run(self, ref) -> None:
        key = Path(ref.path)
        old = self._instances.get(key)
        if old is not None:
            if old.is_alive():
                return
            self._dismiss(old, refresh=False)
        taken = {i.port for i in self._instances.values() if i.is_alive()}
        try:
            port = find_free_port(exclude=taken)
        except RuntimeError as e:
            QMessageBox.critical(self, "Reports", str(e))
            return
        n = self._counters.get(key, 0) + 1
        self._counters[key] = n
        inst = ReportInstance(ref, n, port, parent=self)
        inst.state_changed.connect(lambda: self.refresh(force=True))
        inst.ready.connect(lambda i=inst: open_in_scripts_browser(i.url))
        self._instances[key] = inst
        self._panel.add_instance(inst)
        inst.start()
        self.refresh(force=True)

    def _dismiss(self, inst: ReportInstance, refresh: bool = True) -> None:
        if inst.is_alive():
            return
        self._panel.remove_instance(inst)
        for k, v in list(self._instances.items()):
            if v is inst:
                del self._instances[k]
        inst.deleteLater()
        if refresh:
            self.refresh(force=True)

    def _is_running(self, path: Path) -> bool:
        inst = self._instances.get(Path(path))
        return inst is not None and inst.is_alive()

    def _archive(self, ref) -> None:
        if self._is_running(ref.path) or self._reports_job.busy():
            return
        answer = QMessageBox.question(
            self, "Archive report",
            f"Archive “{ref.name}”?\n\nThe folder is zipped into "
            f"{Path(ref.folder) / ARCHIVE_DIR}, verified, and then removed. "
            "Restore it any time from the Archive tab.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return
        self._run_job(self._reports_job, f"Archiving {ref.name}…",
                      arch.archive_report, ref.path, busy_key=Path(ref.path),
                      on_done=lambda z: self._banner.show_message(
                          "success", f"Archived to {z}"))

    def _delete_report(self, ref) -> None:
        if self._is_running(ref.path) or self._reports_job.busy():
            return
        dlg = DeleteConfirmDialog("this report folder", str(ref.path), self)
        if dlg.exec() != DeleteConfirmDialog.Accepted:
            return
        self._run_job(self._reports_job, f"Deleting {ref.name}…",
                      _delete_job, ref.path, busy_key=Path(ref.path),
                      on_done=lambda _r: self._banner.show_message(
                          "success", f"Deleted {ref.path}"))

    # ══ Archive tab ════════════════════════════════════════════════════════════
    def _build_archive_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        pin_minimum_height(page)
        head = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda: self.refresh(force=True))
        head.addWidget(refresh)
        head.addWidget(Caption("Archived reports from every report folder's "
                               "_archive/ — newest first."), 1)
        lay.addLayout(head)
        self._archive_job = _JobStrip()
        lay.addWidget(self._archive_job)
        self._archive_rows = QVBoxLayout()
        self._archive_rows.setSpacing(10)
        lay.addLayout(self._archive_rows)
        lay.addStretch()
        return page

    def _refresh_archives(self) -> None:
        refs = list_archives(self.folders())
        self._clear(self._archive_rows)
        for ref in refs:
            self._archive_rows.addWidget(self._archive_card(ref))
        if not refs:
            self._archive_rows.addWidget(Caption("No archived reports."))

    def _archive_card(self, ref) -> Card:
        card = Card()
        busy = Path(ref.path) in self._busy or self._archive_job.busy()
        head = QHBoxLayout()
        name = QLabel(ref.name)
        name.setStyleSheet("font-weight: 600; font-size: 14px;")
        head.addWidget(name)
        when = ref.archived_at.strftime("%Y-%m-%d %H:%M:%S") if ref.archived_at else "?"
        head.addWidget(Caption(f"archived {when} · {human_size(ref.size)}"))
        head.addStretch()
        restore = QPushButton("Restore")
        restore.setProperty("primary", True)
        restore.setEnabled(not busy)
        restore.clicked.connect(lambda _=False, r=ref: self._restore(r))
        head.addWidget(restore)
        open_btn = QPushButton("Open archive folder")
        open_btn.clicked.connect(lambda _=False, p=ref.path.parent: _open_folder(p))
        head.addWidget(open_btn)
        delete = _danger(QPushButton("Delete"))
        delete.setEnabled(not busy)
        delete.clicked.connect(lambda _=False, r=ref: self._delete_archive(r))
        head.addWidget(delete)
        card.body.addLayout(head)
        card.body.addWidget(Caption(f"{ref.path}  —  from {ref.folder}"))
        return card

    def _restore(self, ref) -> None:
        if self._archive_job.busy():
            return
        target = Path(ref.folder) / ref.name
        if target.exists():
            QMessageBox.warning(
                self, "Restore", f"A folder named “{ref.name}” already exists in "
                f"{ref.folder}.\n\nRename or remove it first — restore never "
                "overwrites or merges.")
            return
        self._run_job(self._archive_job, f"Restoring {ref.name}…",
                      arch.restore_archive, ref.path, busy_key=Path(ref.path),
                      on_done=lambda p: self._banner.show_message(
                          "success", f"Restored to {p}"))

    def _delete_archive(self, ref) -> None:
        if self._archive_job.busy():
            return
        dlg = DeleteConfirmDialog("this archive", ref.path.name, self)
        if dlg.exec() != DeleteConfirmDialog.Accepted:
            return
        self._run_job(self._archive_job, f"Deleting {ref.path.name}…",
                      _delete_job, ref.path, busy_key=Path(ref.path),
                      on_done=lambda _r: self._banner.show_message(
                          "success", f"Deleted {ref.path.name}"))

    # ══ Compress tab ═══════════════════════════════════════════════════════════
    def _build_compress_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        pin_minimum_height(page)
        lay.addWidget(Caption(
            "Turn an Optimizer run copied into a report into a COMPACT run: one "
            "row per parameter combination with every metric the Optimizer "
            "heatmap shows (pre-cost), no trades. The kit heatmap shows both "
            "identically. The source is never modified — delete it by hand once "
            "you are happy."))

        box = QVBoxLayout()
        box.addWidget(QLabel("Source run (inside a report folder)"))
        src_row = QHBoxLayout()
        self._source = QComboBox()
        self._source.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self._source, 1)
        browse_src = QPushButton("Browse…")
        browse_src.clicked.connect(self._browse_source)
        src_row.addWidget(browse_src)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(lambda: self.refresh(force=True))
        src_row.addWidget(rescan)
        box.addLayout(src_row)
        self._source_note = Banner()
        box.addWidget(self._source_note)

        box.addWidget(QLabel("Destination folder (inside a report folder) — "
                             "the run is written to <destination>/<run name>/"))
        dst_row = QHBoxLayout()
        self._dest = QLineEdit()
        self._dest.setPlaceholderText("e.g. …/reports/my_report/data")
        dst_row.addWidget(self._dest, 1)
        browse_dst = QPushButton("Browse…")
        browse_dst.clicked.connect(self._browse_dest)
        dst_row.addWidget(browse_dst)
        box.addLayout(dst_row)

        go_row = QHBoxLayout()
        self._compress_btn = QPushButton("Compress")
        self._compress_btn.setProperty("primary", True)
        self._compress_btn.clicked.connect(self._compress)
        go_row.addWidget(self._compress_btn)
        go_row.addStretch()
        box.addLayout(go_row)
        lay.addWidget(wrap_card(box))

        self._compress_job = _JobStrip()
        lay.addWidget(self._compress_job)
        self._compress_result = Banner()
        lay.addWidget(self._compress_result)
        lay.addStretch()
        return page

    def _refresh_sources(self) -> None:
        current = self._source.currentData()
        folders = self.folders()
        runs = compact.find_runs(folders)
        self._source.blockSignals(True)
        self._source.clear()
        for path, kind in runs:
            owner = owning_reports_folder(path, folders)
            rel = path.relative_to(owner) if owner is not None else path
            text = f"{rel.as_posix()}   [{owner.name if owner else '?'}]"
            if kind == "compact":
                text += "   (compact)"
            self._source.addItem(text, (str(path), kind))
        if not runs:
            self._source.addItem("— no Optimizer runs inside the report folders —",
                                 None)
        if current is not None:
            for i in range(self._source.count()):
                if self._source.itemData(i) == current:
                    self._source.setCurrentIndex(i)
        self._source.blockSignals(False)
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        data = self._source.currentData()
        self._source_note.clear_message()
        if data and data[1] == "compact":
            self._source_note.show_message(
                "info", "This run is already compact — there is nothing to compress.")
        self._compress_btn.setEnabled(bool(data) and data[1] == "full"
                                      and not self._compress_job.busy())

    def _start_dir(self) -> str:
        folders = self.folders()
        return str(folders[0]) if folders else ""

    def _browse_source(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Optimizer run folder",
                                             self._start_dir())
        if not d:
            return
        folders = self.folders()
        owner = owning_reports_folder(d, folders)
        if owner is None or is_in_archive(d, owner):
            QMessageBox.warning(self, "Compress",
                                "The source must be inside a report folder "
                                "(and not in _archive/).")
            return
        kind = compact.run_kind(d)
        if kind is None:
            QMessageBox.warning(self, "Compress", "That folder is not an Optimizer "
                                "run (needs trades.parquet + meta.json).")
            return
        data = (str(Path(d)), kind)
        for i in range(self._source.count()):
            if self._source.itemData(i) == data:
                self._source.setCurrentIndex(i)
                return
        self._source.addItem(f"{Path(d).relative_to(owner).as_posix()}   "
                             f"[{owner.name}]" + ("   (compact)" if kind == "compact"
                                                  else ""), data)
        self._source.setCurrentIndex(self._source.count() - 1)

    def _browse_dest(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Destination folder",
                                             self._dest.text() or self._start_dir())
        if not d:
            return
        owner = owning_reports_folder(d, self.folders())
        if owner is None or is_in_archive(d, owner):
            QMessageBox.warning(self, "Compress",
                                "The destination must be inside a report folder "
                                "(and not in _archive/).")
            return
        self._dest.setText(str(Path(d)))

    def _compress(self) -> None:
        data = self._source.currentData()
        dest = self._dest.text().strip()
        self._compress_result.clear_message()
        if not data or data[1] != "full":
            return
        if not dest:
            self._compress_result.show_message("error", "Pick a destination folder.")
            return
        source = Path(data[0])
        replace = False
        target = compact.compress_target(source, dest)
        if target.exists():
            answer = QMessageBox.question(
                self, "Replace?", f"{target} already exists.\n\nReplace it with "
                "the new compact run? (Nothing else in the destination is touched.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
            replace = True

        def done(result):
            self._compress_result.show_message(
                "success",
                f"Compact run written to {result['target']} — "
                f"{result['n_combinations']} combinations · source "
                f"{human_size(result['source_bytes'])} → result "
                f"{human_size(result['result_bytes'])}.")

        self._compress_btn.setEnabled(False)
        self._run_job(self._compress_job, f"Compressing {source.name}…",
                      compact.compress_run, source, Path(dest), self.folders(),
                      replace=replace, busy_key=source, on_done=done)

    # ══ close policy ═══════════════════════════════════════════════════════════
    def closeEvent(self, event) -> None:
        alive = [i for i in self._instances.values() if i.is_alive()]
        if alive:
            answer = QMessageBox.question(
                self, "Reports running",
                f"{len(alive)} report process(es) are still running. Stop them "
                "and close this window?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for inst in alive:
                inst.stop()
            for inst in alive:
                inst.wait_finished(1500)
        super().closeEvent(event)


def _delete_job(path, on_progress=None):
    """Worker body for a permanent delete (progress = busy state only)."""
    if on_progress is not None:
        on_progress(0, 1, f"Deleting {Path(path).name}…")
    arch.delete_path(path)
    return path
