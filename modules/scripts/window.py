"""
ScriptsWindow — quick-script launcher.

Lists every .py in the configured script folders (Settings gear → Script
folders; in-repo default scripts/). A script carrying the `# app: streamlit`
marker (or `STREAMLIT = True`) in its first 30 lines launches as its own
`streamlit run` server on a free port and opens in the dedicated scripts
browser — its own window on first run, a new tab there on later runs, never
a tab in the user's current browser; a plain script runs as `python -u`
with its output in the shared console.
Multiple instances — including of the same script — run side by side; the
console panel's chips switch between their buffered logs. Closing the window
kills every process it spawned (after confirmation).
"""

from pathlib import Path

from PySide6.QtCore import QProcess, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout)

from modules.common.backend import plugins
from modules.common.ui import theme
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.widgets import Caption, Card, SectionHeader, hline
from .backend.browser import find_app_browser, launch_args
from .backend.ports import find_free_port
from .backend.scan import is_streamlit_script, script_mtime
from .log_panel import STATE_COLORS, InstanceLogPanel
from .process_manager import ScriptInstance


class ScriptsWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(
            settings, "Scripts",
            "Quick research scripts — a `# app: streamlit` marker launches a "
            "browser app, anything else runs as plain Python in the console.",
            parent)
        self._instances: list[ScriptInstance] = []
        self._counters: dict[Path, int] = {}
        self._refs: list[plugins.PluginRef] = []

        header = QHBoxLayout()
        header.setSpacing(8)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        self._sort = QComboBox()
        self._sort.addItems(["Date modified", "Name"])
        self._sort.currentIndexChanged.connect(lambda _i: self._refresh())
        header.addWidget(refresh)
        header.addWidget(QLabel("Sort:"))
        header.addWidget(self._sort)
        folders = " · ".join(
            str(d) for d in self.settings.plugin_dirs("scripts"))
        header.addSpacing(10)
        header.addWidget(Caption(f"Folders: {folders}"), stretch=1)
        self.content.addLayout(header)

        self._rows_holder = QVBoxLayout()
        self._rows_holder.setSpacing(10)
        self.content.addLayout(self._rows_holder)

        self.content.addWidget(hline())
        self.content.addWidget(SectionHeader("Console"))
        self._panel = InstanceLogPanel()
        self.content.addWidget(self._panel)
        self.content.addStretch()

        self._refresh()

    # ── scan / rows ───────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        refs = plugins.list_plugins(self.settings.plugin_dirs("scripts"))
        if self._sort.currentIndex() == 0:
            refs.sort(key=lambda r: script_mtime(r.path), reverse=True)
        else:
            refs.sort(key=lambda r: r.label.lower())
        self._refs = refs
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        while self._rows_holder.count():
            item = self._rows_holder.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()      # no ghost frame before deferred delete
                item.widget().deleteLater()

        listed_paths = set()
        for ref in self._refs:
            listed_paths.add(Path(ref.path))
            self._rows_holder.addWidget(self._make_row(ref))
        if not self._refs:
            self._rows_holder.addWidget(Caption(
                "No scripts found — drop a .py file into a script folder and "
                "hit Refresh."))

        orphans = [i for i in self._instances
                   if i.script_path not in listed_paths]
        if orphans:
            card = Card()
            title = QLabel("Removed scripts")
            title.setStyleSheet("font-weight: 600;")
            card.body.addWidget(title)
            card.body.addWidget(Caption(
                "These instances were launched from files no longer in the "
                "scanned folders."))
            for inst in orphans:
                card.body.addLayout(self._make_instance_line(inst))
            self._rows_holder.addWidget(card)

    def _make_row(self, ref) -> Card:
        card = Card()
        streamlit = is_streamlit_script(ref.path)

        head = QHBoxLayout()
        head.setSpacing(10)
        name = QLabel(ref.label)
        name.setStyleSheet("font-weight: 600; font-size: 14px;")
        tag = QLabel("streamlit" if streamlit else "python")
        tag.setStyleSheet(
            f"color: {theme.ACCENT_SOFT if streamlit else theme.TEXT_MUTED}; "
            f"background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 8px; padding: 1px 8px; font-size: 11px;")
        head.addWidget(name)
        head.addWidget(tag)
        head.addStretch()

        running = sum(1 for i in self._instances
                      if i.script_path == Path(ref.path) and i.is_alive())
        if running:
            badge = QLabel(f"{running} running")
            badge.setStyleSheet(
                f"color: {STATE_COLORS[ScriptInstance.RUNNING]}; "
                f"font-size: 12px;")
            head.addWidget(badge)

        run = QPushButton("Run")
        run.setProperty("primary", True)
        run.clicked.connect(lambda _=False, r=ref: self._run(r))
        head.addWidget(run)

        if any(i.script_path == Path(ref.path) for i in self._instances):
            kill = QPushButton("Kill")
            kill.setToolTip("Instantly stop AND clear every instance of "
                            "this script (chips and logs included)")
            kill.setStyleSheet(
                "QPushButton { color: #ff9d9d; border-color: #6b2f2f; } "
                "QPushButton:hover { border-color: #c94444; }")
            kill.clicked.connect(lambda _=False, r=ref: self._kill_script(r))
            head.addWidget(kill)
        card.body.addLayout(head)

        card.body.addWidget(Caption(
            f"{ref.path}  —  modified {script_mtime(ref.path):%Y-%m-%d %H:%M}"))

        for inst in [i for i in self._instances
                     if i.script_path == Path(ref.path)]:
            card.body.addLayout(self._make_instance_line(inst))
        return card

    def _make_instance_line(self, inst: ScriptInstance) -> QHBoxLayout:
        line = QHBoxLayout()
        line.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {STATE_COLORS[inst.state]};")
        text = f"#{inst.instance_no}"
        if inst.port:
            text += f"   :{inst.port}"
        lbl = QLabel(text)
        line.addWidget(dot)
        line.addWidget(lbl)
        line.addStretch()

        log_btn = QPushButton("Log")
        log_btn.clicked.connect(
            lambda _=False, i=inst: self._panel.select_instance(i))
        line.addWidget(log_btn)

        restart = QPushButton("Restart")
        restart.setToolTip(
            "Relaunch with the current file on the same port — the open "
            "browser tab reconnects, no new tab is opened")
        restart.clicked.connect(lambda _=False, i=inst: i.restart())
        line.addWidget(restart)

        if inst.url:
            open_btn = QPushButton("Open")
            open_btn.setEnabled(inst.is_alive())
            open_btn.clicked.connect(
                lambda _=False, i=inst: self._open_url(i.url))
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

    # ── launching ─────────────────────────────────────────────────────────────
    def _run(self, ref) -> None:
        # kind is re-detected at Run time, so editing a marker takes effect
        # without a Refresh
        kind = "streamlit" if is_streamlit_script(ref.path) else "python"
        port = None
        if kind == "streamlit":
            taken = {i.port for i in self._instances
                     if i.port and i.is_alive()}
            try:
                port = find_free_port(exclude=taken)
            except RuntimeError as e:
                QMessageBox.critical(self, "Scripts", str(e))
                return
        key = Path(ref.path)
        n = self._counters.get(key, 0) + 1
        self._counters[key] = n

        inst = ScriptInstance(ref, kind, n, port=port, parent=self)
        inst.state_changed.connect(self._rebuild_rows)
        if kind == "streamlit":
            inst.ready.connect(lambda i=inst: self._open_url(i.url))
        self._instances.append(inst)
        self._panel.add_instance(inst)
        inst.start()
        self._rebuild_rows()

    @staticmethod
    def _open_url(url: str) -> None:
        """Open in the dedicated scripts browser: the first run opens its
        own window, later runs land as new tabs in it — the user's own
        browser (and its tabs) is never touched. Plain default-browser
        fallback only if no Chrome/Edge exists."""
        browser = find_app_browser()
        if browser is not None:
            QProcess.startDetached(str(browser), launch_args(url))
        else:
            QDesktopServices.openUrl(QUrl(url))

    def _dismiss(self, inst: ScriptInstance) -> None:
        if inst.is_alive():
            return
        self._panel.remove_instance(inst)
        self._instances.remove(inst)
        inst.deleteLater()
        self._rebuild_rows()

    def _kill_script(self, ref) -> None:
        """One-click stop + dismiss of every instance of this script."""
        key = Path(ref.path)
        targets = [i for i in self._instances if i.script_path == key]
        for inst in targets:
            inst.stop()
        for inst in targets:
            inst.wait_finished(2000)
        for inst in targets:
            self._panel.remove_instance(inst)
            self._instances.remove(inst)
            inst.deleteLater()
        self._rebuild_rows()

    # ── close policy ──────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        alive = [i for i in self._instances if i.is_alive()]
        if alive:
            answer = QMessageBox.question(
                self, "Scripts running",
                f"{len(alive)} script process(es) are still running. "
                "Stop them and close this window?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for inst in alive:
                inst.stop()
            for inst in alive:
                inst.wait_finished(1500)
        super().closeEvent(event)
