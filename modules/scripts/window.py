"""
ScriptsWindow — quick-script launcher.

Lists every .py in the configured script folders (Settings gear → Script
folders; in-repo default scripts/). A script carrying the `# app: streamlit`
marker (or `STREAMLIT = True`) in its first 30 lines launches as its own
`streamlit run` server on a free port and opens in the dedicated scripts
browser — its own window on first run, a new tab there on later runs, never
a tab in the user's current browser; a plain script runs as `python -u`
with its output in the shared console.
Script folders may nest arbitrarily: subfolders show as folder rows (styled
distinctly from script cards), clicking one enters it, and the breadcrumb
bar / Back button walk back out. Instances keep running while you browse —
ones launched from outside the current folder stay controllable in a
"Running elsewhere" card and in the console chips.
Multiple instances — including of the same script — run side by side; the
console panel's chips switch between their buffered logs. Closing the window
kills every process it spawned (after confirmation).
"""

from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout)

from modules.common.backend import plugins
from modules.common.ui import theme
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.widgets import Caption, Card, SectionHeader, hline
from .backend.browser import find_app_browser, launch_args
from .backend.ports import find_free_port
from .backend.scan import (folder_summary, is_streamlit_script, list_folder,
                           script_mtime)
from .log_panel import STATE_COLORS, InstanceLogPanel
from .process_manager import ScriptInstance


class _FolderRow(QFrame):
    """A subfolder entry — deliberately NOT a Card: accent left edge,
    folder glyph and pointing-hand cursor make it read as navigation,
    not as a runnable script."""

    def __init__(self, path: Path, on_open, parent=None):
        super().__init__(parent)
        self.setObjectName("folderRow")
        self._path = path
        self._on_open = on_open
        self.setCursor(Qt.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        icon = QLabel("\U0001f4c1")
        icon.setStyleSheet("font-size: 15px; background: transparent;")
        name = QLabel(path.name)
        name.setStyleSheet("font-weight: 600; font-size: 14px; "
                           f"color: {theme.ACCENT_SOFT}; background: transparent;")
        n_dirs, n_scripts = folder_summary(path)
        bits = ([f"{n_dirs} folder{'s' if n_dirs != 1 else ''}"] if n_dirs else [])
        bits.append(f"{n_scripts} script{'s' if n_scripts != 1 else ''}")
        info = Caption(" · ".join(bits))
        arrow = QLabel("Open ›")
        arrow.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
        lay.addWidget(icon)
        lay.addWidget(name)
        lay.addWidget(info)
        lay.addStretch()
        lay.addWidget(arrow)

        self.setStyleSheet(f"""
            QFrame#folderRow {{
                background: {theme.SURFACE};
                border: 1px solid {theme.BORDER};
                border-left: 3px solid {theme.ACCENT};
                border-radius: 8px;
            }}
            QFrame#folderRow:hover {{
                background: {theme.SURFACE_2};
                border-color: {theme.BORDER_LIGHT};
                border-left-color: {theme.ACCENT_HOVER};
            }}""")

    def mouseReleaseEvent(self, event) -> None:
        if (event.button() == Qt.LeftButton
                and self.rect().contains(event.position().toPoint())):
            self._on_open(self._path)
        super().mouseReleaseEvent(event)


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
        self._folders: list[Path] = []
        self._cwd: Path | None = None       # None = merged root view

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
        header.addSpacing(10)
        self._where = Caption("")
        header.addWidget(self._where, stretch=1)
        self.content.addLayout(header)

        self._nav = QHBoxLayout()
        self._nav.setSpacing(4)
        self.content.addLayout(self._nav)

        self._rows_holder = QVBoxLayout()
        self._rows_holder.setSpacing(10)
        self.content.addLayout(self._rows_holder)

        self.content.addWidget(hline())
        self.content.addWidget(SectionHeader("Console"))
        self._panel = InstanceLogPanel()
        self.content.addWidget(self._panel)
        self.content.addStretch()

        self._refresh()

    # ── navigation ────────────────────────────────────────────────────────────
    def _roots(self) -> list[Path]:
        return [Path(d) for d in self.settings.plugin_dirs("scripts")]

    def _owning_root(self, path: Path) -> Path | None:
        for root in self._roots():
            try:
                path.relative_to(root)
                return root
            except ValueError:
                continue
        return None

    def _go_to(self, target: Path | None) -> None:
        self._cwd = target
        self._refresh()

    def _go_back(self) -> None:
        if self._cwd is None:
            return
        parent = self._cwd.parent
        self._go_to(None if parent in self._roots() else parent)

    def _rebuild_nav(self) -> None:
        while self._nav.count():
            item = self._nav.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()      # no ghost crumb before deferred delete
                item.widget().deleteLater()

        back = QPushButton("← Back")
        back.setEnabled(self._cwd is not None)
        back.clicked.connect(self._go_back)
        self._nav.addWidget(back)
        self._nav.addSpacing(8)

        def crumb(text, target, current=False):
            btn = QPushButton(text)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            color = theme.TEXT if current else theme.TEXT_MUTED
            weight = "600" if current else "400"
            btn.setStyleSheet(
                f"QPushButton {{ border: none; background: transparent; "
                f"color: {color}; font-weight: {weight}; padding: 2px 4px; }} "
                f"QPushButton:hover {{ color: {theme.ACCENT_SOFT}; }}")
            btn.clicked.connect(lambda _=False, t=target: self._go_to(t))
            self._nav.addWidget(btn)

        crumb("⌂ scripts", None, current=self._cwd is None)
        if self._cwd is not None:
            root = self._owning_root(self._cwd)
            parts = (self._cwd.relative_to(root).parts
                     if root is not None else (self._cwd.name,))
            base = root if root is not None else self._cwd.parent
            for i, part in enumerate(parts):
                sep = QLabel("›")
                sep.setStyleSheet(f"color: {theme.TEXT_MUTED};")
                self._nav.addWidget(sep)
                crumb(part, Path(base, *parts[:i + 1]),
                      current=i == len(parts) - 1)
        self._nav.addStretch()

    # ── scan / rows ───────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        if self._cwd is None:
            roots = self._roots()
            self._folders = [f for root in roots for f in list_folder(root)[0]]
            refs = plugins.list_plugins(roots)
            self._where.setText(
                "Folders: " + " · ".join(str(r) for r in roots))
        else:
            folders, files = list_folder(self._cwd)
            self._folders = folders
            refs = [plugins.PluginRef(p.stem, p, self._cwd, p.stem)
                    for p in files]
            self._where.setText(str(self._cwd))
        if self._sort.currentIndex() == 0:
            refs.sort(key=lambda r: script_mtime(r.path), reverse=True)
        else:
            refs.sort(key=lambda r: r.label.lower())
        self._refs = refs
        self._rebuild_nav()
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        while self._rows_holder.count():
            item = self._rows_holder.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()      # no ghost frame before deferred delete
                item.widget().deleteLater()

        for folder in self._folders:
            self._rows_holder.addWidget(_FolderRow(folder, self._go_to))

        listed_paths = set()
        for ref in self._refs:
            listed_paths.add(Path(ref.path))
            self._rows_holder.addWidget(self._make_row(ref))
        if not self._refs and not self._folders:
            self._rows_holder.addWidget(Caption(
                "Nothing here — drop a .py file (or a folder of them) into a "
                "script folder and hit Refresh."))

        elsewhere = [i for i in self._instances
                     if i.script_path not in listed_paths]
        if elsewhere:
            card = Card()
            title = QLabel("Running elsewhere")
            title.setStyleSheet("font-weight: 600;")
            card.body.addWidget(title)
            card.body.addWidget(Caption(
                "Instances launched from other folders (or from files that "
                "were removed) — still yours to control."))
            for inst in elsewhere:
                card.body.addLayout(
                    self._make_instance_line(inst, with_name=True))
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

    def _make_instance_line(self, inst: ScriptInstance,
                            with_name: bool = False) -> QHBoxLayout:
        line = QHBoxLayout()
        line.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {STATE_COLORS[inst.state]};")
        text = inst.label if with_name else f"#{inst.instance_no}"
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
