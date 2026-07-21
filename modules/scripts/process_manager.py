"""
ScriptInstance — one spawned script process (Streamlit server or plain
Python), owned by a ScriptsWindow.

QProcess is Qt, so this lives on the UI side of the module (backend/ stays
Qt-free per the repo convention). The QProcess is parented to the instance
and the instance to the window, so even an abnormal window teardown kills a
surviving process via the Qt object tree (the explicit stop in the window's
closeEvent is the normal path).

Working directory is the script's OWN folder, never the repo root: the repo
root holds an untracked scratch `inspect.py` that shadows stdlib `inspect`
and breaks numpy (which Streamlit itself imports). Scripts that need repo
imports use the sys.path.append idiom shown in scripts/example_hello.py.
"""

import sys
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal


class ScriptInstance(QObject):
    STARTING = "starting"
    RUNNING = "running"
    EXITED = "exited"
    CRASHED = "crashed"

    state_changed = Signal()
    output = Signal(str)     # raw chunk, already folded into self.log
    ready = Signal()         # streamlit server is up -> open the browser

    def __init__(self, ref, kind: str, instance_no: int,
                 port: int | None = None, parent=None):
        super().__init__(parent)
        self.ref = ref
        self.kind = kind                      # "streamlit" | "python"
        self.script_path = Path(ref.path)
        self.instance_no = instance_no
        self.port = port
        self.label = f"{ref.name} #{instance_no}"
        self.url = f"http://localhost:{port}" if port else None
        self.state = self.STARTING
        self.log: deque[str] = deque(maxlen=4000)   # per-instance ring buffer
        self._pending = ""                    # partial line awaiting its "\n"
        self._stop_requested = False
        self._ready_announced = False         # open the browser only ONCE

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.setWorkingDirectory(str(self.script_path.parent))
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.started.connect(self._on_started)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.kind == "streamlit":
            args = ["-m", "streamlit", "run", str(self.script_path),
                    "--server.port", str(self.port),
                    "--server.headless", "true",
                    "--server.runOnSave", "true",   # saving the file reruns
                    "--browser.gatherUsageStats", "false"]
        else:
            # -u: unbuffered, so prints stream live instead of one flush at exit
            args = ["-u", str(self.script_path)]
        self._proc.start(sys.executable, args)

    def is_alive(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    def stop(self) -> None:
        if self.is_alive():
            self._stop_requested = True
            # kill(), not terminate(): on Windows terminate() posts WM_CLOSE,
            # a no-op for console processes. `streamlit run` serves in-process
            # (no child tree), so killing the one process is sufficient.
            self._proc.kill()

    def wait_finished(self, msecs: int = 1500) -> None:
        if self.is_alive():
            self._proc.waitForFinished(msecs)

    def restart(self) -> None:
        """Kill (if alive) and relaunch with the CURRENT file — same port,
        same chip, so an open browser tab reconnects instead of a new one
        opening (`ready` is only announced on the first start)."""
        if self.is_alive():
            self._stop_requested = True
            self._proc.kill()
            self._proc.waitForFinished(3000)
        self._stop_requested = False
        self.log.append("─── restarted ───")
        self._set_state(self.STARTING)
        self.output.emit("─── restarted ───\n")
        self.start()

    # ── log ───────────────────────────────────────────────────────────────────
    def log_text(self) -> str:
        text = "\n".join(self.log)
        if self._pending:
            text = f"{text}\n{self._pending}" if text else self._pending
        return text

    # ── internals ─────────────────────────────────────────────────────────────
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit()

    def _on_started(self) -> None:
        if self.kind == "python":
            self._set_state(self.RUNNING)

    def _on_output(self) -> None:
        text = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        self.log.extend(line.rstrip("\r") for line in lines)
        self.output.emit(text)
        if self.kind == "streamlit" and self.state == self.STARTING and (
                any("Local URL:" in line for line in lines)
                or "Local URL:" in self._pending):
            self._set_state(self.RUNNING)
            if not self._ready_announced:
                self._ready_announced = True
                self.ready.emit()

    def _on_finished(self, code: int, status) -> None:
        if self._pending:
            self.log.append(self._pending.rstrip("\r"))
            self._pending = ""
        ok = self._stop_requested or (
            status == QProcess.NormalExit and code == 0)
        self._set_state(self.EXITED if ok else self.CRASHED)

    def _on_error(self, err) -> None:
        # Crashes also fire finished(); only spawn failure never reaches it.
        if err == QProcess.FailedToStart:
            self.log.append(f"[failed to start: {self._proc.errorString()}]")
            self._set_state(self.CRASHED)
