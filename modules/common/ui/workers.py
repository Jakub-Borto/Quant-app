"""
Background-job infrastructure: run a backend function on Qt's thread pool and
report progress/result/error back to the GUI thread via signals.

The signal shapes mirror the project's universal progress callback
`on_progress(current, total, message)` (transforms + optimizer engine).

Cancellation contract (identical semantics to a Streamlit Stop): cancel()
sets a flag, and the NEXT on_progress call raises JobCancelled *inside the
callback*, so the exception propagates out of run_all()/run_grid() — the
optimizer engine's `finally: executor.shutdown(cancel_futures=True)` relies
on exactly this. Functions without a progress callback (strategy.run, sizers,
MC run) are not cancellable — windows disable Cancel for those.
"""

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class JobCancelled(Exception):
    """Raised inside on_progress when the user pressed Cancel."""


class WorkerSignals(QObject):
    progress  = Signal(int, int, str)   # current, total, message
    finished  = Signal(object)          # the function's return value
    error     = Signal(str, str)        # (message, traceback text)
    cancelled = Signal()


class FunctionWorker(QRunnable):
    """
    Run `fn(*args, **kwargs)` on the global QThreadPool.

    With needs_progress=True the worker injects its own `on_progress` into
    kwargs; the callback re-emits to the GUI thread (queued) and doubles as
    the cancellation point.
    """

    def __init__(self, fn, *args, needs_progress: bool = False, **kwargs):
        super().__init__()
        self.signals = WorkerSignals()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._needs_progress = needs_progress
        self._cancel_requested = False
        self.setAutoDelete(False)   # windows keep a reference until done

    # ── control ───────────────────────────────────────────────────────────────
    def cancel(self) -> None:
        self._cancel_requested = True

    @property
    def cancellable(self) -> bool:
        return self._needs_progress

    # ── callback handed to the backend ────────────────────────────────────────
    def _on_progress(self, current, total, message: str = "") -> None:
        if self._cancel_requested:
            raise JobCancelled()
        self.signals.progress.emit(int(current), int(total), str(message))

    # ── QRunnable ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            if self._needs_progress:
                kwargs["on_progress"] = self._on_progress
            result = self._fn(*self._args, **kwargs)
        except JobCancelled:
            self.signals.cancelled.emit()
        except Exception as e:  # noqa: BLE001 — everything surfaces in the UI
            self.signals.error.emit(str(e), traceback.format_exc())
        else:
            self.signals.finished.emit(result)


def start_worker(worker: FunctionWorker) -> None:
    """Submit to the global pool (thin wrapper so call sites read clearly)."""
    QThreadPool.globalInstance().start(worker)
