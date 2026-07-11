"""
Save Trades / "Go to Analytics" / "Go to Monte Carlo" buttons — shared by
the Backtester save row and the Optimizer cell detail.

Save Trades writes the host's CURRENT filtered trades to the data root's
trades/ folder under the host's base name (filter-aware dedup, _N suffix on
name collisions). The Go-to buttons write them to the root's temp/ folder
({ASSET}_temp_file_N.parquet; identical rows + filter metadata reuse the
existing file) and open the target module in a new window with that file
preselected.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from modules.common.backend.data_roots import temp_dir, trades_dir
from modules.common.backend.trade_files import save_temp_trades, save_trades

# Strong refs to spawned windows. MODULE-level on purpose: host windows get
# WA_DeleteOnClose from the main menu, so closing one drops its Python
# wrapper — an instance-level list would be GC'd with it and PySide6 would
# then delete the ownerless spawned top-level windows. Module scope lets
# spawned Analytics/MC windows outlive their spawner. (Known caveat: the
# main menu's close-all doesn't track these windows; the app quits when the
# user closes the last one.)
_SPAWNED_WINDOWS: list = []


def _forget_spawned(window) -> None:
    if window in _SPAWNED_WINDOWS:
        _SPAWNED_WINDOWS.remove(window)


class TradeActionsRow(QWidget):
    """
    The three buttons side by side, no margins/stretches — drop into any
    layout.

    The host supplies:
    - context_provider() -> dict | None. None (or empty trades) = nothing
      ready, clicks no-op. Keys: trades (DataFrame; a derived day_type
      column is stripped before saving), asset (ticker, first filename
      token downstream), root (the data root receiving the file), save_name
      (base filename for Save Trades, no extension), filtered (bool),
      day_types (list), trade_types ("all" | list).
    - banner: a host-owned Banner for success/error messages.
    """

    def __init__(self, settings, context_provider, banner, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._context_provider = context_provider
        self._banner = banner

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        save_btn = QPushButton("Save Trades")
        save_btn.clicked.connect(self._on_save)
        analytics_btn = QPushButton("Go to Analytics")
        analytics_btn.clicked.connect(self._on_go_analytics)
        mc_btn = QPushButton("Go to Monte Carlo")
        mc_btn.clicked.connect(self._on_go_monte_carlo)
        row.addWidget(save_btn)
        row.addWidget(analytics_btn)
        row.addWidget(mc_btn)

    # ── shared context handling ───────────────────────────────────────────────
    def _save_ready_context(self) -> dict | None:
        """The host context with save-ready trades (day_type stripped), or
        None when nothing is ready (buttons are hidden in that state)."""
        ctx = self._context_provider()
        if ctx is None:
            return None
        trades = ctx["trades"]
        if trades is None or trades.empty:
            return None
        # Strip day_type before saving — it's derived, not strategy output
        save_cols = [c for c in trades.columns if c != "day_type"]
        return {**ctx, "trades": trades[save_cols]}

    # ── save trades (regular save into <root>/trades/) ────────────────────────
    def _on_save(self) -> None:
        ctx = self._save_ready_context()
        if ctx is None:
            return
        try:
            result = save_trades(
                trades_dir(ctx["root"]), ctx["trades"], ctx["save_name"],
                ctx["filtered"], ctx["day_types"], ctx["trade_types"])
        except Exception as e:  # noqa: BLE001 — disk errors surface in the banner
            self._banner.show_message("error", f"Could not save trades: {e}")
            return
        if result is None:
            self._banner.show_message(
                "info", "Identical trades file already exists — not saved.")
        else:
            self._banner.show_message("success", f"Saved to {result}")

    # ── go to Analytics / Monte Carlo (temp handoff) ──────────────────────────
    def _save_temp_file(self):
        """Write the host's current filtered trades to <root>/temp/ and
        return the path, or None on failure (error shown in the banner)."""
        ctx = self._save_ready_context()
        if ctx is None:
            return None
        try:
            return save_temp_trades(
                temp_dir(ctx["root"]), ctx["trades"], ctx["asset"],
                ctx["filtered"], ctx["day_types"], ctx["trade_types"])
        except Exception as e:  # noqa: BLE001 — disk errors surface in the banner
            self._banner.show_message(
                "error", f"Could not write temp trades file: {e}")
            return None

    def _spawn_module_window(self, window_cls, title: str, path) -> None:
        try:
            window = window_cls(self._settings, initial_trades=path)
        except Exception as e:  # noqa: BLE001 — a broken module must not kill us
            self._banner.show_message("error", f"Could not open {title}: {e}")
            return
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.destroyed.connect(lambda _=None, w=window: _forget_spawned(w))
        _SPAWNED_WINDOWS.append(window)
        window.showMaximized()
        self._banner.show_message(
            "success", f"Using trades file {path} — opened in {title}.")

    def _on_go_analytics(self) -> None:
        path = self._save_temp_file()
        if path is None:
            return
        # lazy import — no import-time coupling between module windows
        from modules.analytics.window import AnalyticsWindow
        self._spawn_module_window(AnalyticsWindow, "Analytics", path)

    def _on_go_monte_carlo(self) -> None:
        path = self._save_temp_file()
        if path is None:
            return
        from modules.monte_carlo.window import MonteCarloWindow
        self._spawn_module_window(MonteCarloWindow, "Monte Carlo", path)
