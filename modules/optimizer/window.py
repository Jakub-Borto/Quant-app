"""
Strategy Optimizer window — QTabWidget with the three sub-modules:
New Run / Explore / Combine.

Run state (the old opt_trades / opt_meta / opt_loaded_run / opt_unsaved
session keys, plus the run's data root) lives in a small RunState object
shared by the tabs. A finished grid run calls adopt_unsaved_run(), which
replaces the old opt_switch_to_explore / opt_run_success / opt_reset_run_select
rerun dance with direct calls: store the run, switch to Explore, pin the
unsaved entry, show the success banner.
"""

import pandas as pd
from PySide6.QtWidgets import QTabWidget, QVBoxLayout

from modules.common.ui.module_window import ModuleWindowBase
from modules.optimizer.combine_tab import CombineTab
from modules.optimizer.explore_tab import ExploreTab
from modules.optimizer.new_run_tab import NewRunTab


class RunState:
    """The currently loaded/unsaved run, shared across tabs."""

    def __init__(self):
        self.trades: pd.DataFrame | None = None
        self.meta: dict | None = None
        self.loaded_run: str | None = None
        self.unsaved: bool = False
        self.run_root = None            # the data root the run belongs to


class OptimizerWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Strategy Optimizer",
                         "Sweep up to 4 strategy params, store every cell's "
                         "trades, explore the metric surface — no auto-picked "
                         "'best' config.", parent)
        self.state = RunState()

        self.tabs = QTabWidget()
        self.new_run = NewRunTab(settings, self.track_worker)
        self.explore = ExploreTab(settings, self.state)
        self.combine = CombineTab(settings, self.track_worker)
        self.tabs.addTab(self.new_run, "New Run")
        self.tabs.addTab(self.explore, "Explore")
        self.tabs.addTab(self.combine, "Combine")
        self.content.addWidget(self.tabs)

        self.new_run.runFinished.connect(self.adopt_unsaved_run)

    def adopt_unsaved_run(self, trades, meta, run_root) -> None:
        """A grid run just finished — hold it in memory and jump to Explore."""
        self.state.trades = trades
        self.state.meta = meta
        self.state.loaded_run = None
        self.state.unsaved = True
        self.state.run_root = run_root

        n_combos = meta.get("n_combos")
        self.explore.show_success(
            f"Done — {n_combos} backtests, {len(trades)} trades. Not saved "
            f"yet — use “Save this run” below to keep it.")
        self.explore.refresh_run_list(select_unsaved=True)
        self.tabs.setCurrentWidget(self.explore)
