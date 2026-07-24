"""
Regime Detector window — QTabWidget with the two sub-modules: New Run /
Explore (the Optimizer window shape). All computation runs on FunctionWorker
threads via ModuleWindowBase.track_worker; the window never freezes.
"""

from PySide6.QtWidgets import QTabWidget

from modules.common.ui.module_window import ModuleWindowBase
from modules.regime_detector.explore_tab import ExploreTab
from modules.regime_detector.run_tab import RunTab


class RegimeDetectorWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Regime Detector",
                         "Turn candle datasets into per-snapshot regime label "
                         "files — each row built only from bars available at "
                         "that moment.", parent)
        self.tabs = QTabWidget()
        self.run_tab = RunTab(settings, self.track_worker)
        self.explore_tab = ExploreTab(settings, self.track_worker)
        self.tabs.addTab(self.run_tab, "New Run")
        self.tabs.addTab(self.explore_tab, "Explore")
        self.content.addWidget(self.tabs)
