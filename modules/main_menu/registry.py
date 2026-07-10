"""
The module registry — one ModuleSpec per launchable module.

Card numbers/titles/blurbs carry over from the old home page
(legacy_streamlit/views/home.py). window_factory(settings) must return a NEW
top-level QWidget every call — the menu never reuses instances, which is what
makes multiple independent windows of the same module possible. Factories
import lazily so the menu opens fast and an error in one module doesn't take
the launcher down.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    number: str
    title: str
    blurb: str

    def create_window(self, settings):
        """Instantiate a fresh module window (lazy import by key)."""
        if self.key == "data_formatter":
            from modules.data_formatter.window import DataFormatterWindow
            return DataFormatterWindow(settings)
        if self.key == "backtester":
            from modules.backtester.window import BacktesterWindow
            return BacktesterWindow(settings)
        if self.key == "analytics":
            from modules.analytics.window import AnalyticsWindow
            return AnalyticsWindow(settings)
        if self.key == "monte_carlo":
            from modules.monte_carlo.window import MonteCarloWindow
            return MonteCarloWindow(settings)
        if self.key == "optimizer":
            from modules.optimizer.window import OptimizerWindow
            return OptimizerWindow(settings)
        raise KeyError(f"Unknown module key: {self.key}")


MODULES = [
    ModuleSpec("data_formatter", "01", "Data Formatter",
               "Convert raw DBN files into enriched 1m candles stored as Parquet."),
    ModuleSpec("backtester", "02", "Backtester",
               "Run vectorized strategies on your datasets. Outputs trades to Parquet."),
    ModuleSpec("analytics", "03", "Analytics",
               "Load trades, apply position sizing, explore equity curve and metrics."),
    ModuleSpec("monte_carlo", "04", "Monte Carlo",
               "Run Monte Carlo simulations to stress test the strategy."),
    ModuleSpec("optimizer", "05", "Optimizer",
               "Sweep strategy params on a grid and explore the metric heatmap."),
]
