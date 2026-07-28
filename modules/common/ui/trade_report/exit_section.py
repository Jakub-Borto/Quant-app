"""
Exit Breakdown table — count / avg ticks / total ticks per exit_reason.

Split out of MetricsSection so it can be ordered and collapsed on its own.
Standalone it must tolerate trades without an exit_reason column (fused into
the metrics section it never had to), so it degrades to an empty table rather
than raising.
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout

from modules.common.backend.trade_stats import exit_breakdown_table
from ..dataframe_model import make_table_view, update_table_view
from .sections import ReportSection


class ExitBreakdownSection(ReportSection):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = make_table_view(pd.DataFrame(), height=190)
        lay.addWidget(self.table)

    def set_trades(self, trades: pd.DataFrame) -> None:
        if "exit_reason" not in trades.columns:
            update_table_view(self.table, pd.DataFrame())
            return
        update_table_view(self.table, exit_breakdown_table(trades))
