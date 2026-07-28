"""
News & Holiday Exposure table — computed from the trade_type-filtered trades
but unaffected by the day_type filter (the calling window controls placement,
preserving the old filter ordering).
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout

from modules.common.backend.trade_stats import news_holiday_rows
from ..dataframe_model import make_table_view, update_table_view
from .sections import ReportSection


class NewsBreakdownTable(ReportSection):
    """The section header now comes from the stack; set_trades() reports
    whether it has anything to show so the host can hide the whole frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._table = make_table_view(pd.DataFrame(), height=278)
        lay.addWidget(self._table)

    def set_trades(self, trades: pd.DataFrame) -> bool:
        """True when the table has rows (trades carry a day_type column)."""
        rows = news_holiday_rows(trades)
        if rows is None:
            update_table_view(self._table, pd.DataFrame())
            return False
        update_table_view(self._table, pd.DataFrame(rows))
        return True
