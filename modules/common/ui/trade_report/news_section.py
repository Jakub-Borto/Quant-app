"""
News & Holiday Exposure table — computed from the trade_type-filtered trades
but unaffected by the day_type filter (the calling window controls placement,
preserving the old filter ordering).
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout, QWidget

from modules.common.backend.trade_stats import news_holiday_rows
from ..dataframe_model import make_table_view, update_table_view
from ..widgets import SectionHeader


class NewsBreakdownTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(SectionHeader("News && Holiday Exposure"))
        self._table = make_table_view(pd.DataFrame(), height=278)
        lay.addWidget(self._table)
        self.setVisible(False)

    def set_trades(self, trades: pd.DataFrame) -> None:
        rows = news_holiday_rows(trades)
        if rows is None:
            self.setVisible(False)
            return
        update_table_view(self._table, pd.DataFrame(rows))
        self.setVisible(True)
