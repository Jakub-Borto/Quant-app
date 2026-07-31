"""
Entry Breakdown — full performance per entry type (the trades' trade_type).

Deliberately NOT filtered by the trade-type filter: the whole point is to
compare entry types against each other, so filtering to one type first would
leave a single row. It IS filtered by day type and regime, so the table
answers "how does each entry behave in the slice I'm currently looking at".
The caption says so, because that asymmetry is otherwise invisible.

Hidden when the strategy emits no trade_type column.
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout

from modules.common.backend.trade_stats import ALL_ENTRIES, entry_breakdown_rows
from ..dataframe_model import make_table_view, update_table_view
from ..widgets import Caption
from .sections import ReportSection


class EntryBreakdownSection(ReportSection):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._caption = Caption(
            "Every entry type under the current day-type and regime filters — "
            "the trade-type filter is deliberately NOT applied here, so the "
            "types stay comparable. “All entries” is the whole-strategy "
            "benchmark and stays pinned to the top when you sort. Both Sharpes "
            "are annualized ×√252.")
        lay.addWidget(self._caption)
        self._table = make_table_view(pd.DataFrame(), height=230,
                                      pinned_labels={ALL_ENTRIES})
        lay.addWidget(self._table)

    def set_trades(self, trades: pd.DataFrame) -> bool:
        """True when the table has rows (trades carry a trade_type column)."""
        rows = entry_breakdown_rows(trades)
        if rows is None:
            update_table_view(self._table, pd.DataFrame())
            return False
        update_table_view(self._table, pd.DataFrame(rows))
        return True
