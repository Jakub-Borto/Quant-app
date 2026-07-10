"""
Read-only Qt table model over a pandas DataFrame (st.dataframe analog).

Values are formatted lazily per cell in data() — never pre-stringify a whole
frame (optimizer trades tables can be 100k+ rows; QTableView virtualizes).
"""

import math

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "∞"
        return f"{value:g}"
    if isinstance(value, pd.Timestamp):
        # drop a pure-midnight time part for date-like stamps
        if value.tz is None and value == value.normalize():
            return value.date().isoformat()
        return str(value)
    return str(value)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    # ── QAbstractTableModel ───────────────────────────────────────────────────
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return _fmt(self._df.iat[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section)


def make_table_view(df: pd.DataFrame, height: int | None = None,
                    hide_index: bool = True) -> QTableView:
    """A configured read-only QTableView over `df`."""
    view = QTableView()
    view.setModel(DataFrameModel(df))
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QAbstractItemView.NoEditTriggers)
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.verticalHeader().setVisible(not hide_index)
    view.verticalHeader().setDefaultSectionSize(26)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(True)
    if height is not None:
        view.setFixedHeight(height)
    return view


def update_table_view(view: QTableView, df: pd.DataFrame) -> None:
    model = view.model()
    if isinstance(model, DataFrameModel):
        model.set_frame(df)
    else:
        view.setModel(DataFrameModel(df))
