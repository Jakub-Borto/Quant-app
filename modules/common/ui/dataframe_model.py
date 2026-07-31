"""
Read-only Qt table model over a pandas DataFrame (st.dataframe analog).

Values are formatted lazily per cell in data() — never pre-stringify a whole
frame (optimizer trades tables can be 100k+ rows; QTableView virtualizes).

Click a header to sort. Sorting is NUMERIC-AWARE: the breakdown tables hold
display strings ("62.5%", "1.41", "∞", "N/A"), and sorting those as text puts
100% before 62.5% and ∞ next to nothing useful. _sort_key parses them back to
numbers, falls back to case-insensitive text for genuinely textual columns,
and always parks blanks last.
"""

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

_BLANKS = {"", "n/a", "—", "-", "nan", "none"}


def _sort_key(series: pd.Series):
    """(values, is_numeric) for one column, ready to argsort."""
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float), True

    parsed = []
    for value in series:
        if isinstance(value, bool):
            parsed.append(float(value))
            continue
        if isinstance(value, (int, float)):
            parsed.append(float(value))
            continue
        text = str(value).strip()
        if text.lower() in _BLANKS:
            parsed.append(float("nan"))
            continue
        if text in ("∞", "inf"):
            parsed.append(float("inf"))
            continue
        if text in ("-∞", "-inf"):
            parsed.append(float("-inf"))
            continue
        body = text[:-1] if text.endswith("%") else text
        try:
            parsed.append(float(body.replace(",", "")))
        except ValueError:
            return series.astype(str).str.lower().to_numpy(), False
    return np.asarray(parsed, dtype=float), True


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
    def __init__(self, df: pd.DataFrame, parent=None, pinned_labels=None):
        super().__init__(parent)
        self._df = df
        # first-column values that always stay at the top, whatever the sort —
        # e.g. the entry breakdown's "All entries" benchmark row, which is only
        # useful if it stays put to read the other rows against
        self._pinned = set(pinned_labels or ())

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def _pinned_mask(self) -> np.ndarray:
        if not self._pinned or self._df.empty:
            return np.zeros(len(self._df), dtype=bool)
        first = self._df.iloc[:, 0].astype(str)
        return first.isin(self._pinned).to_numpy()

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

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        """Reorder the underlying rows. Blanks always sort last, in both
        directions — a column of mostly N/A should not bury the real values."""
        if not (0 <= column < len(self._df.columns)) or self._df.empty:
            return
        values, numeric = _sort_key(self._df.iloc[:, column])
        positions = np.argsort(values, kind="stable")
        if order != Qt.AscendingOrder:
            positions = positions[::-1]
        if numeric:                      # argsort trails NaN; keep it trailing
            blank = np.isnan(values[positions])
            positions = np.concatenate([positions[~blank], positions[blank]])

        pinned = self._pinned_mask()[positions]
        positions = np.concatenate([positions[pinned], positions[~pinned]])

        self.beginResetModel()
        self._df = self._df.iloc[positions]
        self.endResetModel()


def make_table_view(df: pd.DataFrame, height: int | None = None,
                    hide_index: bool = True, sortable: bool = True,
                    pinned_labels=None) -> QTableView:
    """A configured read-only QTableView over `df`. Headers are clickable to
    sort unless `sortable=False`; `pinned_labels` are first-column values that
    stay at the top through any sort (a benchmark row)."""
    view = QTableView()
    view.setModel(DataFrameModel(df, pinned_labels=pinned_labels))
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QAbstractItemView.NoEditTriggers)
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.verticalHeader().setVisible(not hide_index)
    view.verticalHeader().setDefaultSectionSize(26)
    header = view.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    # NOT stretchLastSection: it inflates the final column to fill the width,
    # so a narrow "Avg Ticks" ends up hundreds of pixels wide with its value
    # marooned on the left. Columns hug their contents; leftover width stays
    # empty, which is what these summary tables should look like.
    header.setStretchLastSection(False)
    if sortable:
        # Clear the indicator FIRST. setSortingEnabled(True) immediately sorts
        # by the current indicator section (0 by default), which would scramble
        # row orders that carry meaning — DAY_TYPE_ORDER, declared regime
        # states with 'unknown' last, "All entries" last. Section -1 means "no
        # sort", and model.sort() ignores it.
        header.setSortIndicator(-1, Qt.AscendingOrder)
        view.setSortingEnabled(True)
        header.setSortIndicatorShown(True)
    if height is not None:
        view.setFixedHeight(height)
    return view


def update_table_view(view: QTableView, df: pd.DataFrame) -> None:
    """Swap in a new frame, keeping whatever sort the user picked — a filter
    change should not silently throw their chosen ordering away."""
    model = view.model()
    if isinstance(model, DataFrameModel):
        model.set_frame(df)
    else:
        view.setModel(DataFrameModel(df))
        model = view.model()

    header = view.horizontalHeader()
    column = header.sortIndicatorSection()
    if view.isSortingEnabled() and 0 <= column < len(df.columns):
        model.sort(column, header.sortIndicatorOrder())
