"""
"Additional data" picker — one row per additional DATA slot a strategy
declares (every DATA key except "main"; see modules.engine / STRATEGY_GUIDE.md).

Each row is the same Type -> Asset -> Dataset cascade as the main picker. Type
and asset follow the main selection; the dataset is auto-picked by name
(data_roots.default_dataset_for_slot: the first folder whose name contains the
slot name, e.g. "indicators" -> ES_1m_indicators). Any row can be changed by
hand. A row with no dataset is an error: ok() is False and problems() says
why, so the window can block Run.

Used by the Backtester and the Optimizer's New Run tab.
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QVBoxLayout, QWidget

from modules.common.backend.data_roots import DatasetRef, default_dataset_for_slot

from . import theme
from .widgets import Caption, SectionHeader


class _SlotRow:
    def __init__(self, slot: str, grid: QGridLayout, row: int, owner):
        self.slot = slot
        self._owner = owner
        self.type = QComboBox()
        self.asset = QComboBox()
        self.dataset = QComboBox()
        self.error = QLabel("")
        self.error.setStyleSheet(f"color: {theme.BAD}; font-size: 12px;")
        name = QLabel(slot)
        name.setStyleSheet("font-weight: 600;")
        grid.addWidget(name, row, 0)
        grid.addWidget(QLabel("Type"), row, 1)
        grid.addWidget(self.type, row, 2)
        grid.addWidget(QLabel("Asset"), row, 3)
        grid.addWidget(self.asset, row, 4)
        grid.addWidget(QLabel("Dataset"), row, 5)
        grid.addWidget(self.dataset, row, 6)
        grid.addWidget(self.error, row, 7)
        self.type.currentIndexChanged.connect(self._on_type)
        self.asset.currentIndexChanged.connect(self._on_asset)
        self.dataset.currentIndexChanged.connect(self._on_dataset)

    # ── cascade ──────────────────────────────────────────────────────────────
    def fill(self, structure: dict, asset_type: str | None, asset: str | None,
             prefer_root) -> None:
        """Reset the row to follow the main selection."""
        self._structure = structure
        self._prefer_root = prefer_root
        self.type.blockSignals(True)
        self.type.clear()
        self.type.addItems(list(structure))
        i = self.type.findText(asset_type or "")
        self.type.setCurrentIndex(max(i, 0))
        self.type.blockSignals(False)
        self._fill_assets(asset)

    def _fill_assets(self, want_asset: str | None) -> None:
        assets = list(self._structure.get(self.type.currentText(), {}))
        self.asset.blockSignals(True)
        self.asset.clear()
        self.asset.addItems(assets)
        i = self.asset.findText(want_asset or "")
        self.asset.setCurrentIndex(max(i, 0))
        self.asset.blockSignals(False)
        self._fill_datasets()

    def _fill_datasets(self) -> None:
        refs = self._structure.get(self.type.currentText(), {}) \
                              .get(self.asset.currentText(), [])
        pick = default_dataset_for_slot(refs, self.slot, self._prefer_root)
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for ref in refs:
            self.dataset.addItem(ref.label, ref)
        if pick is not None:
            self.dataset.setCurrentIndex(refs.index(pick))
        else:
            self.dataset.setCurrentIndex(-1)
        self.dataset.blockSignals(False)
        self._refresh_error()

    def _on_type(self) -> None:
        self._fill_assets(self.asset.currentText())
        self._owner._emit()

    def _on_asset(self) -> None:
        self._fill_datasets()
        self._owner._emit()

    def _on_dataset(self) -> None:
        self._refresh_error()
        self._owner._emit()

    # ── state ────────────────────────────────────────────────────────────────
    def ref(self) -> DatasetRef | None:
        return self.dataset.currentData()

    def problem(self) -> str | None:
        if self.ref() is not None:
            return None
        return (f"No dataset for '{self.slot}' under "
                f"{self.type.currentText() or '?'}/{self.asset.currentText() or '?'} "
                f"— no folder name contains '{self.slot}'. Pick one by hand or "
                f"create it with the Data Formatter.")

    def _refresh_error(self) -> None:
        self.error.setText("" if self.ref() is not None else "⚠ no matching dataset")


class AdditionalDataPanel(QWidget):
    """Rows for a strategy's additional data slots. Hidden when it has none."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(SectionHeader("Additional data"))
        lay.addWidget(Caption(
            "Extra datasets this strategy reads next to the main one (its DATA "
            "slots). Each is auto-picked from the same type/asset by name — "
            "change any row if needed. A day missing a file in any of them is "
            "skipped, with a warning after the run."))
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(6)
        lay.addWidget(self._grid_host)
        self._rows: list[_SlotRow] = []
        self._structure: dict = {}
        self._main = (None, None, None)
        self.setVisible(False)

    # ── setup ────────────────────────────────────────────────────────────────
    def set_slots(self, slots: list[str]) -> None:
        """Rebuild one row per slot (call on strategy change)."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = [_SlotRow(slot, self._grid, r, self) for r, slot in enumerate(slots)]
        self.setVisible(bool(slots))
        self._refill()

    def set_structure(self, structure: dict) -> None:
        self._structure = structure
        self._refill()

    def follow_main(self, asset_type: str | None, asset: str | None, root) -> None:
        """Reset every row to the main dataset's type/asset (call when the main
        selection changes)."""
        self._main = (asset_type, asset, root)
        self._refill()

    def _refill(self) -> None:
        asset_type, asset, root = self._main
        for row in self._rows:
            row.fill(self._structure, asset_type, asset, root)
        self._emit()

    def _emit(self) -> None:
        self.changed.emit()

    # ── results ──────────────────────────────────────────────────────────────
    def slots(self) -> list[str]:
        return [r.slot for r in self._rows]

    def folders(self) -> dict[str, Path]:
        """{slot: dataset folder} for every row that has a dataset."""
        return {r.slot: r.ref().path for r in self._rows if r.ref() is not None}

    def descriptions(self) -> dict[str, str]:
        """{slot: "type/asset/dataset"} — what the Optimizer records in meta.json."""
        return {r.slot: f"{r.ref().asset_type}/{r.ref().asset}/{r.ref().dataset}"
                for r in self._rows if r.ref() is not None}

    def problems(self) -> list[str]:
        return [p for p in (r.problem() for r in self._rows) if p]

    def ok(self) -> bool:
        return not self.problems()
