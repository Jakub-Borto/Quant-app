"""
DeleteConfirmDialog — the typed-code guard in front of every permanent
delete in the Reports module (report folders and archived zips).

Every opening shows a FRESH random 6-character code (uppercase letters +
digits without look-alikes, backend.archive.generate_delete_code); the
Delete button stays disabled until the field matches it exactly
(case-sensitive). Cancel / Escape abort.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout)

from modules.common.ui import theme
from .backend.archive import generate_delete_code


class DeleteConfirmDialog(QDialog):
    def __init__(self, what: str, target: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete permanently")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.code = generate_delete_code()

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        head = QLabel(f"Delete {what}?")
        head.setStyleSheet("font-size: 15px; font-weight: 600;")
        lay.addWidget(head)
        tgt = QLabel(target)
        tgt.setWordWrap(True)
        tgt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tgt.setStyleSheet(f"color: {theme.ACCENT_SOFT};")
        lay.addWidget(tgt)
        warn = QLabel("This is permanent — it does not go to the recycle bin "
                      "and cannot be undone.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #ff9d9d;")
        lay.addWidget(warn)

        lay.addWidget(QLabel("Type this code to confirm:"))
        code = QLabel(self.code)
        code.setObjectName("deleteCode")
        code.setAlignment(Qt.AlignCenter)
        code.setTextInteractionFlags(Qt.NoTextInteraction)   # type it, don't paste it
        code.setStyleSheet(
            f"font-family: Consolas, monospace; font-size: 22px; font-weight: 700; "
            f"letter-spacing: 6px; padding: 8px; background: {theme.SURFACE_2}; "
            f"border: 1px solid {theme.BORDER_LIGHT}; border-radius: 8px;")
        lay.addWidget(code)

        self.field = QLineEdit()
        self.field.setPlaceholderText("code")
        self.field.setMaxLength(len(self.code))
        self.field.textChanged.connect(self._sync)
        lay.addWidget(self.field)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet(
            "QPushButton:enabled { color: white; background: #a33232; "
            "border-color: #c94444; font-weight: 600; }")
        self.delete_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.delete_btn)
        lay.addLayout(buttons)
        for b in (cancel, self.delete_btn):      # Enter never triggers either
            b.setAutoDefault(False)
            b.setDefault(False)

    def matches(self) -> bool:
        return self.field.text() == self.code

    def _sync(self) -> None:
        self.delete_btn.setEnabled(self.matches())

    def accept(self) -> None:
        if self.matches():           # the button is the only path, but be strict
            super().accept()
