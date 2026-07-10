"""
Dark theme for the whole app: palette constants, the application-wide QSS,
and the pyqtgraph global configuration.

The accent color is carried over from the old Streamlit theme
(.streamlit/config.toml primaryColor = #273fc4).
"""

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# ── palette ───────────────────────────────────────────────────────────────────
ACCENT        = "#273fc4"
ACCENT_HOVER  = "#3450e0"
ACCENT_ACTIVE = "#1d329f"
BG            = "#0f1115"   # window background
SURFACE       = "#171a21"   # cards / inputs
SURFACE_2     = "#1e222b"   # hover / headers
BORDER        = "#2a2f3a"
TEXT          = "#e8eaf0"
TEXT_MUTED    = "#98a0b3"
GOOD          = "#2ca02c"
BAD           = "#d64545"
WARN          = "#d9a441"

CHART_BG = "#12151c"
CHART_FG = "#c8cdd8"

_QSS = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {TEXT_MUTED}; }}

QPushButton {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
}}
QPushButton:hover {{ background-color: #262b36; border-color: #3a4150; }}
QPushButton:pressed {{ background-color: {SURFACE}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; background-color: {SURFACE}; }}
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:pressed {{ background-color: {ACCENT_ACTIVE}; }}
QPushButton[primary="true"]:disabled {{
    background-color: {SURFACE_2}; border-color: {BORDER}; color: {TEXT_MUTED};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
QDateEdit:hover, QTimeEdit:hover {{ border-color: #3a4150; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QDateEdit:focus, QTimeEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 7px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    color: {TEXT_MUTED};
}}
QTabBar::tab:selected {{ background: {SURFACE_2}; color: {TEXT}; border-bottom-color: {SURFACE_2}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QTableView {{
    background-color: {SURFACE};
    alternate-background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: {BORDER};
}}
QHeaderView::section {{
    background-color: {SURFACE_2};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    color: {TEXT_MUTED};
}}
QTableCornerButton::section {{ background-color: {SURFACE_2}; border: none; }}

QProgressBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 14px;
    text-align: center;
    color: {TEXT};
    font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}

QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: Consolas, monospace;
    font-size: 12px;
}}

QScrollBar:vertical {{ background: {BG}; width: 11px; }}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #3a4150; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px;
}}

QListWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{ background: {ACCENT}; }}

QSlider::groove:horizontal {{
    height: 4px; background: {BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0;
    background: {ACCENT}; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion style + dark palette + QSS + pyqtgraph global config + the
    app-wide wheel guard (scrolling must scroll the page, not edit whatever
    combo/spinbox the cursor passes over)."""
    from .wheel_guard import install_wheel_guard
    install_wheel_guard(app)

    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(BG))
    pal.setColor(QPalette.WindowText,      QColor(TEXT))
    pal.setColor(QPalette.Base,            QColor(SURFACE))
    pal.setColor(QPalette.AlternateBase,   QColor(SURFACE_2))
    pal.setColor(QPalette.Text,            QColor(TEXT))
    pal.setColor(QPalette.Button,          QColor(SURFACE_2))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase,     QColor(SURFACE_2))
    pal.setColor(QPalette.ToolTipText,     QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(pal)

    app.setStyleSheet(_QSS)

    pg.setConfigOptions(antialias=True, background=CHART_BG, foreground=CHART_FG)
