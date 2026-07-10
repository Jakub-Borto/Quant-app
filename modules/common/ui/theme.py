"""
Dark theme for the whole app — THE one place that controls how everything
looks. `_QSS` below is Qt Style Sheets (Qt's CSS dialect): edit it and every
window restyles on next launch.

Notes for future edits:
- The accent color carries over from the old Streamlit theme (#273fc4).
- Type selectors match subclasses too (QAbstractSpinBox rules also hit
  QSpinBox / QDoubleSpinBox / QDateEdit / QTimeEdit).
- IMPORTANT: once you touch a sub-control in QSS (::indicator, ::up-button,
  ::down-arrow …) Qt stops native-painting it — you must style it FULLY or
  it renders invisible. That is exactly what happened to the optimizer's
  sweep checkboxes before the indicator states below existed.
- Icons (checkmark, chevrons) are SVGs in modules/common/ui/assets/,
  referenced with absolute paths resolved at import time.
"""

from pathlib import Path

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
SURFACE_3     = "#262b36"   # pressed / stronger hover
BORDER        = "#2a2f3a"
BORDER_LIGHT  = "#3a4150"   # hover borders, checkbox boxes
TEXT          = "#e8eaf0"
TEXT_MUTED    = "#98a0b3"
GOOD          = "#2ca02c"
BAD           = "#d64545"
WARN          = "#d9a441"

CHART_BG = "#12151c"
CHART_FG = "#c8cdd8"

_ASSETS  = Path(__file__).resolve().parent / "assets"
_CHECK   = (_ASSETS / "check.svg").as_posix()
_CHEV_DN = (_ASSETS / "chevron-down.svg").as_posix()
_CHEV_UP = (_ASSETS / "chevron-up.svg").as_posix()

_QSS = f"""
/* ── base ─────────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {TEXT_MUTED}; }}

/* ── buttons ──────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
}}
QPushButton:hover {{ background-color: {SURFACE_3}; border-color: {BORDER_LIGHT}; }}
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
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{ background-color: {SURFACE_2}; border-color: {BORDER}; }}

/* ── text/number inputs ───────────────────────────────────────────────────── */
QComboBox, QLineEdit, QAbstractSpinBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QComboBox:hover, QLineEdit:hover, QAbstractSpinBox:hover {{
    border-color: {BORDER_LIGHT};
}}
QComboBox:focus, QLineEdit:focus, QAbstractSpinBox:focus {{
    border-color: {ACCENT_HOVER};
}}
QComboBox:disabled, QLineEdit:disabled, QAbstractSpinBox:disabled {{
    color: {TEXT_MUTED}; background-color: #13151b;
}}

/* combo drop-down */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
}}
QComboBox::down-arrow {{
    image: url("{_CHEV_DN}");
    width: 12px; height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 6px;
    padding: 4px;
    selection-background-color: {ACCENT};
    selection-color: white;
    outline: none;
}}

/* spin-box steppers (also QDateEdit / QTimeEdit) */
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    border: none;
    border-left: 1px solid {BORDER};
    background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 6px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 6px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {SURFACE_3};
}}
QAbstractSpinBox::up-arrow {{ image: url("{_CHEV_UP}"); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url("{_CHEV_DN}"); width: 10px; height: 10px; }}

/* ── check boxes & radios ─────────────────────────────────────────────────── */
/* Styling ::indicator disables native painting — every state must be drawn
   here or the box turns invisible (the old optimizer sweep-checkbox bug). */
QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {TEXT_MUTED}; }}

QCheckBox::indicator, QGroupBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER_LIGHT};
    background: {SURFACE};
}}
QCheckBox::indicator:hover, QGroupBox::indicator:hover {{
    border-color: {ACCENT_HOVER};
}}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{_CHECK}");
}}
QCheckBox::indicator:checked:hover, QGroupBox::indicator:checked:hover {{
    background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER};
}}
QCheckBox::indicator:disabled {{ background: #13151b; border-color: {BORDER}; }}

QRadioButton::indicator {{
    width: 16px; height: 16px;
    border-radius: 8px;
    border: 1px solid {BORDER_LIGHT};
    background: {SURFACE};
}}
QRadioButton::indicator:hover {{ border-color: {ACCENT_HOVER}; }}
QRadioButton::indicator:checked {{
    border-color: {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 #ffffff, stop:0.38 #ffffff,
                                stop:0.5 {ACCENT}, stop:1 {ACCENT});
}}
QRadioButton::indicator:disabled {{ background: #13151b; border-color: {BORDER}; }}

/* ── group boxes ──────────────────────────────────────────────────────────── */
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

/* ── tabs (accent underline) ──────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 20px;
    margin-right: 4px;
    color: {TEXT_MUTED};
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT_HOVER};
    font-weight: 600;
}}

/* ── tables ───────────────────────────────────────────────────────────────── */
QTableView {{
    background-color: {SURFACE};
    alternate-background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT};
    selection-color: white;
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

/* ── progress bar ─────────────────────────────────────────────────────────── */
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

/* ── consoles ─────────────────────────────────────────────────────────────── */
QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: Consolas, monospace;
    font-size: 12px;
    selection-background-color: {ACCENT};
}}

/* ── scrollbars ───────────────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: {BG}; width: 11px; }}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {BORDER_LIGHT}; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {BORDER_LIGHT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── tooltips ─────────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    padding: 6px;
}}

/* ── lists ────────────────────────────────────────────────────────────────── */
QListWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; }}

/* ── sliders ──────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px; background: {BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0;
    background: {ACCENT}; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
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
