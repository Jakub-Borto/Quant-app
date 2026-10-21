"""
The app's colour palette and font stack — Qt-free, so both the PySide6 theme
(modules/common/ui/theme.py) and Qt-free consumers (the Reports module's
Streamlit launch theme and its report kit) read the SAME values.
"""

ACCENT        = "#273fc4"
ACCENT_HOVER  = "#3450e0"
ACCENT_ACTIVE = "#1d329f"
ACCENT_SOFT   = "#8fa2ff"   # accent-tinted text (card numbers, links)
BG            = "#10131A"   # window background (page behind the cards)
SURFACE       = "#151924"   # cards / inputs
SURFACE_2     = "#1c2130"   # hover / headers
SURFACE_3     = "#262c3d"   # pressed / stronger hover
BORDER        = "#262c39"
BORDER_LIGHT  = "#3a4358"   # hover borders, checkbox boxes
TEXT          = "#e8eaf0"
TEXT_MUTED    = "#96a0b5"
GOOD          = "#2ca02c"
BAD           = "#d64545"
WARN          = "#d9a441"

CHART_BG = "#11141c"
CHART_FG = "#c8cdd8"

FONT_FAMILY = '"Segoe UI", "Inter", sans-serif'
MONO_FAMILY = "Consolas, monospace"
