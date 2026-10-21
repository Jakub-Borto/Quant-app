"""
Report styling: the app's palette and fonts (the same values the launcher
passes to every report as its Streamlit theme) and ONE Altair chart template.

    from modules.reports.kit import style
    style.chart(alt.Chart(df).mark_line().encode(x="date:T", y="pnl:Q"))

`style.chart` renders with the template (Streamlit's own chart theme would
otherwise override it). `HEATMAP_COLORSCALE` / `heatmap_rgb` are the app's
heatmap colour scale, for any chart that wants it.
"""

import altair as alt

from modules.common.backend import palette
from modules.optimizer.backend.heatmap_model import (HEATMAP_COLORSCALE,
                                                     _colorscale_rgb)

COLORS = {
    "accent": palette.ACCENT,
    "accent_hover": palette.ACCENT_HOVER,
    "accent_soft": palette.ACCENT_SOFT,
    "bg": palette.BG,
    "surface": palette.SURFACE,
    "surface_2": palette.SURFACE_2,
    "surface_3": palette.SURFACE_3,
    "border": palette.BORDER,
    "border_light": palette.BORDER_LIGHT,
    "text": palette.TEXT,
    "text_muted": palette.TEXT_MUTED,
    "good": palette.GOOD,
    "bad": palette.BAD,
    "warn": palette.WARN,
    "chart_bg": palette.CHART_BG,
    "chart_fg": palette.CHART_FG,
}
FONT = palette.FONT_FAMILY.replace('"', "")
MONO_FONT = palette.MONO_FAMILY

# series colours: accent first, then the heatmap scale's warm end + neutrals
CATEGORY_COLORS = [palette.ACCENT_HOVER, "#e8871e", "#2ca02c", "#c0392b",
                   palette.ACCENT_SOFT, "#f4d03f", "#96a0b5", "#2b6ca9"]

THEME_NAME = "quant_app"

__all__ = ["COLORS", "FONT", "MONO_FONT", "CATEGORY_COLORS",
           "HEATMAP_COLORSCALE", "heatmap_rgb", "hex_color", "chart_config",
           "chart", "THEME_NAME"]


def heatmap_rgb(position: float) -> tuple:
    """RGB of the app's heatmap colour scale at position 0..1."""
    return _colorscale_rgb(position)


def hex_color(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def chart_config() -> dict:
    """The Vega-Lite config of the report chart template."""
    axis = {
        "domainColor": palette.BORDER_LIGHT,
        "gridColor": palette.BORDER,
        "tickColor": palette.BORDER_LIGHT,
        "labelColor": palette.CHART_FG,
        "titleColor": palette.CHART_FG,
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": 12,
        "titleFontSize": 12,
        "titleFontWeight": 600,
    }
    return {
        "background": palette.CHART_BG,
        "padding": 10,
        "font": FONT,
        "view": {"stroke": palette.BORDER, "fill": palette.CHART_BG},
        "axis": axis,
        "legend": {"labelColor": palette.CHART_FG, "titleColor": palette.CHART_FG,
                   "labelFont": FONT, "titleFont": FONT},
        "title": {"color": palette.TEXT, "font": FONT, "fontSize": 14,
                  "fontWeight": 600, "anchor": "start"},
        "header": {"labelColor": palette.CHART_FG, "titleColor": palette.CHART_FG},
        "range": {"category": CATEGORY_COLORS,
                  "heatmap": [c for _p, c in HEATMAP_COLORSCALE]},
        "mark": {"color": palette.ACCENT_HOVER},
        "text": {"color": palette.TEXT, "font": FONT},
    }


@alt.theme.register(THEME_NAME, enable=True)
def _theme():
    return alt.theme.ThemeConfig({"config": chart_config()})


def chart(altair_chart, *, width="stretch", **kwargs):
    """Render an Altair chart with the report template (not Streamlit's)."""
    import streamlit as st
    alt.theme.enable(THEME_NAME)
    return st.altair_chart(altair_chart, theme=None, width=width, **kwargs)
