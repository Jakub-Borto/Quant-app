"""
The reports kit — what report scripts import (Qt-free; grows as reports
need things).

    from modules.reports.kit import heatmap, style

    style.chart(my_altair_chart)                 # the app's chart template
    heatmap(key="es_grid", runs={"ES – fade": "data/optimizer_es_fade"})

style    the app palette/fonts, the heatmap colour scale, one Altair chart
         template (`style.chart` renders with it)
heatmap  an Optimizer run (full or compact) shown exactly as the Optimizer's
         Explore heatmap would — same metrics, same colour logic, same
         hatching rule
state    saved views + lock (`.report_state.json` in the report folder),
         shared by every kit component; each component takes a required,
         page-unique `key`
"""

from . import style
from .heatmap import heatmap

__all__ = ["heatmap", "style"]
