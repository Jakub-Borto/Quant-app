"""Example report — shows every convention in reports/README.md.

The heatmap data in data/ is SYNTHETIC (a made-up demo run, not market
results): data/es_demo_full is a full Optimizer-run copy, data/compact/
es_demo_full its Compress-tab output — the component shows both identically.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from modules.reports.kit import heatmap, style

HERE = Path(__file__).parent

st.title("Example report — breakout stop/target grid")
st.caption("Synthetic demo data · every number below is made up")

st.markdown((HERE / "intro.md").read_text(encoding="utf-8"))

st.info(r"A strategy with win rate $p$ and reward-to-risk $R$ breaks even when "
        r"$p \cdot R = 1 - p$, i.e. $p^* = \frac{1}{1+R}$. At \$5.76 commission "
        r"per round turn the bar moves up — costs are not in these heatmaps "
        r"(they are pre-cost, exactly like the Optimizer).")

st.markdown(r"""
### The grid

$$
\text{expectancy} = p \cdot R \cdot s - (1-p)\cdot s \tag{1}
$$

where $s$ is the stop in ticks. Equation (1) is linear in $s$, so a wider stop
only helps when it raises $p$.
""")

heatmap(key="es_grid",
        runs={"ES – demo (full run)": "data/es_demo_full",
              "ES – demo (compact)": "data/compact/es_demo_full"},
        title="ES demo grid", default_metric="sharpe_daily")

st.markdown("### A plain chart with the report template")
x = np.linspace(0.5, 3.0, 26)
df = pd.DataFrame({"R": x, "break-even win rate": 1 / (1 + x)})
style.chart(alt.Chart(df).mark_line(strokeWidth=2.5).encode(
    x=alt.X("R:Q", title="reward-to-risk R"),
    y=alt.Y("break-even win rate:Q", axis=alt.Axis(format="%"))))

st.success("Conclusion: written here, in Markdown, with $\\LaTeX$ where it helps.")
