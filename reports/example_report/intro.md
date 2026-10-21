## Why this report exists

Prose lives in `.md` files next to `report.py` and is rendered with
`st.markdown`. Inline math works like $\sigma_{\text{daily}}\sqrt{252}$, and
money is written escaped — \$1,250 — because an unescaped dollar sign starts
math.

The heatmap below is the Optimizer's Explore heatmap inside a report: same
metrics, same colours, same hatching for cells under the min-trades
threshold. Whatever you leave it showing is what the report shows next
time; lock it once it tells the story you want.
