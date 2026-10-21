# Reports

Research reports for the **Reports** module (main menu, card 08). This
README is not a report — it has no `report.py`.

## Folder conventions

- **One report = one folder** directly inside a report folder (this
  `reports/` folder, plus any extra folders added under Settings → Report
  folders). The folder name is the report's display name.
- The entry point is **`report.py`** — a Streamlit script. Everything else is
  yours to arrange by hand: `data/`, helper `.py` modules (importable from
  `report.py`), `.md` prose files, images, sub-folders.
- Folders starting with `_` or `.` are ignored (`_archive/` holds archived
  reports; `_drafts/` stays hidden).
- No import button: create the folder and copy data in by hand. The only
  things that ever write inside a report folder are the Compress tab (into
  the destination you pick) and the kit's saved views (`.report_state.json`).

## Writing

- Prose in `.md` files, rendered with `st.markdown(Path("intro.md").read_text())`.
- Math: inline `$…$`, display `$$…$$` (KaTeX). Number equations by hand with
  `\tag{n}`.
- **Write money as `\$5.76`** — an unescaped `$` starts math.
- Callouts: `st.info` / `st.warning` / `st.success` (LaTeX works inside).
- Layout is **wide** by default; call `st.set_page_config(layout="centered")`
  to override. The dark theme is applied at launch — no per-report config.

## The kit

```python
from modules.reports.kit import heatmap, style

style.chart(my_altair_chart)            # charts: the app's chart template

heatmap(key="es_grid",                  # heatmaps: Optimizer runs
        runs={"ES – fade/against": "data/optimizer_es_fade_against"},
        title="ES grid", default_metric="sharpe_daily")
```

- `runs` maps a label to a path **relative to the report folder**: a copied
  Optimizer run folder (`trades.parquet` + `meta.json`) or a compact run made
  by the Compress tab — both look identical.
- `key` is required and must be unique on the page: it is the component's
  identity for its saved view (`.report_state.json`), so reordering or
  inserting components never mixes views up.
- Every setting you change is saved; **Lock** freezes a finished component.
- All heatmap values are pre-cost, exactly as in the Optimizer.

Imports: the repo root is on the import path (appended — after the standard
library) whenever a report is launched from the app, so `modules.*` imports
work with no boilerplate.

See `example_report/` for all of the above in one place (its data is
synthetic).
