# CLAUDE.md — Quant Research Platform

Orientation for Claude Code working in this repo. A companion PDF
(`Quant_app_documentation.pdf`) holds the long-form version of the ORIGINAL
Streamlit-era design (module contracts and data schemas in it still apply;
UI/file-layout sections are outdated since the PySide6 rebuild).

## What this is

A modular **intraday futures research platform**, now a native **PySide6
desktop app** (rebuilt from Streamlit in July 2026, zero logic changes) for
30+ instruments. It turns raw Databento market data into enriched candle
datasets, runs vectorized backtests, applies position sizing, runs
Monte-Carlo stress tests, and sweeps strategy parameter grids.

Run it:

```bash
python main.py
```

`main.py` is spawn-safe (nothing outside its `__main__` guard — optimizer
pool workers re-import it). It boots `modules/app.py`: dark theme, settings,
and the **main menu** — a card launcher where every module opens in its own
window. The same module can be opened multiple times; instances are fully
independent, and long-running work runs on worker threads (windows never
freeze; the menu's gear opens the folder settings).

## Plugin-drop architecture (the core idea)

Plugin folders are scanned dynamically (`importlib.util.spec_from_file_location`
→ `exec_module`) — **drop a file in the folder and it appears in the UI**.
No registration. The filename stem becomes the UI display name;
`__init__.py` / `base.py` are excluded. Strategies may also be **packages**
(folder with `__init__.py` exposing `run`, `PARAMS`, `PARAM_SECTIONS`) — see
`strategies/ivb_model/` and its own `CLAUDE.md`.

Any plugin may also declare `PARAMS_OPTIONS = {param: [choice, ...]}`:
default-in-list → dropdown; '0'/'1' bitstring default with one char per
option → named checkbox group returning the bitstring; bool defaults render
as checkboxes. The full declaration contract (incl. how each type sweeps in
the Optimizer) is the docstring of `modules/common/ui/params_form.py`.

| Folder | Used by | Contract |
|---|---|---|
| `data_transforms/` | Data Formatter | `run_all(input_folder, output_folder, skip_existing, on_progress[, params]) -> None` (a transform MAY declare `PARAMS` like a strategy — the UI renders widgets from it and passes the values as `params`; transforms without `PARAMS` get the plain 4-arg call) |
| `strategies/` | Backtester, Optimizer | `run(folder_path, start_date, end_date, params) -> pd.DataFrame` (+ `PARAMS`, optional `PARAMS_OPTIONS`; the Optimizer sweeps int/float params over min/max/step, str params over a value list, bool params over [False, True], dropdown params over a subset of their choices, bit-flag params over a bitstring list) |
| `position_sizing/` | Analytics, Monte Carlo | `apply(trades, params) -> pd.DataFrame` (+ `PARAMS`) |
| `modules/monte_carlo/methods/` | Monte Carlo | `run(trades, sizer_module, sizer_params, params) -> dict` (+ `PARAMS`; `PROP_FIRM = True` opts into the dedicated prop-firm UI) |
| `scripts/` | Scripts | no Python contract — any quick one-off script. A `# app: streamlit` comment (or `STREAMLIT = True`) in the first 30 lines → launched as `streamlit run` on a free port + opened in the dedicated scripts browser (Chrome/Edge with a private `--user-data-dir` profile — first run opens its window, later runs add tabs there; Opera ignores the flags, so it's never a candidate); otherwise run as `python -u` with output in the module's console. cwd is the script's own folder (NOT repo root — root `inspect.py` shadowing); repo imports via the sys.path.append idiom in `scripts/example_hello.py` |

The first three folders and `scripts/` are **configurable in Settings**
(gear icon): each category searches the in-repo default folder plus any
extra folders you add. MC methods are internal (not a settings category).

`on_progress(current, total, message)` is the universal progress callback
(transforms + optimizer engine). Cancellation = an exception raised INSIDE
the callback (`modules/common/ui/workers.py` does this) — the engine's pool
shutdown depends on it.

## Settings & data roots

`settings.json` (repo root, gitignored, auto-created) holds the extra plugin
folders and the **data roots**. Each data root is a full tree:

```
<root>/raw_dbn/{type}/{ASSET}/{dataset}/   *.dbn.zst   (immutable inputs)
<root>/parquet/{type}/{asset}/{dataset}/   YYYY-MM-DD.parquet (working layer)
<root>/trades/{name}.parquet               backtest outputs (flat)
<root>/optimizations/{run}/                optimizer runs (trades.parquet + meta.json)
<root>/news_and_holidays/ff_usd_events.parquet
```

Pickers show the union across roots; **outputs are written to the root the
input came from**. The default root is `D:/market_data` (machine-local;
`DEFAULT_DATA_ROOT` in `modules/common/backend/settings.py` — the data tree
moved out of the in-repo `data/` in July 2026).

## Repo map

```
main.py                    entry point (spawn-safe __main__ guard only)
modules/
  app.py                   QApplication bootstrap
  main_menu/               launcher window (cards, settings gear)
  common/
    backend/               pure, Qt-free: settings, asset_info (THE single
                           ASSET_INFO + HIDDEN_PARAMS), plugins (multi-folder
                           discovery/loading), data_roots (multi-root scans,
                           output routing, ff-events resolution), trade_files
                           (save_trades + save_temp_trades + filter
                           kv-metadata), trade_stats (compute_metrics,
                           DAY_TYPE_ORDER, RR series), benchmark (α/β
                           regression), chart_window
    ui/                    shared Qt: theme, workers (FunctionWorker +
                           cancellation), widgets, params_form, dataframe
                           model, settings dialog, charts/ (pyqtgraph:
                           equity, candlestick, histogram, fan, heatmap,
                           path), trade_report/ (the shared report panel +
                           TradeActionsRow (Save Trades / Go to Analytics /
                           Go to Monte Carlo), used by Backtester AND
                           Optimizer cell detail)
  data_formatter/          backend/scan.py + window.py
  backtester/              backend/{run,day_types}.py + window.py
  analytics/               backend/{io,sizing,costs,metrics}.py +
                           instance_editor/results_view/window.py
  monte_carlo/             methods/ (plugin dir) + backend/{stats,cost_ctx}.py
                           + prop_firm_panel/window.py
  optimizer/               backend/ = the former optimization/ package
                           (engine, param_space, metrics, buckets, io, loader,
                           combine/, + heatmap_model, run_setup) — pure,
                           tested; UI: sweep_panel, new_run_tab, explore_tab,
                           cell_detail, combine_tab, window.py
  scripts/                 quick-script launcher: backend/{ports,scan,browser}.py
                           (pure) + process_manager.py (QProcess per script
                           instance) + log_panel.py + window.py
strategies/                strategy plugins (single-file or package)
data_transforms/           raw DBN -> enriched parquet plugins
position_sizing/           fixed.py, kelly.py, risk_based.py
scripts/                   quick-script plugins for the Scripts module
forex_factory_scraper/     FF calendar text -> ff_usd_events.parquet
orderbook_replay_cpp/      C++ (pybind11) L3 order-book replay kernel
tests/                     pytest suite (optimizer backend + metrics + Qt smoke)
```

(The data root lives OUTSIDE the repo at `D:/market_data` since July 2026.)

**Convention:** inside every module, `backend/` is pure computation with NO
Qt imports (safe for process-pool workers and tests); `window.py` + other UI
files are the PySide6 frontend.

## The pipeline (how modules cross-connect)

```
raw_dbn/  --(Data Formatter + a transform)-->  parquet/  (one YYYY-MM-DD.parquet per day)
parquet/  --(Backtester + a strategy)-------->  trades/{name}.parquet  (+ day_type from FF data)
parquet/  --(Optimizer + a strategy grid)---->  optimizations/{run}/  (all cells' trades + meta)
optimizations/{container}/ --(Combine)------->  {container}/_combined/{run}/  (variant-set selection, no re-runs)
trades/   --(Analytics + a sizer)------------>  sized equity curve + $ metrics
trades/   --(Monte Carlo + a sizer)---------->  equity_matrix -> fan chart + stats
```

Data passes **as files on disk** between stages (parquet) and **as
DataFrames** within a stage. The contract between stages is the parquet
column schema, not Python imports.

## Conventions that bite if ignored

- **Three-level hierarchy** `type/asset/dataset` is enforced everywhere.
  Asset folders are UPPERCASE tickers (`ES`, `NQ`, `GC`). One parquet **per
  calendar day**, named `YYYY-MM-DD.parquet`.
- **Index** of every candle parquet is a tz-aware `DatetimeIndex` in
  `America/New_York`. (Charts convert to NY-wall-clock epoch for pyqtgraph —
  display only.)
- **`direction`** is lowercase `"long"` / `"short"` everywhere.
- **`pnl_points`, not ticks.** Strategies output `pnl_points` only; the
  backtester converts via `ticks = pnl_points * ticks_per_point` from
  `ASSET_INFO`. Never store `ticks` in a strategy.
- **OHLC is float64.**
- **`volume_delta_pct`** = `volume_delta / volume * 100`, bounded ±100;
  zero-volume bars = `0.0`.
- **Enriched columns** (`tick_volume`, `passive_orders`) exist only for
  **ES/NQ**. Order-flow strategies need the enriched set.
- **Asset from filename:** Analytics / Monte Carlo derive the asset from the
  trades filename's FIRST underscore token (`ES_...parquet` → ES).
- **Qt spin boxes need explicit ranges** — the 0..99.99 default silently
  clamps real values (a logic bug, not a cosmetic one).
- **Backend never imports Qt** — a worker-process import chain that pulls in
  PySide6 is a bug (tests/test_qt_smoke.py enforces this).
- **No `from __future__ import annotations` in a plugin file that defines a
  `@dataclass`.** The plugin loader execs files WITHOUT `sys.modules`
  registration; on Python 3.13 string dataclass annotations crash
  `dataclasses._is_type` (`sys.modules.get(module)` is `None`) at UI load
  time. Normal imports (pytest) don't catch it — load plugins via
  `plugins.load_module` in tests (see tests/test_options_gex.py).

## ASSET_INFO / HIDDEN_PARAMS

`modules/common/backend/asset_info.py` is the ONE copy (the old app had four)
mapping ticker → `{tick_size, ticks_per_point, dollars_per_tick,
commissions_per_contract, parent}` (`parent` links micros to full-size
contracts). `HIDDEN_PARAMS = {"tick_size"}` lives there too — auto-injected
into strategy params, never shown in the UI.

## C++ extension (`orderbook_replay_cpp`)

pybind11/setuptools module used by the `1s_mbo_*` transforms to replay L3
(MBO) events into per-second book snapshots (`replay_full`,
`replay_cropped`). Output is byte-identical to the pre-July-2026 kernel, so
existing parquet stays valid. Builds with MSVC (VS 2022 Community,
auto-detected). Rebuild:

```bash
./venv/Scripts/python.exe -m pip install ./orderbook_replay_cpp
```

`1s_mbo_cropped.py` **requires** it; `1s_mbo_full_book.py` has a pure-Python
fallback.

## Where to read more

- `Quant_app_documentation.pdf` — module contracts & schemas (Streamlit-era
  UI sections outdated).
- `IVB_Model_Documentation.pdf` + `strategies/ivb_model/CLAUDE.md` — the
  flagship strategy.
- The pre-rebuild Streamlit frontend lives only in git history (main branch,
  commit 468c843 and earlier).
