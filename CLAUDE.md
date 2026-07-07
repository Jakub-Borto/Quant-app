# CLAUDE.md — Quant Research Platform

Orientation for Claude Code working in this repo. A companion PDF
(`Quant_app_documentation.pdf`) holds the long-form version; this file is the fast map.

## What this is

A modular **intraday futures research platform** (Python + Streamlit) for 30+ instruments.
It turns raw Databento market data into enriched candle datasets, runs vectorized backtests,
applies position sizing, and runs Monte-Carlo stress tests.

Run it:

```bash
streamlit run app.py
```

`app.py` is a single Streamlit process. It routes between **views** via `st.session_state.page`;
each view exposes one `render()` function and navigates with a local `go_page(page)` helper
(no cross-view imports).

## Plugin-drop architecture (the core idea)

Four plugin folders are scanned dynamically (`importlib.util.spec_from_file_location` →
`exec_module`) — **drop a file in the folder and it appears in the UI**. No registration, no
imports to edit. The filename stem becomes the UI display name. `__init__.py` / `base.py` are
excluded.

| Folder | Loaded by | Contract |
|---|---|---|
| `transforms/` | Data Formatter | `run_all(input_folder, output_folder, skip_existing, on_progress) -> None` |
| `strategies/` | Backtester, Optimizer | `run(folder_path, start_date, end_date, params) -> pd.DataFrame` (+ `PARAMS`; the Optimizer sweeps any int/float param over a UI-chosen min/max/step, str params over a value list) |
| `position_sizing/` | Analytics, Monte Carlo | `apply(trades, params) -> pd.DataFrame` (+ `PARAMS`) |
| `monte_carlo/` | Monte Carlo view | `run(trades, sizer_module, sizer_params, params) -> dict` (+ `PARAMS`) |

A strategy may also be a **package** (a folder with `__init__.py` exposing `run`, `PARAMS`,
`PARAM_SECTIONS`) — see `strategies/ivb_model/` and its own `CLAUDE.md`.

## Repo map

```
app.py                     Streamlit entry point / router
views/                     UI pages (home, data_formatter, backtester, analytics, monte_carlo, optimizer);
                           trade_report.py = shared report components (metrics, exposure, equity/trade charts) — not a view
transforms/                raw DBN -> enriched parquet (the run_all plugins)
strategies/                backtest strategies (single-file or package); base.py = helpers
position_sizing/           fixed.py, kelly.py, risk_based.py
monte_carlo/               base.py (utilities), bootstrap.py
optimization/              Strategy Optimizer core (param_space, engine, metrics, buckets, io, loader) — pure, tested;
                           engine runs grids serially or on a process pool (per-worker strategy caches, memory-budgeted worker count)
ff_data_scraper/           Forex Factory calendar text -> ff_usd_events.parquet
heatmap_rs/                Rust (PyO3) L3 order-book replay kernel
tests/                     pytest suite (optimization package + optimizer view smoke)
data/
  raw_dbn/{type}/{asset}/{dataset}/   *.dbn.zst   (immutable inputs)
  parquet/{type}/{asset}/{dataset}/   YYYY-MM-DD.parquet  (working layer)
  trades/{name}.parquet               backtest outputs (flat, no hierarchy)
  optimizations/{run}/                optimizer runs: trades.parquet (all cells) + meta.json
  news_and_holidays/ff_usd_events.parquet
```

## The pipeline (how modules cross-connect)

```
raw_dbn/  --(Data Formatter + a transform)-->  parquet/  (one YYYY-MM-DD.parquet per day)
parquet/  --(Backtester + a strategy)-------->  trades/{name}.parquet  (+ day_type from FF data)
parquet/  --(Optimizer + a strategy grid)---->  optimizations/{run}/  (all cells' trades + meta)
trades/   --(Analytics + a sizer)------------>  sized equity curve + $ metrics
trades/   --(Monte Carlo + a sizer)---------->  equity_matrix -> fan chart + stats
```

Data is passed **as files on disk** between stages (parquet), and **as DataFrames** within a
stage. The contract between stages is the parquet column schema, not Python imports.

## Conventions that bite if ignored

- **Three-level hierarchy** `type/asset/dataset` is enforced everywhere. Asset folders are
  UPPERCASE tickers (`ES`, `NQ`, `GC`). One parquet **per calendar day**, named `YYYY-MM-DD.parquet`.
- **Index** of every candle parquet is a tz-aware `DatetimeIndex` in `America/New_York`.
- **`direction`** is lowercase `"long"` / `"short"` everywhere — exact string matching.
- **`pnl_points`, not ticks.** Strategies output `pnl_points` only; the backtester converts to
  ticks via `ticks = pnl_points * ticks_per_point` from `ASSET_INFO`. Never store `ticks` in a strategy.
- **OHLC is float64.**
- **`volume_delta_pct`** is `volume_delta / volume * 100` — signed order-flow imbalance as a
  percent of total bar volume, bounded to `±100`. Zero-volume bars = `0.0`.
- **Enriched columns** (`tick_volume`, `passive_orders`) exist only for **ES/NQ** (Databento
  subscription window). Plain OHLCV exists for ~30 assets. Order-flow strategies need the enriched set.

## ASSET_INFO / HIDDEN_PARAMS

`ASSET_INFO` (defined in `views/backtester.py`, and mirrored in the other views) maps each
ticker → `{tick_size, ticks_per_point, dollars_per_tick}`. The backtester injects `tick_size`
into strategy params automatically; `HIDDEN_PARAMS = {"tick_size"}` suppresses its UI widget.
Analytics / Monte Carlo derive `dollars_per_tick` from the **first token of the trades filename**
(e.g. `ES_...parquet` → ES) and inject it into sizer params.

## Rust extension (`heatmap_rs`)

PyO3/maturin module used by the `1s_mbo_*` transforms to replay L3 (MBO) events into per-second
book snapshots. Two functions: `replay_full(...)` and `replay_cropped(...)`. To rebuild it see the
`heatmap-rs-build` memory note (maturin + PATH/VIRTUAL_ENV gotchas). `1s_mbo_cropped.py` **requires**
the built extension; `1s_mbo_full_book.py` has a pure-Python fallback.

## Where to read more

- `Quant_app_documentation.pdf` — full module-by-module contracts, schemas, data-flow.
- `IVB_Model_Documentation.pdf` + `strategies/ivb_model/CLAUDE.md` — the flagship strategy.
