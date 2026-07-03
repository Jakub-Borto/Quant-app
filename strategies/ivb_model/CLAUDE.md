# CLAUDE.md — IVB Model (modular package)

Strategy-package guide for `strategies/ivb_model/`. Long-form rationale and the
trade logic live in `IVB_Model_Documentation.pdf`; this file is the code map and the contract.

## What it is

An **Initial Balance (IB) breakout** intraday futures strategy with **order-flow confirmation**.
Each RTH day: define the IB, build its volume profile, detect a breakout, wait for a retest of the
value area, then require one of **seven** entry patterns before entering. Supports direction
flipping on invalidation.

Loaded by the backtester as a **package** (not a single file) via `__init__.py`, which exposes
`run`, `PARAMS`, `PARAM_SECTIONS`.

## File responsibilities

| File | Provides | Needs / Returns |
|---|---|---|
| `__init__.py` | `run(folder_path, start, end, params) -> DataFrame` | globs `YYYY-MM-DD.parquet`, resolves the indicators folder, calls `process_day` per day, returns `OUTPUT_COLUMNS` frame |
| `params.py` | `PARAMS`, `PARAM_SECTIONS`, `OUTPUT_COLUMNS` | the full default param dict + UI grouping + output schema |
| `core.py` | `detect_breakout`, `detect_retest`, `find_entry`, `process_day` | orchestrates one day; dispatches entry finders + the risk script |
| `profile.py` | `compute_ivb_profile(ib_bars) -> (poc, vah, val)` | reads `tick_volume`; peak-based 70% value area |
| `baselines.py` | `build_rolling_baseline`, `build_passive_baseline`, `build_cvd_change_baseline` (+ `BASELINE_WARMUP_START`) | day-level volume-per-tick / passive-size / CVD-change baselines |
| `absorption.py` | `is_absorption_candle(...)`, `find_absorption_trigger(...)` | shared wick-absorption grading from `tick_volume` |
| `entries/` | 7 finders + `FINDER_REGISTRY` | each `find_entry(**shared) -> (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)` |
| `risk/` | `RISK_REGISTRY` of self-contained risk scripts | SL/TP placement + fill simulation; selected by `risk_script` |

The 2-bar-only helpers (`merge_tick_volume`, `build_two_bar_baseline`) live inside
`entries/two_bar_absorption.py` — their sole consumer — so `baselines.py` is purely the three
day-level baselines.

## Per-day execution flow (`process_day`)

```
run() -> for each day file -> process_day(session, params, ind_df):
  1. slice RTH (09:30–16:00 NY); need >= ib_minutes bars
  2. IB high/low from first ib_minutes bars; abort if range <= 0
  3. compute_ivb_profile(ib_bars)            -> poc, vah, val   (profile.py)
  4. build_rolling_baseline / build_passive_baseline (long & short)  (baselines.py)
  5. CVD + cvd_change_std from ind_df (if present); VWAP deviation bands from ind_df (if present)
  6. detect_breakout(post_ib)                -> direction, breakout_pos
  7. loop (<= max_flips):
       detect_retest()                       -> retest_pos (else no trade)
       post_retest = [breakout bar] + [retest .. retest+entry_window]
       find_entry()  -> calls all enabled finders, earliest entry wins
          if entry      -> break
          if invalidate -> flip direction, resume after invalidation_ts, re-detect breakout
  8. risk dispatch: RISK_REGISTRY[risk_script-1](post_retest, post_entry, ...) -> trade dict | None
  9. attach trade_type + notes(JSON, incl. any risk_notes)
```

`find_entry` reads the `valid_entries` flag string (one bit per finder in `FINDER_REGISTRY`
order). It runs each enabled finder over the same `post_retest` window and shared baselines, then
returns the **earliest `entry_ts`**; if none entered, the **earliest `invalidation_ts`** (which
drives a flip).

## The seven entry finders (`entries/`)

All share the signature and return tuple
`(entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)`. Whole-trade invalidation =
close back through `val` (long) / `vah` (short). Entry always = **open of the bar after** the
confirming candle. The `valid_entries` bit order is exactly the `FINDER_REGISTRY` order below.

1. **`absorption_delta`** — an absorption candle, then a confirming candle (correct direction,
   body ≥ `body_threshold`, `volume_delta_pct` past `±delta_threshold`). Uses `absorption_mult`,
   `wick_threshold`.
2. **`consecutive_absorption`** — `consec_abs_n` absorption candles clustered within
   `consec_abs_ticks` of the same level/body-midpoint; **no** delta confirmation. Uses
   `consec_abs_mult`, `consec_wick_threshold`.
3. **`two_bar_absorption`** — a reversal pair (small wicks ≤ `two_bar_wick_ticks`) merged into a
   synthetic candle, graded against a 2-bar paired baseline (`two_bar_abs_mult`), then a confirming
   candle.
4. **`passive_absorption_size_only`** — a big resting order on the defended side by raw size
   (`size ≥ passive_baseline × passive_size_order_mult`) **and** absorption on the same candle
   (`passive_size_absorption_mult`, `passive_size_wick_threshold`), then a confirming candle.
   Consumes `passive_orders`.
5. **`passive_wall`** — a cluster of `passive_wall_n` big resting orders (raw size ≥
   `passive_baseline × passive_wall_mult`) within `passive_wall_ticks` of one level. No absorption
   candle and no delta confirmation — the wall of stacked liquidity is the signal. Consumes
   `passive_orders`.
6. **`cvd_divergence_absorption`** — a CVD divergence at a price extreme read as absorption (price
   could not extend while CVD pushed further), confirmed by an entry candle. **Requires CVD** from
   the indicators folder; disables itself when CVD is absent.
7. **`cvd_divergence_exhaustion`** — the mirror, read as exhaustion (price did extend to a new
   extreme but CVD could not follow), confirmed by an entry candle. **Requires CVD.**

The two CVD finders are deliberately kept independent — each carries its own copy of
`_test_divergence` / `_is_entry_candle` rather than sharing a helper.

### Adding an eighth entry type

1. Drop `entries/my_entry.py` exposing `find_entry(**shared) -> 5-tuple`.
2. Append it to `FINDER_REGISTRY` in `entries/__init__.py`.
3. Extend the default `valid_entries` string by one bit and add any params to `params.py`.

## Risk scripts (`risk/`)

`risk_script` is a **1-based selector** into `RISK_REGISTRY` (in `risk/__init__.py`):

| `risk_script` | Script | Stop | Target |
|---|---|---|---|
| 1 | `basic_risk` | VAL/VAH (`sl_type=0`) or swing (`sl_type=1`) | fixed RR (`rr`) |
| 2 | `zone_sl_risk` | pullback-extreme vs VAL/POC/VAH zones | fixed RR (`zone_rr`) |
| 3 | `vwap_tp_risk` | VAL/VAH or zone logic (`sl_placement`) | tick-vwap ±2σ/±3σ band (`vwap_std`, `vwap_session`, `vwap_tp_mode`) |

Each risk script is **fully self-contained**: it owns its stop placement *and* its own copy of the
fill simulator (`_run_trade`, plus `_run_trade_trailing` in `vwap_tp_risk`). There is no shared
`sl_tp` module and no cross-script imports — the duplication is intentional so each script can be
edited in isolation. `vwap_tp_risk` returns **None (no trade)** when the day has no VWAP bands.
A risk script may attach a `trade["risk_notes"]` dict (e.g. `tp_type`, `escalated`); it is popped
in `process_day` and merged into `notes` rather than becoming a stray column.

## Required input columns (per `YYYY-MM-DD.parquet`, tz-aware NY index)

Candles produced by `transforms/1m_advanced.py` (ES/NQ only):

| Column | Used for |
|---|---|
| `open` `high` `low` `close` | IB range, breakout/retest, bar structure, SL/TP |
| `buy_volume` `sell_volume` | rolling absorption baseline (per-tick) |
| `volume_delta_pct` | entry-candle delta confirmation |
| `tick_volume` (JSON `{price:[buy,sell]}`) | volume profile + absorption grading |
| `passive_orders` (JSON `{price:[size,count]}`) | `passive_*` finders + passive baseline |

`volume` is present but not used directly by the core logic.

### Indicators folder (optional, per `indicators_folder` param)

`__init__.py` resolves a sibling dataset under the same `parquet/{type}/{asset}/` and loads a
matching `YYYY-MM-DD.parquet` per day. When absent (empty param / missing file / bad read) the day
runs without indicators: the two CVD finders disable themselves and `vwap_tp_risk` returns no
trade. Columns consumed:

| Column(s) | Used for |
|---|---|
| `cumulative_delta` | CVD pivots (both `cvd_divergence_*` finders) + `build_cvd_change_baseline` |
| `vwap_tick_{globex,rth}_std{2,3}_{up,dn}` | `vwap_tp_risk` deviation-band targets (see `VWAP_BAND_COLUMNS` in `core.py`) |

`big_trades_folder` is declared in `PARAMS` but **reserved** (not yet consumed).

## Output schema (`OUTPUT_COLUMNS`)

`date, direction, trade_type, entry_time, exit_time, entry_price, exit_price, sl, tp,
exit_reason, pnl_points, notes`

- `trade_type` — which finder fired (one of the seven names above).
- `exit_reason` — `tp` / `sl` / `eod` / `tp_timeout` / `sl_timeout`.
- `pnl_points` — `exit-entry` (long), `entry-exit` (short). **No `ticks` column** (backtester adds it).
- `notes` — JSON: `breakout_time, retest_time, flip_count, ivb_high, ivb_low, poc, vah, val` merged
  with finder-specific keys and any risk-script `risk_notes`.

## Params (see `params.py` for defaults + `PARAM_SECTIONS` grouping)

General: `ib_minutes, trade_timeout, max_flips, valid_entries, risk_script, indicators_folder,
big_trades_folder`. Windows: `retest_window, entry_window, entry_after_absorption,
absorption_baseline_window`. Entry candle: `delta_threshold, body_threshold`. Then per-finder
groups (absorption+delta, consecutive, two-bar, passive size-only, passive wall, CVD divergence)
and per-risk-script groups (basic: `rr`/`sl_type`; zone: `zone_rr`; vwap: `sl_placement`/
`vwap_std`/`vwap_session`/`vwap_tp_mode`).

`PARAM_SECTIONS` intentionally omits `tick_size`: it is auto-injected from `ASSET_INFO` by the
backtester and listed in `HIDDEN_PARAMS`, so it has no UI widget. (That gap is deliberate, not a
missing entry.)

## Pairs with

- **Transform:** `transforms/1m_advanced.py` (enriched ES/NQ 1-minute candles).
- **Indicators:** the dataset named by `indicators_folder` (CVD + tick-VWAP bands).
- **Backtester:** `views/backtester.py` (injects `tick_size`, converts `pnl_points`→ticks, tags `day_type`).

## Roadmap / ideas

Future ideas captured in code as comments rather than implemented (see the comment block at the
end of `risk/vwap_tp_risk.py`):
- if the POC is too close to VAH/VAL, place the SL somewhere else;
- if price is on the other side of VWAP, consider targeting the 2nd standard-deviation band.
