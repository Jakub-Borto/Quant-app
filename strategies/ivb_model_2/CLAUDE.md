# CLAUDE.md — IVB Model 2 (package)

Strategy-package guide for `strategies/ivb_model_2_folder_2/`. Long-form rationale and the
trade logic live in `IVB_Model_Documentation.pdf`; this file is the code map and the contract.

## What it is

An **Initial Balance (IB) breakout** intraday futures strategy with **order-flow confirmation**.
Each RTH day: define the IB, build its volume profile, detect a breakout, wait for a retest of the
value area, then require one of **four** absorption-based entry patterns before entering. Supports
direction flipping on invalidation.

Loaded by the backtester as a **package** (not a single file) via `__init__.py`, which exposes
`run`, `PARAMS`, `PARAM_SECTIONS`.

## File responsibilities

| File | Provides | Needs / Returns |
|---|---|---|
| `__init__.py` | `run(folder_path, start, end, params) -> DataFrame` | globs `YYYY-MM-DD.parquet`, calls `process_day` per day, returns `OUTPUT_COLUMNS` frame |
| `params.py` | `PARAMS`, `PARAM_SECTIONS`, `OUTPUT_COLUMNS` | the full default param dict + UI grouping + output schema |
| `core.py` | `detect_breakout`, `detect_retest`, `find_entry`, `process_day` | orchestrates one day; dispatches entry finders |
| `profile.py` | `compute_ivb_profile(ib_bars) -> (poc, vah, val)` | reads `tick_volume`; peak-based 70% value area |
| `baselines.py` | `build_rolling_baseline`, `build_passive_baseline`, `build_two_bar_baseline`, `merge_tick_volume` | volume-per-tick baselines used by absorption grading |
| `absorption.py` | `is_absorption_candle(...)`, `find_absorption_trigger(...)` | shared wick-absorption grading from `tick_volume` |
| `entries/` | 4 finders + `FINDER_REGISTRY` | each `find_entry(...) -> (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)` |
| `risk/` | `compute_sl_tp`, `run_trade` (+ `trailing.py` placeholder) | SL/TP placement + fill simulation |

## Per-day execution flow (`process_day`)

```
run() -> for each day file -> process_day(session, params):
  1. slice RTH (09:30–16:00 NY); need >= ib_minutes bars
  2. IB high/low from first ib_minutes bars; abort if range <= 0
  3. compute_ivb_profile(ib_bars)            -> poc, vah, val   (profile.py)
  4. build_rolling_baseline / build_passive_baseline (long & short)  (baselines.py)
  5. detect_breakout(post_ib)                -> direction, breakout_pos
  6. loop (<= max_flips):
       detect_retest()                       -> retest_pos (else no trade)
       post_retest = [breakout bar] + [retest .. retest+entry_window]
       find_entry()  -> calls all enabled finders, earliest entry wins
          if entry      -> break
          if invalidate -> flip direction, resume after invalidation_ts, re-detect breakout
  7. compute_sl_tp(post_retest)              -> sl, tp          (risk/sl_tp.py)
  8. run_trade(post_entry)                   -> trade dict      (risk/sl_tp.py)
  9. attach trade_type + notes(JSON)
```

`find_entry` reads the `valid_entries` flag string (`"1111"`, one bit per finder in
`FINDER_REGISTRY` order). It runs each enabled finder over the same `post_retest` window and shared
baselines, then returns the **earliest `entry_ts`**; if none entered, the **earliest `invalidation_ts`**
(which drives a flip).

## The four entry finders (`entries/`)

All share the signature and return tuple
`(entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)`. Whole-trade invalidation =
close back through `val` (long) / `vah` (short). Entry always = **open of the bar after** the
confirming candle.

1. **`absorption_delta`** — an absorption candle, then a confirming candle (correct direction,
   body ≥ `body_threshold`, `volume_delta_pct` past `±delta_threshold`). Uses `absorption_mult`,
   `wick_threshold`.
2. **`consecutive_absorption`** — `consec_abs_n` absorption candles clustered within
   `consec_abs_ticks` of the same level/body-midpoint; **no** delta confirmation. Uses
   `consec_abs_mult`, `consec_wick_threshold`.
3. **`two_bar_absorption`** — a reversal pair (small wicks ≤ `two_bar_wick_ticks`) merged into a
   synthetic candle, graded against a 2-bar paired baseline (`two_bar_abs_mult`), then a confirming
   candle.
4. **`passive_absorption`** — a big resting order on the defended side
   (`size/count ≥ passive_baseline × passive_order_mult`) **and** absorption on the same candle
   (`passive_absorption_mult`, `passive_wick_threshold`), then a confirming candle. This is the only
   finder that consumes `passive_orders`.

### Adding a fifth entry type

1. Drop `entries/my_entry.py` exposing `find_entry(**shared) -> 5-tuple`.
2. Append it to `FINDER_REGISTRY` in `entries/__init__.py`.
3. Extend the default `valid_entries` string by one bit and add any params to `params.py`.

## Required input columns (per `YYYY-MM-DD.parquet`, tz-aware NY index)

Produced by `transforms/1m_advanced.py` (ES/NQ only):

| Column | Used for |
|---|---|
| `open` `high` `low` `close` | IB range, breakout/retest, bar structure, SL/TP |
| `buy_volume` `sell_volume` | rolling absorption baseline (per-tick) |
| `volume_delta_pct` | entry-candle delta confirmation |
| `tick_volume` (JSON `{price:[buy,sell]}`) | volume profile + absorption grading |
| `passive_orders` (JSON `{price:[size,count]}`) | **required** — `passive_absorption` finder + passive baseline |

`volume` is present but not used directly by the core logic.

## Output schema (`OUTPUT_COLUMNS`)

`date, direction, trade_type, entry_time, exit_time, entry_price, exit_price, sl, tp,
exit_reason, pnl_points, notes`

- `trade_type` — which finder fired (`absorption_delta` / `consecutive_absorption` /
  `two_bar_absorption` / `passive_absorption`).
- `exit_reason` — `tp` / `sl` / `eod` / `tp_timeout` / `sl_timeout`.
- `pnl_points` — `exit-entry` (long), `entry-exit` (short). **No `ticks` column** (backtester adds it).
- `notes` — JSON: `breakout_time, retest_time, flip_count, ivb_high, ivb_low, poc, vah, val` merged
  with finder-specific keys (e.g. `absorption_time`, `trigger_price/volume`, passive details).

## Params (see `params.py` for defaults + `PARAM_SECTIONS` grouping)

General: `ib_minutes, trade_timeout, max_flips, valid_entries, risk_script`.
Windows: `retest_window, entry_window, entry_after_absorption, absorption_baseline_window`.
Entry candle: `delta_threshold, body_threshold`. Absorption+delta: `wick_threshold, absorption_mult`.
Consecutive / two-bar / passive: their own groups. Risk: `rr, sl_type` (0=VAL/VAH, 1=swing).
`tick_size` is auto-injected from `ASSET_INFO` (in `HIDDEN_PARAMS`). `risk_script` is reserved —
`risk/__init__.py` currently always uses `sl_tp`; `trailing.py` is a placeholder.

## Pairs with

- **Transform:** `transforms/1m_advanced.py` (enriched ES/NQ 1-minute candles).
- **Backtester:** `views/backtester.py` (injects `tick_size`, converts `pnl_points`→ticks, tags `day_type`).
