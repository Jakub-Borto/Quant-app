# CLAUDE.md — IVB Model (modular, vectorized package)

Strategy-package guide for `strategies/ivb_model/`. Long-form rationale and the trade logic
live in `IVB_Model_Documentation.pdf`; this file is the code map and the contract. The engine
contract every strategy follows (DATA, prepare_day, process_day, Trade, caching, lookback) is
documented in full in `STRATEGY_GUIDE.md` at the repo root.

## Engine contract (September 2026 migration)

The package no longer has a `run()`; the engine (`modules/engine/`) owns the day loop, the
column-limited reads + prefetch, the RAM day cache, the timing table and the output frame.
`__init__.py` exposes:

- `DATA = {"main": CANDLE_COLUMNS, "indicators": INDICATOR_COLUMNS}`. The indicators slot is a
  **required** "Additional data" row in the Backtester/Optimizer, auto-picked by folder name
  (`ES_1m_indicators`, `NQ_1m_indicators`). A day whose indicators file is missing is **skipped
  with a warning** by the engine (before the migration it traded without CVD/VWAP bands).
- `PREPARE_PARAMS = ["session_start"]` and `prepare_day(date, data, params)` =
  `core.build_day_core(candles, indicators, session_start_minutes(params))` → `DayData`
  (param-independent apart from the session start; cached by the engine).
- `process_day(day, params)` → `core.process_day(day.prepared, params)` converted to a
  `modules.engine.Trade` (notes as a dict; the engine stores `json.dumps(notes)`).
- `PARAMS`, `PARAM_SECTIONS`, `PARAMS_OPTIONS` — assembled in `params.py` (see below).

Verified byte-identical to the pre-migration package over the full ES and NQ datasets (all three
risk scripts, all seven finders, session-start and min-RR variants).

## Vectorized internals

- `_daydata.py` — `DayData` materializes each day once: OHLC/delta/volume numpy arrays,
  `tick_volume` / `passive_orders` parsed once into per-bar `(prices, a, b)` arrays in JSON
  document order (None = missing/empty/unparseable), per-bar max defended-side passive size,
  baselines/CVD/VWAP bands positionally aligned. `EntryWindow` (post_retest) and
  `TradeWindow` (post_entry) carry the shared masks every finder needs: `invalid`/`first_inv`,
  `confirm_any` (body+delta), `confirm_dir` (+candle direction), `wick_frac`.
- Timing: every stage is wrapped in `modules.engine.timed` (sections like `entry:<finder>`,
  `risk:<script>`, `day:baselines`) and lands in the engine's ONE table per run. Children nest
  inside parents, so percentages overlap.
- **The cached DayData is mutated per run**: `core.process_day` starts with
  `day.reset_run_state()` and rebuilds only the param-dependent state (baselines, CVD std,
  VWAP bands). The engine calls process_day strictly sequentially per process, so this is safe;
  never process one cached DayData from two threads at once.
- **Gated per-run state**: baselines are built only for enabled consumers. Each finder declares
  `BASELINES` (`rolling` / `passive` / `cvd`); core unions the enabled finders (`valid_entries`)
  with the finders the risk script re-runs in-trade (`extra_active_finders`, trailing only) and
  builds exactly those. VWAP bands are attached only when the risk script declares
  `NEEDS_VWAP_BANDS = True`. JSON parsing and the passive-max pass are lazy memoized properties
  on `DayData`, computed at most once per cached day.

## What it is

An **Initial Balance (IB) breakout** intraday futures strategy with **order-flow confirmation**.
Each RTH day: define the IB, build its volume profile, detect a breakout, wait for a retest of the
value area, then require one of **seven** entry patterns before entering. Supports direction
flipping on invalidation.

## File responsibilities

| File | Provides | Needs / Returns |
|---|---|---|
| `__init__.py` | `DATA`, `PREPARE_PARAMS`, `prepare_day`, `process_day` (+ the params) | the engine contract; thin wrappers over `core` |
| `params.py` | `CORE_PARAMS`, `CORE_SECTIONS`, assembled `PARAMS` / `PARAM_SECTIONS` / `PARAMS_OPTIONS` | core params + every finder's and risk script's OWN params (duplicate names raise at import) |
| `_daydata.py` | `DayData`, `EntryWindow`, `TradeWindow`, `parse_json_column`, `prev_rolling_max/min` | the per-day numpy context + shared window masks |
| `core.py` | `build_day_core`, `session_start_minutes`, `detect_breakout`, `detect_retest`, `find_entry`, `process_day`, `VWAP_BAND_COLUMNS` | orchestrates one day on absolute positions; dispatches entry finders + the risk script |
| `profile.py` | `compute_ivb_profile(day, ib_end) -> (poc, vah, val)` | reads the pre-parsed tick_volume; peak-based 70% value area (algorithm untouched) |
| `baselines.py` | `build_rolling_baseline`, `build_passive_baseline`, `build_cvd_change_baseline` (+ `BASELINE_WARMUP_MINUTES`) | day-level baselines as arrays aligned to the session (pandas rolling kept for identical floats); warm-up starts `session_start` + 5 min |
| `absorption.py` | `absorption_scan(tv, wick_low, wick_high, required, direction)`, `wick_bounds(...)` | shared absorption level scan on pre-parsed `tick_volume`; first qualifying level in document order; wick/baseline prechecks live vectorized in the callers |
| `entries/` | 7 finder modules + `FINDER_MODULES` / `FINDER_REGISTRY` / `FINDER_NAMES` / `DEFAULT_FLAGS` | each: `find_entry(win, params) -> (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)` + `PARAMS`, `SECTION`, `DEFAULT_ON`, `BASELINES` |
| `risk/` | self-contained risk scripts + `RISK_SCRIPTS` (name → module) / `RISK_REGISTRY` (name → run) | each: `run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params)` + `PARAMS`, `SECTION`, `OPTIONS`, `NEEDS_VWAP_BANDS` (+ optional `extra_active_finders`) |

The 2-bar-only helpers (`merge_tick_volume`, `build_two_bar_baseline`) live inside
`entries/two_bar_absorption.py` — their sole consumer — so `baselines.py` is purely the three
day-level baselines.

## Per-day execution flow (`process_day`)

```
engine -> prepare_day (cached): build_day_core(candles, indicators, session_start) -> DayData
engine -> process_day(day, params) -> core.process_day(DayData, params):
  1. RTH slice was done in build_day_core (`session_start` "HH:MM" NY, default 09:30, to 16:00)
  2. IB high/low from first ib_minutes bars; abort if the range <= 0
  3. detect_breakout(post_ib)                -> direction, breakout_pos
  4. compute_ivb_profile(ib_bars)            -> poc, vah, val   (profile.py)
  5. gated baselines (rolling / passive / CVD std) + VWAP bands (if the risk script needs them)
  6. loop (<= max_flips):
       detect_retest()                       -> retest_pos (else no trade)
       post_retest = [breakout bar] + [retest .. retest+entry_window]
       find_entry()  -> calls all enabled finders, earliest entry wins
          if entry      -> break
          if invalidate -> flip direction, resume after invalidation, re-detect breakout
  7. risk dispatch: RISK_SCRIPTS[risk_script].run(post_retest, post_entry, ...) -> trade dict | None
  8. attach trade_type + notes (dict: process-day keys, then entry notes, then risk notes)
```

`find_entry` reads the `valid_entries` flag string (one bit per finder in `FINDER_MODULES`
order). It runs each enabled finder over the same `post_retest` window and shared baselines, then
returns the **earliest entry**; if none entered, the **earliest invalidation** (which drives a
flip).

## The seven entry finders (`entries/`)

All share the signature `find_entry(win: EntryWindow, params)` and return the tuple
`(entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)` (bar indices relative to
the window; the dispatcher maps them back to positions/timestamps). Whole-trade invalidation =
close back through `val` (long) / `vah` (short) — precomputed as `win.invalid` / `win.first_inv`.
Entry always = **open of the bar after** the confirming candle. Each finder file declares its
own params (and its `SECTION` box, `DEFAULT_ON` bit and `BASELINES`).

1. **`absorption_delta`** — an absorption candle, then a confirming candle (correct direction,
   body ≥ `body_threshold`, `volume_delta_pct` past `±delta_threshold`). Own params:
   `absorption_mult`, `wick_threshold`.
2. **`consecutive_absorption`** — `consec_abs_n` absorption candles clustered within
   `consec_abs_ticks` of the same level/body-midpoint; **no** delta confirmation. Own params:
   `consec_abs_n`, `consec_abs_mult`, `consec_abs_ticks`, `consec_wick_threshold`. An absorption
   level is dropped once any later candle **closes through** it (long: close strictly below the
   level; short: strictly above) — closing exactly at the level keeps it.
3. **`two_bar_absorption`** — a reversal pair (small wicks ≤ `two_bar_wick_ticks`) merged into a
   synthetic candle, graded against a 2-bar paired baseline (`two_bar_abs_mult`), then a
   confirming candle.
4. **`passive_absorption_size_only`** — a big resting order on the defended side by raw size
   (`size ≥ passive_baseline × passive_size_order_mult`) **and** absorption on the same candle
   (`passive_size_absorption_mult`, `passive_size_wick_threshold`), then a confirming candle.
   Consumes `passive_orders`.
5. **`passive_wall`** — a cluster of `passive_wall_n` big resting orders (raw size ≥
   `passive_baseline × passive_wall_mult`) within `passive_wall_ticks` of one level. No absorption
   candle and no delta confirmation. Consumes `passive_orders`.
6. **`cvd_divergence_absorption`** — a CVD divergence at a price extreme read as absorption (price
   could not extend while CVD pushed further), confirmed by an entry candle. Needs the
   indicators' `cumulative_delta`. Own `cvd_*` params. Off by default.
7. **`cvd_divergence_exhaustion`** — the mirror, read as exhaustion. Own independent `cvd_exh_*`
   params. Off by default.

The two CVD finders are deliberately kept independent — each carries its own copy of
`_test_divergence` / `_is_entry_candle` and its own param set.

### Adding an eighth entry type

1. Drop `entries/my_entry.py` exposing `find_entry(win, params) -> 5-tuple` and its declarations:
   `SECTION = "My Entry"`, `DEFAULT_ON = True|False`, `BASELINES = (...)`,
   `PARAMS = {"my_entry_x": ...}` (names must be unique across the whole strategy).
2. Append the module to `FINDER_MODULES` in `entries/__init__.py`.

That's all: the `valid_entries` / `trail_entries` defaults grow by one bit from `DEFAULT_ON`, the
checkbox groups pick the name up from `FINDER_NAMES`, and `params.py` collects its params and
section. Removing a finder = deleting its file and its `FINDER_MODULES` line (note that
`vwap_trailing_risk` has its own in-trade detector copies keyed by finder position).

## Risk scripts (`risk/`)

`risk_script` selects **by name** from `RISK_SCRIPTS` (in `risk/__init__.py`); an unknown /
legacy value falls back to `basic_risk`. **Every script owns its params under its own prefix —
no param is shared between scripts**, so deleting one script (its file + its `RISK_SCRIPTS`
line) never breaks another:

| `risk_script` | Prefix | Stop | Target |
|---|---|---|---|
| `basic_risk` | `basic_` | `basic_sl_type`: `"VAL/VAH"`, `"swing_low"` or `"zone_logic"` | fixed RR (`basic_rr`); timeout `basic_trade_timeout` |
| `vwap_tp_risk` | `vwap_tp_` | `vwap_tp_sl_placement`: `"VAL/VAH"`, `"zone_logic"` or `"swing_low"` | tick-vwap ±2σ/±3σ band (`vwap_tp_std`, `vwap_tp_session`, `vwap_tp_mode`); min-RR trio `vwap_tp_is_over_rr` / `vwap_tp_minimal_rr` / `vwap_tp_force_trade`; timeout `vwap_tp_trade_timeout` |
| `vwap_trailing_risk` | `trail_` | `trail_sl_placement`, plus a signal-driven trailing stop (`trail_entries`, `trail_in_profit`, `trail_late`) | band via `trail_vwap_std`, `trail_vwap_session`, `trail_tp_mode`; trio `trail_is_over_rr` / `trail_minimal_rr` / `trail_force_trade`; timeout `trail_trade_timeout` |

Old → new names (pre-September-2026 optimizer runs use the old ones): `rr`→`basic_rr`,
`sl_type`→`basic_sl_type`, `trade_timeout`→ one per script, `sl_placement`/`vwap_std`/
`vwap_session`/`vwap_tp_mode` → `vwap_tp_*` and `trail_*` copies, `is_over_rr`/`minimal_rr`/
`force_trade` → `vwap_tp_*`, `trailing_*` → `trail_*`, `late_trailing` → `trail_late`.

Hooks read by `core`: `NEEDS_VWAP_BANDS` (attach the day's VWAP bands) and
`extra_active_finders(params, n)` (the trailing script's in-trade finders, so their baselines
get built).

### The vwap target: tick grid + minimum RR

Both vwap scripts snap the band target to the instrument's **tick grid, toward the entry** (long →
floor, short → ceil) via their own `_tick_tp` / `_tick_tp_array` copies, so the simulated fill sits
on a tradable price and the target is never beyond the raw band. In trailing TP mode the whole
per-bar band array is snapped, not just the entry value. The stops (VAL/VAH, zone, swing) are real
price levels already on the grid and are **not** rounded.

A **minimum-RR switch** can push the target one band out: when it is on and the selected band is too
close to the entry to clear the required reward:risk (measured on the *rounded* target), the target
escalates 2σ → 3σ. Only `std2`/`std3` are loaded (`VWAP_BAND_COLUMNS` in `core.py`). `rr_vwap_push`
in the notes records whether a push happened *and stuck*.

**When no vwap band is usable** — price already past 3σ at entry, *or* no band clears the minimum RR:

| `*_force_trade` | outcome |
|---|---|
| `True` (default) | trade a **fixed-RR** target: `*_minimal_rr × risk` when the min-RR switch is on, else `1 × risk`. `tp_type` reports the multiple (`"1:1"` / `"2:1"`). |
| `False` | **no trade** |

Each risk script is **fully self-contained**: it owns its stop placement (incl. its own copies
of `_zone_sl` / `_swing_sl`) *and* its own copy of the fill simulator. There is no shared
`sl_tp` module and no cross-script imports — the duplication is intentional. All three share the
signature `run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params)`;
`levels` carries `val/vah/poc` and the day context (baselines, CVD, VWAP bands) rides on
`trade_win.day`. Both vwap scripts return **None (no trade)** when the day has no usable VWAP
bands. A risk script may attach a `trade["risk_notes"]` dict; `process_day` pops it and merges it
into `notes`.

In trailing TP mode, only a genuine band TP hit reports the level actually filled in the `tp`
column; every other pre-timeout exit reports the band **at entry**. The post-timeout tail reports
breakeven.

`vwap_trailing_risk` re-detects the entry-style signals on the live trade bars, gated by the
`trail_entries` bit string (same order as `valid_entries`). A signal confirmed by a candle
meeting both `body_threshold` and `delta_threshold` ratchets the stop to the signal candle's
extreme from the next bar on — the stop only ever tightens. `trail_in_profit` (default True)
keeps the signal log breakeven-or-better only; `trail_late` (default False) lags the trail one
signal behind. **Every** trailing signal needs the confirming candle, and there is no VAL/VAH
invalidation in-trade. A hit on a trailed stop reports `exit_reason = "trailing_sl"` with the
trailed level as `exit_price`, while the `sl` column keeps the originally placed stop; applied
trails are logged in `risk_notes` as `trail_count` plus flat `trailN_*` keys per trail.

## Required input columns

`DATA["main"]` (from `data_transforms/1m_advanced.py`, ES/NQ only):

| Column | Used for |
|---|---|
| `open` `high` `low` `close` | IB range, breakout/retest, bar structure, SL/TP |
| `buy_volume` `sell_volume` | rolling absorption baseline (per-tick) |
| `volume_delta_pct` | entry-candle delta confirmation |
| `tick_volume` (JSON `{price:[buy,sell]}`) | volume profile + absorption grading |
| `passive_orders` (JSON `{price:[size,count]}`) | `passive_*` finders + passive baseline |

`DATA["indicators"]` (from `data_transforms/1m_advanced_indicators.py`):

| Column(s) | Used for |
|---|---|
| `cumulative_delta` | CVD pivots (both `cvd_divergence_*` finders + the CVD trail detectors) + `build_cvd_change_baseline` |
| `vwap_tick_{globex,rth}_std{2,3}_{up,dn}` | vwap risk scripts' deviation-band targets (`VWAP_BAND_COLUMNS` in `core.py`) |

A declared column missing from a file is an engine error naming the column and file.

## Output

The engine's 12 columns: `date, direction, trade_type, entry_time, exit_time, entry_price,
exit_price, sl, tp, exit_reason, pnl_points, notes`.

- `trade_type` — which finder fired (one of the seven names above).
- `exit_reason` — `tp` / `sl` / `eod` / `tp_timeout` / `sl_timeout` (+ `trailing_sl`).
- `pnl_points` — computed by the engine: `exit-entry` (long), `entry-exit` (short).
- `notes` — JSON: `breakout_time, retest_time, flip_count, ivb_high, ivb_low, poc, vah, val` merged
  with finder-specific keys and any risk-script `risk_notes` (in that order).

## Params

Core (in `params.py`): `tick_size` (auto-filled from ASSET_INFO, read-only), `session_start`
("HH:MM" NY; anchors the RTH slice, the IB and the baseline warm-up; a PREPARE_PARAM), `ib_minutes`,
`max_flips`, `valid_entries`, `risk_script`; windows `retest_window, entry_window,
entry_after_absorption, absorption_baseline_window`; entry candle `delta_threshold,
body_threshold`. Then every finder's section and every risk script's section, collected from their
own files.

## Pairs with

- **Transform:** `data_transforms/1m_advanced.py` (enriched ES/NQ 1-minute candles).
- **Indicators:** `data_transforms/1m_advanced_indicators.py` → the `indicators` DATA slot.
- **Engine:** `modules/engine/` (day loop, cache, Trade); **Backtester** converts `pnl_points` →
  ticks and tags `day_type`.

## Roadmap / ideas

Future ideas captured in code as comments rather than implemented (see the comment block at the
end of `risk/vwap_tp_risk.py`):
- if the POC is too close to VAH/VAL, place the SL somewhere else;
- if price is on the other side of VWAP, consider targeting the 2nd standard-deviation band.
- a `big_trades` DATA slot (the ES/NQ `*_big_trades` datasets) once a finder consumes it.
