# CLAUDE.md — IVB Model (modular package, vectorized copy)

Strategy-package guide for `strategies/ivb_model_optimized/` — the **vectorized copy** of
`strategies/ivb_model/`. Trade logic and outputs are byte-identical to the original (verified
per-config against it over the full ES dataset); only the internals changed: per-day numpy
context instead of DataFrame slices, JSON parsed once per day, and accumulating stage timers
printed once per `run()`. Long-form rationale and the trade logic live in
`IVB_Model_Documentation.pdf`; this file is the code map and the contract.

## Vectorized internals (what differs from `ivb_model`)

- `_daydata.py` — `DayData` materializes each day once: OHLC/delta/volume numpy arrays,
  `tick_volume` / `passive_orders` parsed once into per-bar `(prices, a, b)` arrays in JSON
  document order (None = missing/empty/unparseable), per-bar max defended-side passive size,
  baselines/CVD/VWAP bands positionally aligned. `EntryWindow` (post_retest) and
  `TradeWindow` (post_entry) carry the shared masks every finder needs: `invalid`/`first_inv`,
  `confirm_any` (body+delta), `confirm_dir` (+candle direction), `wick_frac`.
- `_timing.py` — accumulating timers; `run()` prints ONE table per backtest run
  (`[ivb timing] ...`), sections like `day:daydata_build`, `entry:<finder>`,
  `risk4:trail:<detector>`. Children nest inside parents, so percentages overlap.
- Finder/risk signatures changed (see below); the package-level `run()` contract, `PARAMS`,
  registries and every output stay exactly as before.
- `__init__.py` reads only the consumed parquet columns (full-read fallback) and prefetches
  the next few days on a small thread pool; days are consumed strictly in file order.
- **Day-core cache**: `DayData` depends only on the files + the resolved `session_start`, so
  `__init__` keeps an LRU cache (`_DAY_CACHE`, 600-day cap, ~0.3 MB/day) keyed by
  candle+indicator file (path, mtime, size) plus the session-start minute (different session
  starts cache separately).
  The dict itself lives on a holder module registered in `sys.modules`
  (`_ivb_day_cache_store`) because the backtester's plugin loader re-executes this `__init__`
  on every run — a plain module global would be wiped each time.
  Re-running the backtester with different params skips reads/parsing entirely — warm runs are
  ~0.3–1s vs ~4s cold over 307 ES days. `core.build_day_core` builds a cache entry;
  `process_day(day, params)` starts with `day.reset_run_state()` and rebuilds only the
  param-dependent state.
- **Gated per-run state**: baselines are built only for enabled consumers (rolling for finder
  bits 0/1/3, passive for 3/4, CVD std for 5/6 — entry bits unioned with the trailing bits
  when `risk_script` is `"vwap_trailing_risk"`; VWAP bands attached only for the two vwap
  scripts). JSON parsing and the
  passive-max pass are lazy memoized properties on `DayData`, computed at most once per
  cached day. A `None` baseline is never read: the window contexts skip gathering it and the
  gate uses exactly the bit strings the dispatchers use.

## What it is

An **Initial Balance (IB) breakout** intraday futures strategy with **order-flow confirmation**.
Each RTH day: define the IB, build its volume profile, detect a breakout, wait for a retest of the
value area, then require one of **seven** entry patterns before entering. Supports direction
flipping on invalidation.

Loaded by the backtester as a **package** (not a single file) via `__init__.py`, which exposes
`run`, `PARAMS`, `PARAM_SECTIONS`, `PARAMS_OPTIONS` (UI choice lists — dropdowns for the
mode-selector params, named checkbox groups for the `valid_entries`/`trailing_entries` bit
strings; see `modules/common/ui/params_form.py` for the contract).

## File responsibilities

| File | Provides | Needs / Returns |
|---|---|---|
| `__init__.py` | `run(folder_path, start, end, params) -> DataFrame` | globs `YYYY-MM-DD.parquet`, resolves the indicators folder, column-pruned prefetched reads, calls `process_day` per day, prints the timing table, returns `OUTPUT_COLUMNS` frame |
| `params.py` | `PARAMS`, `PARAM_SECTIONS`, `PARAMS_OPTIONS`, `OUTPUT_COLUMNS` | the full default param dict + UI grouping + choice lists + output schema |
| `_timing.py` | `timed(name)`, `reset()`, `report(wall)` | accumulating stage timers, one printed table per run |
| `_daydata.py` | `DayData`, `EntryWindow`, `TradeWindow`, `parse_json_column`, `prev_rolling_max/min` | the per-day numpy context + shared window masks |
| `core.py` | `detect_breakout`, `detect_retest`, `find_entry`, `process_day` | orchestrates one day on absolute positions; dispatches entry finders + the risk script |
| `profile.py` | `compute_ivb_profile(day, ib_end) -> (poc, vah, val)` | reads the pre-parsed tick_volume; peak-based 70% value area (algorithm untouched) |
| `baselines.py` | `build_rolling_baseline`, `build_passive_baseline`, `build_cvd_change_baseline` (+ `BASELINE_WARMUP_MINUTES`) | day-level baselines as arrays aligned to the session (pandas rolling kept for identical floats); warm-up starts `session_start` + 5 min |
| `absorption.py` | `absorption_scan(tv, wick_low, wick_high, required, direction)`, `wick_bounds(...)` | shared absorption level scan on pre-parsed `tick_volume`; first qualifying level in document order (= the old dict-iteration order); wick/baseline prechecks live vectorized in the callers |
| `entries/` | 7 finders + `FINDER_REGISTRY` + `FINDER_NAMES` | each `find_entry(win, params) -> (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)` — window-relative bar indices into `win.pos` |
| `risk/` | name-keyed `RISK_REGISTRY` of self-contained risk scripts | `run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params)`; SL/TP placement + fill simulation; selected by `risk_script` (name) |

The 2-bar-only helpers (`merge_tick_volume`, `build_two_bar_baseline`) live inside
`entries/two_bar_absorption.py` — their sole consumer — so `baselines.py` is purely the three
day-level baselines.

## Per-day execution flow (`process_day`)

```
run() -> for each day file -> process_day(session, params, ind_df):
  1. slice RTH (`session_start` "HH:MM" NY, default 09:30, through 16:00); need >= ib_minutes bars
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
  8. risk dispatch: RISK_REGISTRY[risk_script](post_retest, post_entry, ...) -> trade dict | None
  9. attach trade_type + notes(JSON, incl. any risk_notes)
```

`find_entry` reads the `valid_entries` flag string (one bit per finder in `FINDER_REGISTRY`
order). It runs each enabled finder over the same `post_retest` window and shared baselines, then
returns the **earliest `entry_ts`**; if none entered, the **earliest `invalidation_ts`** (which
drives a flip).

## The seven entry finders (`entries/`)

All share the signature `find_entry(win: EntryWindow, params)` and return tuple
`(entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)` (bar indices relative to
the window; the dispatcher maps them back to positions/timestamps). Whole-trade invalidation =
close back through `val` (long) / `vah` (short) — precomputed as `win.invalid` / `win.first_inv`.
Entry always = **open of the bar after** the confirming candle. The `valid_entries` bit order is
exactly the `FINDER_REGISTRY` order below.

1. **`absorption_delta`** — an absorption candle, then a confirming candle (correct direction,
   body ≥ `body_threshold`, `volume_delta_pct` past `±delta_threshold`). Uses `absorption_mult`,
   `wick_threshold`.
2. **`consecutive_absorption`** — `consec_abs_n` absorption candles clustered within
   `consec_abs_ticks` of the same level/body-midpoint; **no** delta confirmation. Uses
   `consec_abs_mult`, `consec_wick_threshold`. An absorption level is dropped once any later
   candle **closes through** it (long: close strictly below the level; short: strictly above) —
   closing exactly at the level keeps it. (`absorption_delta` enforces the same rule inside its
   confirm scan via an early break.)
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
   the indicators folder; disables itself when CVD is absent. Uses the `cvd_*` params.
7. **`cvd_divergence_exhaustion`** — the mirror, read as exhaustion (price did extend to a new
   extreme but CVD could not follow), confirmed by an entry candle. **Requires CVD.** Uses its own
   independent `cvd_exh_*` params (pivot_k / min_separation / max_separation /
   wick_tolerance_ticks / min_score).

The two CVD finders are deliberately kept independent — each carries its own copy of
`_test_divergence` / `_is_entry_candle` and its own `cvd_*` / `cvd_exh_*` param set rather than
sharing.

### Adding an eighth entry type

1. Drop `entries/my_entry.py` exposing `find_entry(**shared) -> 5-tuple`.
2. Append it to `FINDER_REGISTRY` in `entries/__init__.py`.
3. Extend the default `valid_entries` AND `trailing_entries` strings by one bit and add any
   params to `params.py` (`PARAMS_OPTIONS["valid_entries"]`/`["trailing_entries"]` mirror
   `FINDER_NAMES`, so the checkbox groups pick the new name up automatically).

## Risk scripts (`risk/`)

`risk_script` selects **by name** from the name-keyed `RISK_REGISTRY` (in `risk/__init__.py`);
an unknown / legacy value falls back to `basic_risk`:

| `risk_script` | Stop | Target |
|---|---|---|
| `basic_risk` | `sl_type`: `"VAL/VAH"`, `"swing_low"` or `"zone_logic"` (pullback-extreme vs VAL/POC/VAH zones) | fixed RR (`rr`) |
| `vwap_tp_risk` | `sl_placement`: `"VAL/VAH"`, `"zone_logic"` or `"swing_low"` | tick-vwap ±2σ/±3σ band (`vwap_std`, `vwap_session`, `vwap_tp_mode`) |
| `vwap_trailing_risk` | as `vwap_tp_risk`, plus a signal-driven trailing stop (`trailing_entries`) | as `vwap_tp_risk` |

(The former `zone_sl_risk` script was folded into `basic_risk` as `sl_type="zone_logic"`; its
`zone_rr` param is gone — the merged mode uses `rr`.)

Each risk script is **fully self-contained**: it owns its stop placement (incl. its own copies
of `_zone_sl` / `_swing_sl`) *and* its own copy of the
fill simulator (`_run_trade`, plus `_run_trade_trailing` in the two vwap scripts). There is no
shared `sl_tp` module and no cross-script imports — the duplication is intentional so each script
can be edited in isolation. All three share the signature
`run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params)` — `entry_win` is
the post_retest `EntryWindow` (for pullback stops), `trade_win` the post_entry `TradeWindow`,
`entry_pos` the absolute day position of the entry bar; `levels` carries `val/vah/poc` and the
day context rides on `trade_win.day` (baselines, CVD, VWAP band arrays). Both vwap scripts return **None (no trade)** when the day has no VWAP
bands. A risk script may attach a `trade["risk_notes"]` dict (e.g. `tp_type`, `escalated`); it is
popped in `process_day` and merged into `notes` rather than becoming a stray column.

`vwap_trailing_risk` re-detects the entry-style signals on the live trade bars, gated by the
`trailing_entries` bit string (same order as `valid_entries`). A signal confirmed by a candle
meeting both `body_threshold` and `delta_threshold` ratchets the stop to the signal candle's
extreme (low long / high short) from the next bar on — the stop only ever tightens.
`trailing_in_profit` (default True) keeps the signal log breakeven-or-better only (an in-loss
level is not even logged); False logs and trails everything, loss included. `late_trailing`
(default False) lags the trail one signal behind: each logged signal moves the stop to the
**previous** logged signal's level, so the first logged signal only arms the log. Unlike the
entry finders, **every** trailing signal needs the confirming candle (also
`consecutive_absorption` and `passive_wall`), and there is no VAL/VAH invalidation in-trade. A
hit on a trailed stop reports `exit_reason = "trailing_sl"` with the trailed level as
`exit_price`, while the `sl` column keeps the originally placed stop; applied trails are logged
in `risk_notes` as `trail_count` plus flat `trailN_*` keys per trail (`trailN_time/type/stop`
and the same fields that finder's `entry_notes` would carry — trigger/passive/wall/CVD-pivot
details — plus `trailN_trigger_type/_trigger_time` for late trails). To support this,
`process_day` passes the day-level baselines and CVD series to every risk script inside
`levels` (older scripts ignore them).

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
| `cumulative_delta` | CVD pivots (both `cvd_divergence_*` finders — absorption via `cvd_*`, exhaustion via `cvd_exh_*` — + the CVD trail detectors) + `build_cvd_change_baseline` |
| `vwap_tick_{globex,rth}_std{2,3}_{up,dn}` | `vwap_tp_risk` / `vwap_trailing_risk` deviation-band targets (see `VWAP_BAND_COLUMNS` in `core.py`) |

`big_trades_folder` is declared in `PARAMS` but **reserved** (not yet consumed).

## Output schema (`OUTPUT_COLUMNS`)

`date, direction, trade_type, entry_time, exit_time, entry_price, exit_price, sl, tp,
exit_reason, pnl_points, notes`

- `trade_type` — which finder fired (one of the seven names above).
- `exit_reason` — `tp` / `sl` / `eod` / `tp_timeout` / `sl_timeout` (+ `trailing_sl` from
  `vwap_trailing_risk`).
- `pnl_points` — `exit-entry` (long), `entry-exit` (short). **No `ticks` column** (backtester adds it).
- `notes` — JSON: `breakout_time, retest_time, flip_count, ivb_high, ivb_low, poc, vah, val` merged
  with finder-specific keys and any risk-script `risk_notes`.

## Params (see `params.py` for defaults + `PARAM_SECTIONS` grouping)

General: `session_start ("HH:MM" NY; anchors the RTH slice, the IB and the baseline warm-up;
part of the day-core cache key), ib_minutes, trade_timeout, max_flips, valid_entries, risk_script, indicators_folder,
big_trades_folder`. Windows: `retest_window, entry_window, entry_after_absorption,
absorption_baseline_window`. Entry candle: `delta_threshold, body_threshold`. Then per-finder
groups (absorption+delta, consecutive, two-bar, passive size-only, passive wall, CVD divergence
absorption `cvd_*` / exhaustion `cvd_exh_*`)
and per-risk-script groups (basic: `rr`/`sl_type` — `"VAL/VAH"`/`"swing_low"`/`"zone_logic"`;
vwap: `sl_placement`/`vwap_std`/`vwap_session`/`vwap_tp_mode`; vwap trailing:
`trailing_entries` + the bool switches `trailing_in_profit`/`late_trailing`).

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
