"""
Strategy Optimizer support package.

Modules (each importable on its own — no Streamlit dependency):

  param_space.py   infer which PARAMS are sweepable, build value lists from
                   user-chosen min/max/step, enumerate the swept axes' grid
  engine.py        run a strategy across a parameter grid, enrich trades
                   (pnl_ticks, day_bucket), stream progress
  metrics.py       PURE metric functions over a trades DataFrame — used both at
                   build time and at every heatmap filter recompute
  buckets.py       trading-date -> day_bucket mapping from the Forex Factory
                   USD calendar (shared with the backtester's day_type tagging)
  io.py            save/load an optimization run under data/optimizations/

The UI lives in views/optimizer.py; this package holds everything testable.
"""
