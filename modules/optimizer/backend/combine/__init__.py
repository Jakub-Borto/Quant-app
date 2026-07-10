"""
Strategy Combiner core — third Optimizer sub-module (Run / Explore / Combine).

Selects a small, diverse set of entry-variant trade streams from saved
optimizer runs and reports how they perform merged into ONE strategy under a
one-open-position-at-a-time rule — without re-running any backtests.

Modules (pure, no Streamlit):

  merge.py     the no-overlap merge primitive (+ re-sort-free incremental merge)
  pool.py      discover entry runs, group trades into variants, day filter,
               IS/OOS split, min-trades floors
  compat.py    compatibility gate over the selected runs' meta.json
  select.py    greedy forward selection + swap step + redundancy penalty
  evaluate.py  per-set metrics (total ticks, daily Sharpe, max drawdown)
  io.py        persist/load a combine run under {container}/_combined/

Everything scores on the MERGED stream (total pnl_ticks of kept trades) —
never on standalone metrics — because the no-overlap rule makes combined P&L
non-additive: trades collide, and an early mediocre trade can pre-empt a
later better one. Selection is in-sample only; the out-of-sample slice is
sealed until the whole selection path is evaluated at the end.
"""
