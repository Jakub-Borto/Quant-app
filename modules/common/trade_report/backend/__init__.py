"""
The pure, Qt-free half of the trade report — safe to import from tests and
from process-pool workers.

    frame.py         the canonical trades shape + the filter-stage primitives
    layout.py        the section registry and saved-layout reconciliation
    trade_stats.py   metrics and the breakdown tables
    trade_notes.py   flattening + querying the strategies' notes JSON
    benchmark.py     the α/β market-exposure regression
    chart_window.py  windowing a session's candles around one trade
"""
