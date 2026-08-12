"""
modules.optimizer — the Strategy Optimizer module.

backend/   the pure optimization engine (former top-level `optimization/`
           package: param_space, engine, metrics, buckets, io, loader,
           combine/) — no Qt, no Streamlit; pytest-covered.
UI files (window.py, tabs) are added by the PySide6 frontend build.

The drill-down report is NOT here: cell_detail.py and combine_detail.py only
slice trades and build save names, and drive a
modules.common.trade_report.ui.TradeReport that the tabs own.
"""
