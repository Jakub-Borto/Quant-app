"""
modules.common — shared infrastructure for all app modules.

backend/       pure helpers (no Qt): settings, asset info, plugin loading,
               data-root scanning, trades-file saving, regime joining.
ui/            shared PySide6 widgets: theme, workers, params form, charts,
               dataframe model, settings dialog.
trade_report/  THE shared trade report — its own backend/ + ui/ split. One
               implementation, used by the Backtester and both Optimizer
               drill-downs; see its __init__ for the public API.
"""
