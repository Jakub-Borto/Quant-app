"""
modules.monte_carlo — the Monte Carlo module.

methods/   drop-in simulation methods (former top-level `monte_carlo/` folder).
           Scanned as a plugin directory — drop a .py file here and it appears
           in the Monte Carlo window's method picker. `base.py` (shared engine
           utilities) and `__init__.py` are excluded from the scan. NOTE:
           methods/ deliberately has no __init__.py; its files are loaded by
           file path and import siblings via a sys.path hack (`from base
           import ...`), exactly as before the move.
backend/   pure result-statistics helpers (added in the rebuild).
UI files (window.py, prop_firm_panel.py) are added by the PySide6 frontend.
"""
