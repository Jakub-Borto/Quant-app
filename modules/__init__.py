"""
modules — application modules of the Quant Research Platform desktop app.

Each subpackage is one module of the app (data_formatter, backtester,
analytics, monte_carlo, optimizer) plus:

  common/     shared infrastructure (pure backend helpers + shared Qt widgets)
  main_menu/  the launcher window
  app.py      QApplication bootstrap

Convention: inside every module, `backend/` is pure computation with NO Qt
imports (safe for process-pool workers), and `window.py` (+ other UI files)
is the PySide6 frontend.
"""
