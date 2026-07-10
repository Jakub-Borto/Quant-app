"""
Quant Research Platform — desktop app entry point.

    python main.py

SPAWN SAFETY: the Optimizer runs grids on a ProcessPoolExecutor; on Windows
(spawn) every worker process RE-IMPORTS this module. Nothing may execute at
import time — no Qt import, no QApplication — or every pool worker would
bootstrap the GUI. Keep everything inside the __main__ guard.
"""

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    from modules.app import main

    raise SystemExit(main())
