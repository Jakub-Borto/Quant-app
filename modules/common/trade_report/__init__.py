"""
modules.common.trade_report — the shared trade report.

ONE implementation, used by the Backtester, the Optimizer's cell drill-down
and the Combine drill-down.

    backend/   pure, Qt-free: the canonical frame contract + filter
               primitives (frame.py), the section registry (layout.py), the
               statistics (trade_stats.py), the notes model (trade_notes.py),
               the α/β regression (benchmark.py), chart windowing.
    ui/        the PySide6 half: TradeReport (report.py), the section stack,
               the panel, and one file per section.

THIS MODULE MUST NOT IMPORT Qt. Importing anything under backend/ executes
this file, and a Qt export here would drag PySide6 into optimizer pool
workers (tests/test_qt_smoke.py::test_worker_import_chain_is_qt_free). The Qt
API is exported from `modules.common.trade_report.ui` instead.

Note: ui/regime_section.py imports modules.regime_detector.backend.io — a
shared package reaching into a feature module. It is deliberate and safe
(regime_detector's backend imports nothing from modules.common, so there is
no cycle); do not "fix" it by hoisting anything the other way.
"""

from .backend.frame import (KEEP, ReportContext, SaveTarget, TradeFrameError,
                            canonical_trades, from_optimizer_rows, save_name)
from .backend.layout import (DEFAULT_ORDER, DEFAULT_SECTIONS, MODE_COLLAPSED,
                             MODE_HIDDEN, MODE_LABELS, MODE_VISIBLE, MODES,
                             REPORT_KEYS, SPEC_BY_KEY, SectionSpec,
                             resolve_layout)

__all__ = ["KEEP", "ReportContext", "SaveTarget", "TradeFrameError",
           "canonical_trades", "from_optimizer_rows", "save_name",
           "DEFAULT_ORDER", "DEFAULT_SECTIONS", "MODE_COLLAPSED",
           "MODE_HIDDEN", "MODE_LABELS", "MODE_VISIBLE", "MODES",
           "REPORT_KEYS", "SPEC_BY_KEY", "SectionSpec", "resolve_layout"]
