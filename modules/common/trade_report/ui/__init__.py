"""
The Qt half of the trade report. This is the one import line a caller needs:

    from modules.common.trade_report.ui import (ReportContext, SaveTarget,
                                                TradeReport, attach_layout_gear)
"""

from ..backend.frame import KEEP, ReportContext, SaveTarget
from .filters import CheckboxFilterRow, make_day_type_filter, make_regime_filter
from .layout_dialog import ReportLayoutDialog, attach_layout_gear
from .report import TradeReport

__all__ = ["KEEP", "ReportContext", "SaveTarget", "TradeReport",
           "CheckboxFilterRow", "make_day_type_filter", "make_regime_filter",
           "ReportLayoutDialog", "attach_layout_gear"]
