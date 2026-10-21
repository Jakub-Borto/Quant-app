"""
Reports — research reports written as Streamlit apps.

backend/  pure, Qt-free: discovery, archive/restore/delete, the Compress tool
          (full Optimizer run -> compact per-combination summary), the launch
          command + the report bootstrap
kit/      the Qt-free package report scripts import (style + heatmap +
          saved views)
window.py the PySide6 launcher (Reports / Archive / Compress tabs)

Keep this __init__ Qt-free: report processes import modules.reports.kit
through it.
"""
