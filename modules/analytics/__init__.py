"""
modules.analytics — the Analytics (position sizing + $ metrics) module.

backend/   pure logic extracted verbatim from legacy_streamlit/views/
           analytics.py: trades IO, sizer execution, the cost model
           (commissions + slippage) and the dollar-space metric registry.
UI files (window.py, instance_editor.py, results_view.py) are the PySide6
frontend added in the rebuild.
"""
