"""Risk management scripts: stop/target placement + trade fill simulation.

Each risk script is fully self-contained — it exposes
    run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params) -> trade dict | None
and carries its own copy of the fill-simulation helpers (no shared module, no cross-script
imports). The order in RISK_REGISTRY maps to the 1-based `risk_script` param
(1 = basic_risk, 2 = zone_sl_risk, 3 = vwap_tp_risk, 4 = vwap_trailing_risk).
"""

from .basic_risk         import run as basic_risk
from .zone_sl_risk       import run as zone_sl_risk
from .vwap_tp_risk       import run as vwap_tp_risk
from .vwap_trailing_risk import run as vwap_trailing_risk

RISK_REGISTRY = [basic_risk, zone_sl_risk, vwap_tp_risk, vwap_trailing_risk]
