"""Risk management: stop/target placement, trade simulation, trailing (future).

Each risk script exposes run(post_retest, post_entry, entry_ts, entry_price, direction,
levels, params) -> trade dict | None. The order in RISK_REGISTRY maps to the 1-based
`risk_script` param (1 = basic_risk, 2 = zone_sl_risk).
"""

from .sl_tp        import compute_sl_tp, run_trade
from .basic_risk   import run as basic_risk
from .zone_sl_risk import run as zone_sl_risk

RISK_REGISTRY = [basic_risk, zone_sl_risk]
