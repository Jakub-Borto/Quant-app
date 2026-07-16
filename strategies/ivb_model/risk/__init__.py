"""Risk management scripts: stop/target placement + trade fill simulation.

Each risk script is fully self-contained — it exposes
    run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params) -> trade dict | None
(entry_win = the post_retest EntryWindow, trade_win = the post_entry TradeWindow, entry_pos =
the absolute day position of the entry bar — see _daydata) and carries its own copy of the
fill-simulation helpers (no shared module, no cross-script imports). The `risk_script` param
selects by name from RISK_REGISTRY; dict order is the UI dropdown order (see
params.PARAMS_OPTIONS["risk_script"]).
"""

from .basic_risk         import run as basic_risk
from .vwap_tp_risk       import run as vwap_tp_risk
from .vwap_trailing_risk import run as vwap_trailing_risk

RISK_REGISTRY = {
    "basic_risk":         basic_risk,
    "vwap_tp_risk":       vwap_tp_risk,
    "vwap_trailing_risk": vwap_trailing_risk,
}
