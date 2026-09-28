"""Risk management scripts: stop/target placement + trade fill simulation.

Each risk script is fully self-contained — it exposes
    run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params) -> trade dict | None
(entry_win = the post_retest EntryWindow, trade_win = the post_entry TradeWindow, entry_pos =
the absolute day position of the entry bar — see _daydata) and carries its own copy of the
fill-simulation helpers (no shared module, no cross-script imports). It also owns its params:
PARAMS (names prefixed with the script's own prefix, never shared with another script),
SECTION, OPTIONS, NEEDS_VWAP_BANDS and optionally extra_active_finders(params, n).

The `risk_script` param selects by name from RISK_SCRIPTS; dict order is the UI dropdown
order. To remove a script: delete its file and its line below — nothing else refers to it.
"""

from . import basic_risk
from . import vwap_tp_risk
from . import vwap_trailing_risk

RISK_SCRIPTS = {
    "basic_risk":         basic_risk,
    "vwap_tp_risk":       vwap_tp_risk,
    "vwap_trailing_risk": vwap_trailing_risk,
}

RISK_REGISTRY = {name: module.run for name, module in RISK_SCRIPTS.items()}
