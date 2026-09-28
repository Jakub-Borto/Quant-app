"""Strategy parameters, assembled.

Only the CORE params live here (session, IB, breakout/retest windows, the entry
candle and the shared absorption settings). Every entry finder (entries/*.py) and
every risk script (risk/*.py) declares its OWN params, section and dropdown
choices in its own file; they are collected below, so deleting a finder or a
risk script (its file + its registry line) removes its params from the UI too.

Risk-script params carry the script's prefix (basic_ / vwap_tp_ / trail_) and are
never shared between scripts: each script works without the others.
"""

from .entries import DEFAULT_FLAGS, FINDER_MODULES, FINDER_NAMES
from .risk import RISK_SCRIPTS

# ── core params ──────────────────────────────────────────────────────────────
CORE_PARAMS = {
    "tick_size":                    0.25,     # instrument tick (auto-filled from ASSET_INFO, read-only)
    "session_start":                "09:30",  # session start "HH:MM" NY time; IB, baselines and the
                                              # RTH slice anchor here (end stays 16:00)
    "ib_minutes":                   30,       # IB range duration: 15, 30, or 60
    "max_flips":                    4,        # max direction flips per day after invalidation
    "valid_entries":                DEFAULT_FLAGS,  # which entries to look for (one bit per finder,
                                                    # FINDER_NAMES order; UI shows named checkboxes)
    "risk_script":                  "vwap_trailing_risk",  # which risk script manages the trade
    "retest_window":                45,       # max bars to wait for retest after breakout
    "entry_window":                 25,       # bars to scan for entry after retest
    "entry_after_absorption":       5,        # max bars to scan for entry candle after absorption
    "absorption_baseline_window":   20,       # rolling N bars for baseline (shared across all entries)
    "delta_threshold":              10.0,     # minimum volume_delta_pct for entry candle
    "body_threshold":               0.5,      # body must cover 50% of bar range
}

CORE_SECTIONS = {
    "General":       ["tick_size", "session_start", "ib_minutes", "max_flips",
                      "valid_entries", "risk_script"],
    "Entry Windows": ["retest_window", "entry_window", "entry_after_absorption",
                      "absorption_baseline_window"],
    "Entry Candle":  ["delta_threshold", "body_threshold"],
}

CORE_OPTIONS = {
    "valid_entries": list(FINDER_NAMES),
    "risk_script":   list(RISK_SCRIPTS),
}


def _assemble():
    params = dict(CORE_PARAMS)
    sections = dict(CORE_SECTIONS)
    options = dict(CORE_OPTIONS)
    owners = {k: "core" for k in params}
    parts = [(m.__name__.rsplit(".", 1)[-1], m) for m in FINDER_MODULES]
    parts += list(RISK_SCRIPTS.items())
    for owner, mod in parts:
        for key, value in mod.PARAMS.items():
            if key in owners:
                raise ValueError(f"ivb_model: param '{key}' of {owner} is already "
                                 f"declared by {owners[key]} — param names must be unique")
            owners[key] = owner
            params[key] = value
        sections[mod.SECTION] = list(mod.PARAMS)
        options.update(getattr(mod, "OPTIONS", {}) or {})
    return params, sections, options


PARAMS, PARAM_SECTIONS, PARAMS_OPTIONS = _assemble()
