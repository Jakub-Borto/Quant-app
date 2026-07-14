"""Parameters, validation and output schema for the VWAP trend strategy."""

PARAMS = {
    # --- data ---
    "indicators_dataset":  "ES_1m_indicators",  # sibling dataset with vwap_bar_* columns (required)
    # --- vwap ---
    "vwap_anchor":         "rth",            # 'rth' | 'globex' -> vwap_bar_{anchor}
    "vwap_band_ticks":     0.0,              # neutral band half-width in TICKS (0 = paper rule)
    "band_rule":           "carry_forward",  # 'carry_forward' (hold through band) | 'flat' (stand aside)
    # --- session / window (NY wall clock "HH:MM") ---
    "trade_start_time":    "09:31",          # first bar eligible to OPEN a position (signal = prior close)
    "trade_end_time":      "16:00",          # forced flat at the close of the last bar at/before this
    "exclude_start":       "",               # optional midday exclusion, e.g. "12:00" ("" = off)
    "exclude_end":         "",               # e.g. "15:00" ("" = off; must be set together)
    # --- data hygiene ---
    "skip_zero_volume":    1,                # 1 = forward-filled zero-volume bars produce no signal
                                             # (int 0/1, not bool, so the optimizer can sweep it)
    # --- risk-reference convention (sl/tp columns; see engine._make_trade) ---
    "sl_convention":       "vwap_at_entry",  # 'vwap_at_entry' | 'realized_exit' | 'none'
    # --- injected by the backtester from ASSET_INFO (HIDDEN_PARAMS) ---
    "tick_size":           0.25,
}

PARAM_SECTIONS = {
    "Data":           ["indicators_dataset"],
    "VWAP":           ["vwap_anchor", "vwap_band_ticks", "band_rule"],
    "Trading Window": ["trade_start_time", "trade_end_time", "exclude_start", "exclude_end"],
    "Data Hygiene":   ["skip_zero_volume"],
    "Risk Reference": ["sl_convention"],
}

# ADVISORY ONLY — recommended optimizer sweep ranges. The platform's optimizer
# does NOT read this dict: it infers sweepability from each PARAMS default's
# type (int/float -> min/max/step range, str -> comma-separated value list) and
# the user picks the ranges in the UI. This documents sensible choices.
PARAM_SPACE = {
    "vwap_band_ticks":  {"type": "float", "min": 0.0, "max": 8.0, "step": 0.5},
    "band_rule":        {"type": "categorical", "values": ["carry_forward", "flat"]},
    "vwap_anchor":      {"type": "categorical", "values": ["rth", "globex"]},
    "trade_start_time": {"type": "categorical", "values": ["09:31", "09:45", "10:00", "10:30"]},
    "trade_end_time":   {"type": "categorical", "values": ["15:00", "15:45", "16:00"]},
    "skip_zero_volume": {"type": "int", "min": 0, "max": 1, "step": 1},
}

OUTPUT_COLUMNS = [
    "date",
    "direction",
    "trade_type",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "exit_reason",
    "pnl_points",
    "notes",
]

VWAP_ANCHORS    = ("rth", "globex")
BAND_RULES      = ("carry_forward", "flat")
SL_CONVENTIONS  = ("vwap_at_entry", "realized_exit", "none")


def _parse_hhmm(value: str, name: str) -> int:
    """'HH:MM' -> minutes since midnight, with an actionable error."""
    try:
        hh, mm = str(value).strip().split(":")
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
    except Exception:
        raise ValueError(
            f"vwap_trend: param '{name}' must be a 'HH:MM' NY wall-clock time, got {value!r}"
        ) from None
    return hh * 60 + mm


def validate(params: dict) -> dict:
    """Merged params -> normalized config dict. Raises ValueError on bad input."""
    p = {**PARAMS, **(params or {})}

    if p["vwap_anchor"] not in VWAP_ANCHORS:
        raise ValueError(f"vwap_trend: vwap_anchor must be one of {VWAP_ANCHORS}, got {p['vwap_anchor']!r}")
    if p["band_rule"] not in BAND_RULES:
        raise ValueError(f"vwap_trend: band_rule must be one of {BAND_RULES}, got {p['band_rule']!r}")
    if p["sl_convention"] not in SL_CONVENTIONS:
        raise ValueError(f"vwap_trend: sl_convention must be one of {SL_CONVENTIONS}, got {p['sl_convention']!r}")

    band_ticks = float(p["vwap_band_ticks"])
    if band_ticks < 0:
        raise ValueError(f"vwap_trend: vwap_band_ticks must be >= 0, got {band_ticks}")
    if not str(p["indicators_dataset"]).strip():
        raise ValueError(
            "vwap_trend: 'indicators_dataset' is required — the sibling dataset holding "
            "the vwap_bar_rth / vwap_bar_globex columns (e.g. 'ES_1m_indicators')"
        )

    start_min = _parse_hhmm(p["trade_start_time"], "trade_start_time")
    end_min   = _parse_hhmm(p["trade_end_time"],   "trade_end_time")
    if start_min >= end_min:
        raise ValueError(
            f"vwap_trend: trade_start_time ({p['trade_start_time']}) must be before "
            f"trade_end_time ({p['trade_end_time']})"
        )

    excl_start = str(p["exclude_start"] or "").strip()
    excl_end   = str(p["exclude_end"]   or "").strip()
    if bool(excl_start) != bool(excl_end):
        raise ValueError("vwap_trend: exclude_start and exclude_end must be set together (or both empty)")
    exclusion = None
    if excl_start:
        e0 = _parse_hhmm(excl_start, "exclude_start")
        e1 = _parse_hhmm(excl_end,   "exclude_end")
        if e0 >= e1:
            raise ValueError(f"vwap_trend: exclusion window inverted ({excl_start} >= {excl_end})")
        if e0 < start_min or e1 > end_min:
            raise ValueError(
                f"vwap_trend: exclusion window [{excl_start}, {excl_end}) must lie inside the "
                f"trading window [{p['trade_start_time']}, {p['trade_end_time']}]"
            )
        exclusion = (excl_start, excl_end)

    tick_size = float(p["tick_size"])
    return {
        "indicators_dataset": str(p["indicators_dataset"]).strip(),
        "anchor":             p["vwap_anchor"],
        "anchor_col":         f"vwap_bar_{p['vwap_anchor']}",
        "band_ticks":         band_ticks,
        "band_points":        band_ticks * tick_size,
        "carry_forward":      p["band_rule"] == "carry_forward",
        "trade_start":        str(p["trade_start_time"]).strip(),
        "trade_end":          str(p["trade_end_time"]).strip(),
        "start_min":          start_min,
        "exclusion":          exclusion,
        "skip_zero_volume":   bool(int(p["skip_zero_volume"])),
        "sl_convention":      p["sl_convention"],
    }
