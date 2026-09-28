"""Parameters of example_prev_day_levels."""

PARAMS = {
    "tick_size":        0.25,     # auto-filled from ASSET_INFO (read-only in the UI)
    "rth_start":        "09:30",  # RTH session start "HH:MM" NY (a PREPARE_PARAM)
    "signal_after":     "10:00",  # no signals before this bar (NY)
    "exit_time":        "15:55",  # flat at the close of the last bar at/before this (NY)
    "lookback_days":    5,        # days averaged for the typical RTH range
    "stop_range_frac":  0.25,     # stop distance = average RTH range x this
    "target_rr":        2.0,      # target distance = stop distance x this
    "require_vwap":     True,     # longs only above RTH VWAP, shorts only below
    "direction_filter": "both",   # "both", "long_only" or "short_only"
}

PARAM_SECTIONS = {
    "Session": ["tick_size", "rth_start", "signal_after", "exit_time"],
    "Levels":  ["lookback_days", "require_vwap", "direction_filter"],
    "Risk":    ["stop_range_frac", "target_rr"],
}

PARAMS_OPTIONS = {
    "direction_filter": ["both", "long_only", "short_only"],
}
