"""ORB parameters."""

PARAMS = {
    "range_minutes":   15,     # opening-range length in minutes, counted from the 09:30 open
    "sl_factor":       0.5,    # stop distance = opening-range height x this
    "rr":              2.0,    # target distance = stop distance x this
    "timeout_minutes": 240,    # minutes after entry at which the timeout rule applies
}

PARAM_SECTIONS = {
    "Opening range": ["range_minutes"],
    "Risk":          ["sl_factor", "rr", "timeout_minutes"],
}
