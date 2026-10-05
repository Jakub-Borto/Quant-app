"""Parameters of delta_pct_signal (spec v1, section 3)."""

PARAMS = {
    "tick_size":          0.25,      # auto-filled from ASSET_INFO (read-only); unused in v1
    "delta_threshold":    40.0,      # theta: |volume_delta_pct| needed for a signal (0 < theta <= 100)
    "hold_bars":          5,         # bars a trade is held (time exit at the close of e + hold - 1)
    "direction_mode":     "follow",  # follow: buy-side -> long; fade: buy-side -> short
    "price_filter":       "any",     # any / with (candle agrees with the delta) / against
    "on_signal_in_trade": "ignore",  # ignore / close_and_reopen / extend_or_reverse
    "min_volume_mult":    1.0,       # volume >= mult x median of the previous bars; 0 = off
    "volume_lookback":    20,        # bars in that median (a PREPARE_PARAM; not meant to be swept)
    "entry_start":        "09:30",   # signal window start, NY, inclusive (>= 18:00 = previous evening)
    "entry_end":          "15:45",   # signal window end, NY, inclusive
    "flat_time":          "15:55",   # forced exit at the close of the last bar at/before this, NY
}

PARAM_SECTIONS = {
    "Signal":  ["delta_threshold", "direction_mode", "price_filter",
                "min_volume_mult", "volume_lookback"],
    "Trade":   ["hold_bars", "on_signal_in_trade"],
    "Session": ["tick_size", "entry_start", "entry_end", "flat_time"],
}

PARAMS_OPTIONS = {
    "direction_mode":     ["follow", "fade"],
    "price_filter":       ["any", "with", "against"],
    "on_signal_in_trade": ["ignore", "close_and_reopen", "extend_or_reverse"],
}
