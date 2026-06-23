"""Strategy parameters and output schema."""

PARAM_SECTIONS = {
    "General":                  ["ib_minutes", "trade_timeout", "max_flips", "valid_entries", "risk_script"],
    "Entry Windows":            ["retest_window", "entry_window", "entry_after_absorption", "absorption_baseline_window"],
    "Entry Candle":             ["delta_threshold", "body_threshold"],
    "Absorption + Delta":       ["wick_threshold", "absorption_mult"],
    "Consecutive Absorption":   ["consec_abs_n", "consec_abs_mult", "consec_abs_ticks", "consec_wick_threshold"],
    "Two Bar Absorption":       ["two_bar_wick_ticks", "two_bar_abs_mult"],
    "Passive Absorption":       ["passive_order_mult", "passive_absorption_mult", "passive_wick_threshold"],
    "Basic Risk Management":    ["rr", "sl_type"]
}


PARAMS = {
    "ib_minutes":                   30,     # IB range duration: 15, 30, or 60
    "delta_threshold":              30.0,   # minimum volume_delta_pct for entry candle
    "body_threshold":               0.5,    # body must cover 50% of bar range
    "retest_window":                30,     # max bars to wait for retest after breakout
    "entry_window":                 15,     # bars to scan for entry after retest
    "entry_after_absorption":       5,      # max bars to scan for entry candle after absorption
    "absorption_baseline_window":   20,     # rolling N bars for baseline (shared across all entries)
    "trade_timeout":                999,    # bars before timeout logic kicks in
    "max_flips":                    4,      # max direction flips per day after invalidation
    # --- absorption + delta params ---
    "wick_threshold":               0.4,    # lower wick must be >= this fraction of total bar range
    "absorption_mult":              2.0,    # wick level volume must be >= this x rolling avg
    "tick_size":                    0.25,   # ES tick size
    # --- consecutive absorption params ---
    "consec_abs_n":                 2,      # number of absorption candles required at same level
    "consec_abs_mult":              2.0,    # absorption multiplier
    "consec_abs_ticks":             4,      # ±ticks tolerance for grouping absorption levels
    "consec_wick_threshold":        0.4,    # wick threshold independent of absorption + delta
    # --- two bar absorption params ---
    "two_bar_wick_ticks":           3,      # max wick size in ticks on defended side for both candles
    "two_bar_abs_mult":             2.0,    # absorption multiplier for merged 2-bar candle
    # --- passive order + absorption params ---
    "passive_order_mult":           3.0,    # passive avg order size must be >= this x rolling baseline
    "passive_absorption_mult":      1.5,    # absorption mult for passive+absorption finder
    "passive_wick_threshold":       0.4,    # wick threshold independent of absorption + delta
    # --- which entries to look for (1=on, 0=off): absorption_delta, consec, two_bar, passive ---
    "valid_entries":                "1111",
    # --- which riska management script to use ---
    "risk_script":                  1,
    # --- basic risk management script ---
    "rr":                           1.0,    # fixed risk to reward ratio
    "sl_type":                      0,      # 0 = VAL, 1 = swing low
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
