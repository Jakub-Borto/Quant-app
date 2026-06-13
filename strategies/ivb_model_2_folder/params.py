"""Strategy parameters and output schema."""

# FOR BACKTESTER TO SHOW IT NICELY
PARAM_SECTIONS = {
    "General":                  ["ib_minutes", "rr", "sl_type", "trade_timeout", "max_flips", "valid_entries"],
    "Entry Windows":            ["retest_window", "entry_window", "entry_after_absorption"],
    "Candle Filters":           ["delta_threshold", "body_threshold"],
    "Absorption":               ["wick_threshold", "absorption_mult", "absorption_window"],
    "Consecutive Absorption":   ["consec_abs_n", "consec_abs_mult", "consec_abs_ticks"],
    "Two Bar Absorption":       ["two_bar_wick_ticks", "two_bar_abs_mult"],
    "Passive Absorption":       ["passive_order_mult", "passive_absorption_mult"],
}


PARAMS = {
    "ib_minutes":               30,     # IB range duration: 15, 30, or 60
    "delta_threshold":          30.0,   # minimum volume_delta_pct for entry candle
    "body_threshold":           0.5,    # body must cover 50% of bar range
    "rr":                       1.0,    # fixed risk to reward ratio
    "sl_type":                  0,      # 0 = VAL, 1 = swing low
    "retest_window":            30,     # max bars to wait for retest after breakout
    "entry_window":             15,     # bars to scan for entry after retest
    "entry_after_absorption":   5,      # max bars to scan for entry candle after absorption
    "trade_timeout":            999,    # bars before timeout logic kicks in
    "max_flips":                4,      # max direction flips per day after invalidation
    # --- absorption params ---
    "wick_threshold":           0.4,    # lower wick must be >= this fraction of total bar range
    "absorption_mult":          2.0,    # wick level sell volume must be >= this x rolling avg
    "absorption_window":        20,     # rolling N bars for avg sell_per_tick baseline (RTH, post 09:35)
    "tick_size":                0.25,   # ES tick size
    # --- consecutive absorption params ---
    "consec_abs_n":             2,      # number of absorption candles required at same level
    "consec_abs_mult":          2.0,    # absorption multiplier for consecutive absorption finder
    "consec_abs_ticks":         4,      # ±ticks tolerance for grouping absorption levels
    # --- two bar absorption params ---
    "two_bar_wick_ticks":       3,      # max wick size in ticks on defended side for both candles
    "two_bar_abs_mult":         2.0,    # absorption multiplier for merged 2-bar candle
    # --- passive order + absorption params ---
    "passive_order_mult":       3.0,    # passive avg order size must be >= this x rolling baseline
    "passive_absorption_mult":  1.5,    # absorption mult for passive+absorption finder
    # --- which entries to look for (1=on, 0=off): pure, consec, two_bar, passive ---
    "valid_entries":            "1111",
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
