"""Strategy parameters and output schema."""

PARAM_SECTIONS = {
    "General":                  ["ib_minutes", "trade_timeout", "max_flips", "valid_entries", "risk_script", "indicators_folder", "big_trades_folder"],
    "Entry Windows":            ["retest_window", "entry_window", "entry_after_absorption", "absorption_baseline_window"],
    "Entry Candle":             ["delta_threshold", "body_threshold"],
    "Absorption + Delta":       ["wick_threshold", "absorption_mult"],
    "Consecutive Absorption":   ["consec_abs_n", "consec_abs_mult", "consec_abs_ticks", "consec_wick_threshold"],
    "Two Bar Absorption":       ["two_bar_wick_ticks", "two_bar_abs_mult"],
    "Passive Absorption (Size Only)": ["passive_size_order_mult", "passive_size_absorption_mult", "passive_size_wick_threshold"],
    "Passive Wall":             ["passive_wall_n", "passive_wall_mult", "passive_wall_ticks"],
    "CVD Divergence (Absorption)": ["cvd_pivot_k", "cvd_min_separation", "cvd_max_separation", "cvd_wick_tolerance_ticks", "cvd_min_score"],
    "CVD Divergence (Exhaustion)": ["cvd_exh_pivot_k", "cvd_exh_min_separation", "cvd_exh_max_separation", "cvd_exh_wick_tolerance_ticks", "cvd_exh_min_score"],
    "Basic Risk Management":    ["rr", "sl_type"],
    "Zone SL Risk":             ["zone_rr"],
    "VWAP Risk":                ["sl_placement", "vwap_std", "vwap_session", "vwap_tp_mode"],
    "VWAP Trailing Risk":       ["trailing_entries", "trailing_in_profit", "late_trailing"]
}


PARAMS = {
    "ib_minutes":                   30,     # IB range duration: 15, 30, or 60
    "delta_threshold":              10.0,   # minimum volume_delta_pct for entry candle
    "body_threshold":               0.5,    # body must cover 50% of bar range
    "retest_window":                45,     # max bars to wait for retest after breakout
    "entry_window":                 25,     # bars to scan for entry after retest
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
    "two_bar_wick_ticks":           8,      # max wick size in ticks on defended side for both candles
    "two_bar_abs_mult":             2.0,    # absorption multiplier for merged 2-bar candle
    # --- passive order (size only) + absorption params ---
    "passive_size_order_mult":      2.0,    # raw resting size must be >= this x rolling baseline
    "passive_size_absorption_mult": 1.5,    # absorption mult for size-only passive finder
    "passive_size_wick_threshold":  0.4,    # wick threshold for size-only passive finder
    # --- passive wall params ---
    "passive_wall_n":               3,      # number of big passive orders required to form a wall
    "passive_wall_mult":            2.0,    # raw resting size >= this x rolling baseline to count as "big"
    "passive_wall_ticks":           8,      # ±ticks tolerance for clustering wall levels
    # --- shared dataset-folder names (same parquet/{type}/{asset}/, final folder swapped) ---
    "indicators_folder":            "ES_1m_indicators",     # indicators dataset (CVD + VWAP bands); empty = no indicators
    "big_trades_folder":            "ES_big_trades",        # big-trades dataset (declared for upcoming use)
    # --- cvd divergence (absorption) params ---
    "cvd_pivot_k":                  2,      # bars on the left required to qualify a pivot (fractal)
    "cvd_min_separation":           3,      # min bars between the two pivots
    "cvd_max_separation":           20,     # max bars between the two pivots (older pivot stale beyond this)
    "cvd_wick_tolerance_ticks":     2,      # tolerance (ticks) for lower/equal high (or higher/equal low)
    "cvd_min_score":                0.3,    # z-score threshold for the CVD divergence
    # --- cvd divergence (exhaustion) params — independent of the absorption finder's ---
    "cvd_exh_pivot_k":              2,      # bars on the left required to qualify a pivot (fractal)
    "cvd_exh_min_separation":       3,      # min bars between the two pivots
    "cvd_exh_max_separation":       20,     # max bars between the two pivots (older pivot stale beyond this)
    "cvd_exh_wick_tolerance_ticks": 2,      # tolerance (ticks) for higher/equal high (or lower/equal low)
    "cvd_exh_min_score":            0.3,    # z-score threshold for the CVD divergence
    # --- which entries to look for (1=on, 0=off): absorption_delta, consec, two_bar, passive_size_only, passive_wall, cvd_divergence_absorption, cvd_divergence_exhaustion ---
    "valid_entries":                "1111100",
    # --- which risk management script to use: 1 = basic_risk, 2 = zone_sl_risk, 3 = vwap_tp_risk, 4 = vwap_trailing_risk ---
    "risk_script":                  4,
    # --- basic risk management script ---
    "rr":                           1.0,    # fixed risk to reward ratio
    "sl_type":                      0,      # 0 = VAL, 1 = swing low
    # --- zone SL risk script ---
    "zone_rr":                      1.0,    # fixed risk to reward ratio for zone_sl_risk
    # --- vwap TP risk script ---
    "sl_placement":                 2,      # 1 = VAL/VAH stop, 2 = zone_sl_risk SL logic
    "vwap_std":                     2,      # which sigma band for TP (2 or 3)
    "vwap_session":                 "globex",  # vwap band session: "globex" or "rth"
    "vwap_tp_mode":                 "now",  # "now" (band frozen at entry) or "trailing" (live)
    # --- vwap trailing risk script (script 4 = vwap_tp_risk + signal-driven trailing stop) ---
    # which signals may trail the stop (1=on, 0=off), same bit order as valid_entries
    "trailing_entries":             "1111100",
    "trailing_in_profit":           1,      # 1 = only breakeven-or-better levels trail (in-loss signals not even logged); 0 = trail everything
    "late_trailing":                0,      # 1 = trail to the PREVIOUS logged signal's level (lag one signal); 0 = trail immediately
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
