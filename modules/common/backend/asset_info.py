"""
THE single ASSET_INFO table + filename-based asset lookups.

Before the rebuild this dict was duplicated in four view files (backtester /
optimizer with the base keys; analytics / monte_carlo with commissions and
micro links added). The four copies were verified identical on the shared
keys, so this is the analytics superset copy, verbatim:

  tick_size                 minimum price increment
  ticks_per_point           ticks in one full point (pnl_points * this = ticks)
  dollars_per_tick          $ value of one tick for one contract
  commissions_per_contract  round-turn/2 commission (per side), where known
  parent                    for micro contracts: the full-size ticker

HIDDEN_PARAMS lives here too: strategy params that are auto-injected from
ASSET_INFO and therefore never get a UI widget.

(A 5th partial ASSET_INFO copy exists inside data_transforms/
1m_advanced_indicators.py — that one is a self-contained plugin and is
deliberately left alone.)
"""

HIDDEN_PARAMS = {"tick_size"}

ASSET_INFO = {
    # Equity Index
    "ES":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 2.88},
    "NQ":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "RTY": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "YM":  {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "MES": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 1.25,    "commissions_per_contract": 0.95, "parent": "ES"},
    "MNQ": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "NQ"},
    "M2K": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "RTY"},
    "MYM": {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "YM"},

    # Rates
    "ZN":  {"tick_size": 0.015625, "ticks_per_point": 64,  "dollars_per_tick": 15.625,  "commissions_per_contract": 2.30},  # 1/64
    "ZB":  {"tick_size": 0.03125,  "ticks_per_point": 32,  "dollars_per_tick": 31.25,   "commissions_per_contract": 2.37},  # 1/32
    "ZF":  {"tick_size": 0.0078125,"ticks_per_point": 128, "dollars_per_tick": 7.8125,  "commissions_per_contract": 2.15},  # 1/128
    "ZT":  {"tick_size": 0.00390625,"ticks_per_point": 256, "dollars_per_tick": 7.8125, "commissions_per_contract": 2.15},  # 1/128 — verify, ZT is quoted in 1/256 in some venues
    "SR3": {"tick_size": 0.0025,   "ticks_per_point": 400, "dollars_per_tick": 6.25,    "commissions_per_contract": 2.10},  # commision

    # Energy
    "CL":  {"tick_size": 0.01, "ticks_per_point": 100, "dollars_per_tick": 10.00,   "commissions_per_contract": 3.00},
    "QM":  {"tick_size": 0.025,"ticks_per_point": 40,  "dollars_per_tick": 12.50,   "commissions_per_contract": 2.70},
    "NG":  {"tick_size": 0.001,"ticks_per_point": 1000,"dollars_per_tick": 10.00,   "commissions_per_contract": 3.10},
    "RB":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20,  "commissions_per_contract": 3.00},  # ~4.20 at 42000 gal contract — price-dependent, verify
    "HO":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20,  "commissions_per_contract": 3.00},  # same as RB

    # Metals
    "GC":  {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 10.00,   "commissions_per_contract": 3.10},
    "MGC": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 1.00,    "commissions_per_contract": 1.20, "parent": "GC"},
    "SI":  {"tick_size": 0.005,"ticks_per_point": 200, "dollars_per_tick": 25.00,   "commissions_per_contract": 3.10},
    "HG":  {"tick_size": 0.0005,"ticks_per_point": 2000,"dollars_per_tick": 12.50,  "commissions_per_contract": 3.10},

    # Grains
    "ZC":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},
    "ZS":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},
    "ZW":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},

    # FX
    "6E":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 6.25, "commissions_per_contract": 3.10},
    "6J":  {"tick_size": 0.0000005,"ticks_per_point": 2000000,"dollars_per_tick": 6.25, "commissions_per_contract": 3.10},
    "6B":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 6.25,  "commissions_per_contract": 3.10},
    "6C":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 5.00, "commissions_per_contract": 3.10},

    # Crypto
    "BTC": {"tick_size": 5.00, "ticks_per_point": 0.2, "dollars_per_tick": 25.00,   "commissions_per_contract": 8.00},
}


def get_dollars_per_tick(trades_filename: str) -> float:
    """Asset is the FIRST underscore token of the trades filename (contract
    between the backtester's save name and the analytics/MC readers)."""
    asset = trades_filename.split("_")[0]
    if asset not in ASSET_INFO:
        raise ValueError(f"Unknown asset '{asset}' derived from filename '{trades_filename}'. Add it to ASSET_INFO.")
    return ASSET_INFO[asset]["dollars_per_tick"]


def _micro_child(asset: str) -> str | None:
    """Return the micro ticker whose `parent` is `asset`, or None. An asset is
    'microable' iff such a child exists — that's the only decomposition flag."""
    for ticker, info in ASSET_INFO.items():
        if info.get("parent") == asset:
            return ticker
    return None


def get_commission_info(trades_filename: str) -> tuple[float | None, float | None]:
    """
    (full_commission, micro_commission) for the file's asset, mirroring
    get_dollars_per_tick. `full` is None if the asset has no
    commissions_per_contract key (graceful degradation → caller bills 0 +
    warns). `micro` is the child's commission when a micro child exists, else
    None (non-microable).
    """
    asset = trades_filename.split("_")[0]
    info  = ASSET_INFO.get(asset, {})
    full  = info.get("commissions_per_contract")

    child = _micro_child(asset)
    micro = ASSET_INFO[child].get("commissions_per_contract") if child else None
    return full, micro
