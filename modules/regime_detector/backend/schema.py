"""
The regime-file contract: required columns, column families, detector-module
declarations, and validators for both.

Output schema (per daily YYYY-MM-DD.parquet):
- index: tz-aware DatetimeIndex, America/New_York — the snapshot timestamp.
  A snapshot stamped T uses only bars STRICTLY BEFORE T.
- always-present columns: ALWAYS_COLUMNS below.
- script columns: named by the detector, classified by COLUMN_TIERS
  (regime / score / diagnostic) and prefixed by input family:
      gbx_        full globex session, this session so far
      gbx_hist_   full globex session, completed prior days only
      rth_        RTH only, this session so far
      rth_hist_   RTH only, completed prior days only
  gbx_hist_* is constant within a file by construction (completed days
  can't change intraday); rth_hist_* may legitimately vary by snapshot
  (time-of-day-matched comparisons). Which one a given _hist_ column is
  gets recorded per column in meta.json.

State convention: regime states are plain strings (no pandas Categorical —
keeps the parquet round-trip boring). Every regime column may additionally
emit UNKNOWN_STATE for warm-up / missing inputs — never a silently wrong
number. 'unknown' is NOT declared in REGIME_STATES (it is not a real
regime); charts always render it in UNKNOWN_COLOR.
"""

# Columns the runner guarantees in every daily file.
ALWAYS_COLUMNS = ("is_final", "price", "contract", "n_bars_gbx",
                  "n_bars_rth", "lookback_days_used")

# Params every detector script must declare in PARAMS (defaults live in
# runner.SHARED_PARAMS; scripts spread them in via base.py so they never
# drift between scripts).
REQUIRED_PARAMS = ("rth_start", "rth_end", "snapshot_minutes", "lookback_days")

# Module-level names every detector script must declare.
REQUIRED_DECLARATIONS = ("PARAMS", "REGIME_STATES", "COLUMN_TIERS",
                         "SCRIPT_VERSION", "run_all")

# Column-tier keys (meta.json `schema` mirrors this split).
TIERS = ("regime", "score", "diagnostic")

# hist prefixes first — "gbx_hist_x" must not classify as plain "gbx_".
_FAMILIES = (("gbx_hist_", "gbx_hist"), ("rth_hist_", "rth_hist"),
             ("gbx_", "gbx"), ("rth_", "rth"))

UNKNOWN_STATE = "unknown"
UNKNOWN_COLOR = "#6a6f7a"


def column_family(name: str) -> str | None:
    """'gbx' | 'gbx_hist' | 'rth' | 'rth_hist' for a prefixed script column,
    None for anything else (including the always-present columns)."""
    for prefix, family in _FAMILIES:
        if name.startswith(prefix):
            return family
    return None


# ── detector-module validation ────────────────────────────────────────────────

def validate_detector(module) -> list[str]:
    """Contract errors for a loaded detector module ([] when clean). The UI
    runs this on selection so a broken script fails loudly before a run."""
    errors = []
    for name in REQUIRED_DECLARATIONS:
        if not hasattr(module, name):
            errors.append(f"missing module-level `{name}`")
    if errors:
        return errors

    if not callable(module.run_all):
        errors.append("`run_all` is not callable")
    if not isinstance(module.SCRIPT_VERSION, str) or not module.SCRIPT_VERSION:
        errors.append("`SCRIPT_VERSION` must be a non-empty string")

    params = module.PARAMS
    if not isinstance(params, dict):
        errors.append("`PARAMS` must be a dict")
    else:
        missing = [p for p in REQUIRED_PARAMS if p not in params]
        if missing:
            errors.append(f"PARAMS missing required key(s): {', '.join(missing)}")

    states = module.REGIME_STATES
    if not isinstance(states, dict) or not states:
        errors.append("`REGIME_STATES` must be a non-empty dict")
        states = {}
    for col, spec in states.items():
        if not isinstance(spec, dict) or "states" not in spec or "colors" not in spec:
            errors.append(f"REGIME_STATES[{col!r}] needs 'states' and 'colors'")
            continue
        if len(spec["states"]) != len(spec["colors"]) or not spec["states"]:
            errors.append(f"REGIME_STATES[{col!r}]: states/colors must be "
                          f"non-empty and the same length")
        if UNKNOWN_STATE in spec["states"]:
            errors.append(f"REGIME_STATES[{col!r}]: do not declare "
                          f"'{UNKNOWN_STATE}' — it is implicit, not a regime")

    tiers = module.COLUMN_TIERS
    if not isinstance(tiers, dict):
        errors.append("`COLUMN_TIERS` must be a dict")
    else:
        bad = [k for k in tiers if k not in TIERS]
        if bad:
            errors.append(f"COLUMN_TIERS has unknown tier(s): {', '.join(bad)}")
        regime_cols = list(tiers.get("regime", []))
        undeclared = [c for c in regime_cols if c not in states]
        if undeclared:
            errors.append("regime column(s) missing from REGIME_STATES: "
                          + ", ".join(undeclared))
        unstated = [c for c in states if c not in regime_cols]
        if unstated:
            errors.append("REGIME_STATES column(s) not in COLUMN_TIERS"
                          "['regime']: " + ", ".join(unstated))
    return errors


# ── day-frame validation ──────────────────────────────────────────────────────

def validate_day_frame(df, states: dict | None = None) -> list[str]:
    """Contract errors for one daily output frame ([] when clean). The runner
    calls this before writing, so a detector bug fails the run instead of
    silently persisting a malformed file."""
    import pandas as pd

    errors = []
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        errors.append("index must be a tz-aware DatetimeIndex")
    elif str(df.index.tz) != "America/New_York":
        errors.append(f"index tz must be America/New_York, got {df.index.tz}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        errors.append("index must be strictly increasing snapshot timestamps")

    missing = [c for c in ALWAYS_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing always-present column(s): {', '.join(missing)}")

    if "is_final" in df.columns and len(df):
        finals = df["is_final"].to_numpy().nonzero()[0]
        if len(finals) != 1 or finals[0] != len(df) - 1:
            errors.append("is_final must be True on exactly the last row")

    for col, spec in (states or {}).items():
        if col not in df.columns:
            errors.append(f"declared regime column {col!r} missing from frame")
            continue
        allowed = set(spec["states"]) | {UNKNOWN_STATE}
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            errors.append(f"{col!r} emits undeclared state(s): {sorted(bad)}")
    return errors
