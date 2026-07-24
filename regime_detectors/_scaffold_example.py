"""
Scaffold detector — proves the pipeline end to end. Deliberately NOT a real
detector; real detector scripts are a separate design conversation.

Labels every snapshot by the SIGN of the session-so-far return: last close
strictly before the snapshot vs the globex session open. Even a 0.01% move
counts — 'positive' for any gain, 'negative' for any loss, 'flat' only for
exactly zero. One historical diagnostic (`gbx_hist_up_ratio`, the fraction
of up days in the lookback window) exercises the completed-days column
family, which must be constant within each file.
"""

from modules.regime_detector.backend.runner import (RegimeContext,
                                                    SHARED_PARAMS)
from modules.regime_detector.backend.schema import UNKNOWN_STATE

SCRIPT_VERSION = "1.0"

PARAMS = {**SHARED_PARAMS}

REGIME_STATES = {
    "gbx_ret_sign": {"states": ["positive", "flat", "negative"],
                     "colors": ["#2ecc71", "#f1c40f", "#e74c3c"]},
}

COLUMN_TIERS = {
    "regime": ["gbx_ret_sign"],
    "score": ["gbx_ret_pct"],
    "diagnostic": ["gbx_hist_up_ratio"],
}


def _summarize(bars):
    """One completed day boiled down for the lookback window."""
    if bars.empty:
        return {"ret_pct": float("nan")}
    o = float(bars["open"].iloc[0])
    c = float(bars["close"].iloc[-1])
    return {"ret_pct": (c / o - 1.0) * 100.0 if o else float("nan")}


def run_all(input_folder, output_folder, skip_existing, on_progress, params):
    ctx = RegimeContext(input_folder, output_folder, params=params,
                        script_name="_scaffold_example",
                        script_version=SCRIPT_VERSION,
                        script_file=__file__,
                        states=REGIME_STATES, tiers=COLUMN_TIERS,
                        summarize=_summarize, skip_existing=skip_existing,
                        on_progress=on_progress)
    for day in ctx.days():
        window = ctx.lookback(day)
        if ctx.should_skip(day):
            continue
        bars = ctx.bars(day)

        ups = [s for s in window.values() if s["ret_pct"] == s["ret_pct"]]
        up_ratio = (sum(1 for s in ups if s["ret_pct"] > 0) / len(ups)) \
            if ups else float("nan")

        rows = []
        for ts in ctx.snapshot_times(day):
            upto = ctx.slice(bars, before=ts)
            if upto.empty:
                rows.append({"ts": ts, "gbx_ret_sign": UNKNOWN_STATE,
                             "gbx_ret_pct": float("nan"),
                             "gbx_hist_up_ratio": up_ratio})
                continue
            session_open = float(upto["open"].iloc[0])
            last_close = float(upto["close"].iloc[-1])
            ret = (last_close / session_open - 1.0) * 100.0 if session_open \
                else float("nan")
            if ret != ret:
                sign = UNKNOWN_STATE
            elif ret > 0:
                sign = "positive"
            elif ret < 0:
                sign = "negative"
            else:
                sign = "flat"
            rows.append({"ts": ts, "gbx_ret_sign": sign, "gbx_ret_pct": ret,
                         "gbx_hist_up_ratio": up_ratio})
        ctx.write(day, rows)
