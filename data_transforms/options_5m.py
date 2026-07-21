"""
options_5m — Transform B: intraday 5-minute option candles (ES).

Reads GLBX.MDP3 daily DEFINITION + STATISTICS + TBBO .dbn.zst files and writes
ONE PARQUET PER SESSION (YYYY-MM-DD.parquet): per contract per 5-minute bucket
over the FULL Globex session (18:00 NY prev day -> 17:00 NY), carrying option
OHLCV, aggressor-split volume, bucket-close quotes, point-in-time statistics
OI (evening -> morning regime), signed cumulative dealer flow, and the
front-month significance flag.  No IV / Greeks math — clean inputs only.
(No spot column: derive spot from the futures 1m candles downstream.)

SPARSE OUTPUT: a row exists ONLY where something happened in that bucket —
a trade (volume > 0, fresh quotes) or a CME OI print landing (volume 0,
OHLC NaN, carried quote pair, is_stale_quote=True).  OI prints published
before the session start (Sunday republications) appear at the first bucket.
One row per (contract, bucket).  When a contract expires, ONE tombstone row
(open_interest = -1, volume 0, final dealer_oi_flow) is written at the bucket
containing its real expiry + 1 second — the guaranteed last mention, like
transform A; only contracts that ever appeared in the output get one.

SPREAD-LEG ATTRIBUTION (needs a DBN v3 DEFINITION sibling folder, auto-
detected): trades on strategy instruments (verticals, UDS combos — invisible
to the outright tape) are decomposed into their option legs via the v3 leg
fields; each leg's outright receives size × ratio signed by trade-aggressor ×
leg-side.  Per bucket this fills spread_volume / spread_buy_volume /
spread_sell_volume and the cumulative dealer_oi_flow_spread (same activation
/ NaN rules as dealer_oi_flow, which stays outright-only); the attributed
net also feeds daily_estimated_oi.  Spreads never appear as rows — their
flow lands on the outright legs.  Without a v3 folder the spread columns are
zero and a note is emitted.

Row columns: timestamp (index, non-unique, tz America/New_York),
instrument_id, open/high/low/close, volume, buy_volume, sell_volume,
spread_volume, spread_buy_volume, spread_sell_volume, bid, ask,
is_stale_quote, open_interest, is_morning_update, daily_estimated_oi,
dealer_oi_flow, dealer_oi_flow_spread,
significant_options_based_on_front_month.
daily_estimated_oi = the latest CME-published OI adjusted per bucket by the
tape's net aggressor volume (buy adds, sell subtracts, mid 0), snapping to
CME's number at each new print.  The published value is as-of the PRIOR
session's close, so the adjustment window starts there — before this
session's print lands, the previous session's net is included so the snap
neither loses nor double-counts the overnight tape.  Can go negative (the
estimation error is left visible); tombstone rows carry -1.  Static contract facts are NOT
columns: each file's parquet metadata carries a JSON dict
{instrument_id: {underlying, series, expiry, strike, cp_flag, activation_date}}
plus scalar "multiplier".

Semantics:
- Aggressor per trade: price >= ask -> buyer-aggressed (dealer flow -size);
  price <= bid -> seller-aggressed (+size); strictly between -> mid (0).
  When bid == ask (locked, both true) the TBBO side field breaks the tie.
- Quotes are PAIR-ATOMIC: bid and ask always come from the same TBBO record
  (a bucket's last trade, or the contract's carried last record) — a fresh
  bid is never mixed with an older ask, so crossed pairs can only be genuine
  crossed/locked market records (~0.004% of the tape).
- dealer_oi_flow = cumulative signed size per contract from its ACTIVATION
  through the bucket (= running sell_volume - buy_volume).  NaN whenever the
  tape does not reach the contract's activation (never guess a start value).
  Accumulation stops at the contract's REAL (holiday-refined) expiry; no rows
  after expiry.
- open_interest is as-of publication: the previous session's rolling value
  until the evening print's ts, then the evening value, then the morning
  value once its print is published (is_morning_update flips at that ts).
- Buckets are 300s of absolute time anchored at the session start (18:00 NY);
  the bucket start is the row timestamp.
- PARAMS.drop_insignificant=True removes significant == False rows and omits
  the (then all-True) column, matching transform A.

Dealer-flow state is cross-day, so the output folder carries a sidecar
`_flow_state.parquet` (per-contract cumulative flow + last quote + last OI,
tagged with the last folded session).  skip_existing=True resumes from it;
if outputs exist but the sidecar is missing/stale, the tape is re-decoded
from the start to rebuild state (existing files are not rewritten).

FRONT-MONTH SIGNIFICANCE — MIRRORS TRANSFORM A EXACTLY (options_eod_chain.py):
same ASSETS_TO_ALWAYS_TRACK / ROLL_RULES dicts, same per-session Rules 1/2/3,
same ku mapping (month-code + own expiry, never the symbol's year digit), same
real-expiry refinement (holiday shifts learned from definitions), same
volume_roll source (candle front_month footers).  No shared module by design:
if either copy changes, sync the other manually.
"""



"""
WHAT TO ADD/FIX:
- bbo 5m
- resolve issue with dealers oi flow not following hidden oi
"""

import builtins
import json
import os
import runpy
import struct
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

TIMING = True

# ── front-month significance classification ───────────────────────────────────
# KEEP IN SYNC WITH TRANSFORM A (options_eod_chain.py) — no shared module by
# request; A and B must produce identical significance for the same session.
ASSETS_TO_ALWAYS_TRACK = {
    "ES": ("ES", "EW", "EYC"),
}

ROLL_RULES = {
    "ES": {"quarter_months": (3, 6, 9, 12), "expiry_weekday": 4,
           "expiry_week": 3, "expiry_time": (9, 30), "roll_offset_days": 8},
}
_DEFAULT_ROLL_RULE = ROLL_RULES["ES"]
_MONTHCHAR = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
              7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

# UI param: default keeps every row (research-safe).  True drops rows whose
# significant_options_based_on_front_month is False and omits the column.
PARAMS = {"drop_insignificant": False}

_NY = "America/New_York"
_NS_PER_DAY = 86_400_000_000_000
_NS_5MIN = 300_000_000_000
_NS_1S = 1_000_000_000
_I64_MAX = np.iinfo(np.int64).max
_TS_VALID_LO = pd.Timestamp("2005-01-01").value
_TS_VALID_HI = pd.Timestamp("2200-01-01").value

_SIG_COL = "significant_options_based_on_front_month"
_STATE_FILE = "_flow_state.parquet"


def _tlog(msg: str) -> None:
    if TIMING:
        print(f"[TIMING] {msg}", flush=True)


# ── fast DBN decode (same approach as transform A) ────────────────────────────

_DBN_V1_STAT_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"),
    ("instrument_id", "<u4"), ("ts_event", "<u8"), ("ts_recv", "<u8"),
    ("ts_ref", "<u8"), ("price", "<i8"), ("quantity", "<i4"),
    ("sequence", "<u4"), ("ts_in_delta", "<i4"), ("stat_type", "<u2"),
    ("channel_id", "<u2"), ("update_action", "u1"), ("stat_flags", "u1"),
    ("_reserved", "S6"),
])

_DBN_V1_TBBO_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"),
    ("instrument_id", "<u4"), ("ts_event", "<u8"), ("price", "<i8"),
    ("size", "<u4"), ("action", "S1"), ("side", "S1"), ("flags", "u1"),
    ("depth", "u1"), ("ts_recv", "<u8"), ("ts_in_delta", "<i4"),
    ("sequence", "<u4"), ("bid_px_00", "<i8"), ("ask_px_00", "<i8"),
    ("bid_sz_00", "<u4"), ("ask_sz_00", "<u4"), ("bid_ct_00", "<u4"),
    ("ask_ct_00", "<u4"),
])

_DBN_V1_DEF_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"),
    ("instrument_id", "<u4"), ("ts_event", "<u8"), ("ts_recv", "<u8"),
    ("min_price_increment", "<i8"), ("display_factor", "<i8"),
    ("expiration", "<u8"), ("activation", "<u8"),
    ("high_limit_price", "<i8"), ("low_limit_price", "<i8"),
    ("max_price_variation", "<i8"), ("trading_reference_price", "<i8"),
    ("unit_of_measure_qty", "<i8"), ("min_price_increment_amount", "<i8"),
    ("price_ratio", "<i8"), ("inst_attrib_value", "<i4"),
    ("underlying_id", "<u4"), ("raw_instrument_id", "<u4"),
    ("market_depth_implied", "<i4"), ("market_depth", "<i4"),
    ("market_segment_id", "<u4"), ("max_trade_vol", "<u4"),
    ("min_lot_size", "<i4"), ("min_lot_size_block", "<i4"),
    ("min_lot_size_round_lot", "<i4"), ("min_trade_vol", "<u4"),
    ("_reserved2", "S4"), ("contract_multiplier", "<i4"),
    ("decay_quantity", "<i4"), ("original_contract_size", "<i4"),
    ("_reserved3", "S4"), ("trading_reference_date", "<u2"),
    ("appl_id", "<i2"), ("maturity_year", "<u2"), ("decay_start_date", "<u2"),
    ("channel_id", "<u2"), ("currency", "S4"), ("settl_currency", "S4"),
    ("secsubtype", "S6"), ("raw_symbol", "S22"), ("group", "S21"),
    ("exchange", "S5"), ("asset", "S7"), ("cfi", "S7"),
    ("security_type", "S7"), ("unit_of_measure", "S31"), ("underlying", "S21"),
    ("strike_price_currency", "S4"), ("instrument_class", "S1"),
    ("_reserved4", "S2"), ("strike_price", "<i8"), ("_reserved5", "S6"),
    ("match_algorithm", "S1"), ("md_security_trading_status", "u1"),
    ("main_fraction", "u1"), ("price_display_format", "u1"),
    ("settl_price_type", "u1"), ("sub_fraction", "u1"),
    ("underlying_product", "u1"), ("security_update_action", "S1"),
    ("maturity_month", "u1"), ("maturity_day", "u1"), ("maturity_week", "u1"),
    ("user_defined_instrument", "S1"), ("contract_multiplier_unit", "i1"),
    ("flow_schedule_type", "i1"), ("tick_rule", "u1"), ("_dummy", "S3"),
])

_SCHEMA_DTYPES = {"statistics": _DBN_V1_STAT_DTYPE,
                  "definition": _DBN_V1_DEF_DTYPE,
                  "tbbo": _DBN_V1_TBBO_DTYPE}


def _decode_dbn(path: Path, schema: str) -> np.ndarray:
    dt = _SCHEMA_DTYPES[schema]
    try:
        with open(path, "rb") as fh:
            dec = zstandard.ZstdDecompressor().stream_reader(
                fh, read_across_frames=True).read()
        if dec[:3] == b"DBN" and dec[3] == 1:
            (mlen,) = struct.unpack_from("<I", dec, 4)
            body_off = 8 + mlen
            if (len(dec) - body_off) % dt.itemsize == 0:
                arr = np.frombuffer(dec, dtype=dt, offset=body_off)
                if len(arr) == 0 or (arr["length"].astype(np.int64) * 4
                                     == dt.itemsize).all():
                    return arr
    except Exception:
        pass
    import databento as db  # fallback only
    return db.DBNStore.from_file(str(path)).to_ndarray()


# ── folder / file resolution ──────────────────────────────────────────────────

def _schema_token(folder: Path) -> str | None:
    for f in folder.glob("*.dbn.zst"):
        parts = f.name.split(".")
        if len(parts) >= 3:
            return parts[-3].lower()
        return None
    return None


def _resolve_folders(input_folder: Path) -> dict:
    """Input may be any of the DEFINITION / STATISTICS / TBBO folders —
    resolve all of them from the siblings.  Definition folders are split by
    DBN encoding: v1 (biggest) supplies the outright statics, a v3 one (if
    present) supplies strategy LEGS for spread attribution ('legs', optional).
    """
    token = _schema_token(input_folder)
    if token not in ("definition", "statistics", "tbbo"):
        raise ValueError(f"{input_folder} does not contain "
                         f".definition/.statistics/.tbbo .dbn.zst files")
    folders = [input_folder] + [s for s in input_folder.parent.iterdir()
                                if s.is_dir() and s != input_folder]
    by_token: dict[str, list] = {}
    for f in folders:
        tok = _schema_token(f)
        if tok:
            by_token.setdefault(tok, []).append(f)
    out: dict = {"legs": None}
    for want in ("statistics", "tbbo"):
        cands = by_token.get(want, [])
        if not cands:
            raise FileNotFoundError(
                f"No {want.upper()} folder found next to {input_folder}")
        out[want] = max(cands, key=lambda f: len(list(f.glob("*.dbn.zst"))))
    v1_defs, v3_defs = [], []
    for f in by_token.get("definition", []):
        (v3_defs if _dbn_version(f) >= 3 else v1_defs).append(f)
    if not v1_defs:
        raise FileNotFoundError(
            f"No (v1) DEFINITION folder found next to {input_folder}")
    out["definition"] = max(v1_defs,
                            key=lambda f: len(list(f.glob("*.dbn.zst"))))
    if v3_defs:
        out["legs"] = max(v3_defs,
                          key=lambda f: len(list(f.glob("*.dbn.zst"))))
    return out


def _find_candle_folder(input_folder: Path) -> tuple[Path | None, str]:
    asset = input_folder.parent.name
    root = None
    for anc in input_folder.parents:
        if anc.name == "raw_dbn":
            root = anc.parent
            break
    roots = [root] if root is not None else []
    if not roots:
        try:
            from modules.common.backend.settings import load_settings
            roots = load_settings().data_roots
        except Exception:
            roots = []
    for r in roots:
        asset_dir = Path(r) / "parquet" / "Futures" / asset
        if not asset_dir.is_dir():
            continue
        matches = [d for d in asset_dir.iterdir()
                   if d.is_dir() and "ohlcv" in d.name.lower()]
        if matches:
            best = max(matches, key=lambda d: len(list(d.glob("*.parquet"))))
            return best, asset
    return None, asset


def _file_day(path: Path) -> int:
    ymd = path.name.split(".")[0].split("-")[-1]
    return int(pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}").value // _NS_PER_DAY)


def _ny_day(ts_ns: np.ndarray) -> np.ndarray:
    idx = pd.DatetimeIndex(ts_ns, tz="UTC").tz_convert(_NY)
    return idx.tz_localize(None).asi8 // _NS_PER_DAY


# ── definitions (same store as transform A) ───────────────────────────────────

def _bytes_hash(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    m = a.view(np.uint8).reshape(len(a), -1).astype(np.uint64)
    h = np.zeros(len(a), dtype=np.uint64)
    for j in range(m.shape[1]):
        h = h * np.uint64(1099511628211) + m[:, j]
    return h.view(np.int64)


def _load_defs(path: Path) -> dict | None:
    arr = _decode_dbn(path, "definition")
    cls = arr["instrument_class"]
    keep = (cls == b"C") | (cls == b"P")
    a = arr[keep]
    if not len(a):
        return None
    iid = a["instrument_id"].astype(np.int64)
    order = np.argsort(iid, kind="stable")
    a, iid = a[order], iid[order]
    exp = a["expiration"].astype(np.int64)
    act = a["activation"].astype(np.int64)
    strike = a["strike_price"].astype(np.int64)
    mult = a["unit_of_measure_qty"].astype(np.int64)
    cp = (a["instrument_class"] == b"C").astype(np.int64)
    sig = (exp ^ (strike * 1000003) ^ (act >> 1) ^ (mult << 7) ^ (cp << 3)
           ^ _bytes_hash(a["underlying"]) ^ _bytes_hash(a["asset"]))
    return {"iid": iid, "sig": sig, "exp": exp, "act": act, "strike": strike,
            "mult": mult, "cp": cp, "underlying": a["underlying"],
            "series": a["asset"]}


class _DefStore:
    """instrument_id -> static facts, day-over-day diffed, as-of lookups.
    Identical mechanics to transform A's store."""

    def __init__(self):
        self.last_row = np.full(1 << 20, -1, dtype=np.int32)
        self.hist: dict[int, list[int]] = {}
        self.n = 0
        cap = 1 << 16
        self.exp = np.empty(cap, dtype=np.int64)
        self.act = np.empty(cap, dtype=np.int64)
        self.strike = np.empty(cap, dtype=np.int64)
        self.mult = np.empty(cap, dtype=np.int64)
        self.cp = np.empty(cap, dtype=np.int8)
        self.series_code = np.empty(cap, dtype=np.int32)
        self.und_code = np.empty(cap, dtype=np.int32)
        self.iid = np.empty(cap, dtype=np.int64)
        self._series_ids: dict[str, int] = {}
        self._und_ids: dict[str, int] = {}
        self.series_cats: list[str] = []
        self.und_cats: list[str] = []
        self._prev: dict | None = None

    def _ensure(self, n: int) -> None:
        if n <= len(self.exp):
            return
        cap = max(n, 2 * len(self.exp))
        for name in ("exp", "act", "strike", "mult", "cp",
                     "series_code", "und_code", "iid"):
            old = getattr(self, name)
            grown = np.empty(cap, dtype=old.dtype)
            grown[:self.n] = old[:self.n]
            setattr(self, name, grown)

    def _code(self, table: dict, cats: list, raw: bytes) -> int:
        s = raw.decode()
        code = table.get(s)
        if code is None:
            code = len(cats)
            table[s] = code
            cats.append(s)
        return code

    def apply(self, d: dict) -> None:
        prev = self._prev
        if prev is None:
            changed = np.ones(len(d["iid"]), dtype=bool)
        else:
            pos = np.searchsorted(prev["iid"], d["iid"])
            pos_c = np.clip(pos, 0, len(prev["iid"]) - 1)
            same = (prev["iid"][pos_c] == d["iid"]) & (prev["sig"][pos_c] == d["sig"])
            changed = ~same
        idx = np.nonzero(changed)[0]
        row0 = self.n
        n_rows = row0 + len(idx)
        self._ensure(n_rows)
        self.exp[row0:n_rows] = d["exp"][idx]
        self.act[row0:n_rows] = d["act"][idx]
        self.strike[row0:n_rows] = d["strike"][idx]
        self.mult[row0:n_rows] = d["mult"][idx]
        self.cp[row0:n_rows] = d["cp"][idx]
        self.iid[row0:n_rows] = d["iid"][idx]
        for off, j in enumerate(idx.tolist()):
            self.series_code[row0 + off] = self._code(
                self._series_ids, self.series_cats, d["series"][j])
            self.und_code[row0 + off] = self._code(
                self._und_ids, self.und_cats, d["underlying"][j])
        self.n = n_rows

        new_iid = d["iid"][idx]
        new_row = np.arange(row0, n_rows, dtype=np.int32)
        if len(new_iid) and int(new_iid.max()) >= len(self.last_row):
            cap = int(new_iid.max() * 5 // 4) + 1
            grown = np.full(cap, -1, dtype=np.int32)
            grown[:len(self.last_row)] = self.last_row
            self.last_row = grown
        old = self.last_row[new_iid]
        for k in np.nonzero(old >= 0)[0]:
            iid_k = int(new_iid[k])
            self.hist.setdefault(iid_k, [int(old[k])]).append(int(new_row[k]))
        self.last_row[new_iid] = new_row
        self._prev = d

    def lookup(self, iids: np.ndarray, session_day: int) -> np.ndarray:
        act_cut = (session_day + 1) * _NS_PER_DAY
        safe = np.minimum(iids, len(self.last_row) - 1)
        out = np.where(iids < len(self.last_row),
                       self.last_row[safe], np.int32(-1)).astype(np.int64)
        redo = out >= 0
        redo[redo] = self.act[out[redo]] >= act_cut
        if redo.any():
            act = self.act
            for k in np.nonzero(redo)[0]:
                hist = self.hist.get(int(iids[k]))
                if hist is None:
                    continue
                for r in reversed(hist):
                    if act[r] < act_cut:
                        out[k] = r
                        break
        return out


# ── statistics (OI only) & TBBO loaders ───────────────────────────────────────

def _load_oi(path: Path) -> dict:
    """stat_type 9 (open interest) records, reduced."""
    arr = _decode_dbn(path, "statistics")
    o = arr[arr["stat_type"] == 9]
    ref = o["ts_ref"].astype(np.int64)
    ok = (ref > _TS_VALID_LO) & (ref < _TS_VALID_HI)
    o, ref = o[ok], ref[ok]
    ts = o["ts_event"].astype(np.int64)
    return {"iid": o["instrument_id"].astype(np.int64),
            "ref_day": ref // _NS_PER_DAY,
            "ts": ts,
            "ny_day": _ny_day(ts),
            "qty": o["quantity"].astype(np.int64)}


def _load_tbbo(path: Path) -> dict:
    """Trades with quote-at-trade, aggressor-classified (price rule, side
    field breaking bid==ask ties, mid/undefined -> 0)."""
    arr = _decode_dbn(path, "tbbo")
    t = arr[arr["action"] == b"T"]
    px_raw = t["price"].astype(np.int64)
    bid_raw = t["bid_px_00"].astype(np.int64)
    ask_raw = t["ask_px_00"].astype(np.int64)
    px = px_raw / 1e9
    bid = np.where(bid_raw == _I64_MAX, np.nan, bid_raw / 1e9)
    ask = np.where(ask_raw == _I64_MAX, np.nan, ask_raw / 1e9)
    size = t["size"].astype(np.int64)
    is_buy = px_raw >= ask_raw
    is_sell = px_raw <= bid_raw
    is_buy &= ask_raw != _I64_MAX
    is_sell &= bid_raw != _I64_MAX
    both = is_buy & is_sell                    # locked quotes: side breaks tie
    side = t["side"]
    is_buy[both] = side[both] == b"B"
    is_sell[both] = side[both] == b"A"
    signed = np.where(is_buy, -size, np.where(is_sell, size, 0))
    return {"iid": t["instrument_id"].astype(np.int64),
            "ts": t["ts_event"].astype(np.int64),
            "px": px, "size": size, "bid": bid, "ask": ask,
            "signed": signed.astype(np.int64)}


def _load_spread_legs(path: Path) -> dict:
    """DBN v3 definitions: leg decomposition of strategy instruments.

    v3 encodes a strategy as leg_count records (one per leg).  Keeps only
    option legs (C/P — futures legs of covered strategies hedge delta and
    don't touch option OI).  sign: +1 when buying the strategy buys the leg.
    """
    import databento as db   # v3 layout -> databento's own decoder
    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    a = arr[arr["leg_count"] > 0]
    lc = a["leg_instrument_class"]
    a = a[(lc == b"C") | (lc == b"P")]
    sign = np.where(a["leg_side"] == b"B", 1,
                    np.where(a["leg_side"] == b"A", -1, 0)).astype(np.int8)
    den = np.maximum(a["leg_ratio_qty_denominator"].astype(np.int64), 1)
    ratio = (a["leg_ratio_qty_numerator"].astype(np.int64) // den)
    return {"sid": a["instrument_id"].astype(np.int64),
            "leg_iid": a["leg_instrument_id"].astype(np.int64),
            "sign": sign.astype(np.int64), "ratio": ratio}


def _dbn_version(folder: Path) -> int:
    """DBN version byte of the first file in the folder (0 when unreadable)."""
    for f in sorted(folder.glob("*.dbn.zst")):
        try:
            r = zstandard.ZstdDecompressor().stream_reader(
                open(f, "rb"), read_across_frames=True)
            head = r.read(4)
            return head[3] if head[:3] == b"DBN" else 0
        except Exception:
            return 0
    return 0


# ── process-pool plumbing (same trick as transform A, distinct plant name) ────

_PLANT_NAME = "_options_5m_worker"


def _worker(kind: str, path: str):
    if kind == "tbbo":
        return _load_tbbo(Path(path))
    if kind == "oi":
        return _load_oi(Path(path))
    if kind == "legs":
        return _load_spread_legs(Path(path))
    return _load_defs(Path(path))


def _plant_worker():
    fn = _worker
    fn.__module__ = "builtins"
    fn.__qualname__ = _PLANT_NAME
    setattr(builtins, _PLANT_NAME, fn)
    return fn


if __name__ == "<run_path>":
    _plant_worker()


# ── front-month roll calendar & significance — VERBATIM FROM TRANSFORM A ─────

def _build_roll_calendar(candle_folder: Path | None, asset: str,
                         y0: int = 2006, y1: int = 2035) -> dict:
    rule = ROLL_RULES.get(asset, _DEFAULT_ROLL_RULE)
    months = rule["quarter_months"]
    wd, wk = rule["expiry_weekday"], rule["expiry_week"]
    hh, mm = rule["expiry_time"]
    off = rule["roll_offset_days"]

    rows = []
    for y in range(y0, y1 + 1):
        for mo in months:
            first = pd.Timestamp(year=y, month=mo, day=1)
            day = 1 + ((wd - first.dayofweek) % 7) + 7 * (wk - 1)
            exp = pd.Timestamp(f"{y}-{mo:02d}-{day:02d} {hh:02d}:{mm:02d}",
                               tz=_NY)
            rows.append((exp.value, _MONTHCHAR[mo],
                         f"{asset}{_MONTHCHAR[mo]}{str(y)[-1]}"))
    rows.sort()
    q_exp = np.array([r[0] for r in rows], dtype=np.int64)
    q_mc = np.array([r[1] for r in rows])
    q_str = np.array([r[2] for r in rows])
    q_cme = q_exp - off * _NS_PER_DAY

    q_vol = np.full(len(q_exp), -1, dtype=np.int64)
    n_vol = 0
    if candle_folder is not None:
        cfiles = [f for f in candle_folder.glob("*.parquet")
                  if f.stem[:1].isdigit()]

        def _footer(f: Path):
            try:
                v = (pq.read_metadata(f).metadata or {}).get(b"front_month")
            except Exception:
                return None
            return (int(pd.Timestamp(f.stem).value), v.decode()) if v else None

        with ThreadPoolExecutor(min(8, len(cfiles) or 1)) as tp:
            fm = dict(filter(None, tp.map(_footer, cfiles)))
        if fm:
            fdays = np.array(sorted(fm), dtype=np.int64)
            fcode = np.array([fm[d] for d in fdays])
            became = np.full(len(q_exp), -1, dtype=np.int64)
            for k in range(len(q_exp)):
                lo = q_exp[k] - 160 * _NS_PER_DAY
                sel = (fdays >= lo) & (fdays < q_exp[k]) & (fcode == q_str[k])
                if sel.any():
                    became[k] = fdays[sel].min()
            q_vol[:-1] = became[1:]
            n_vol = int((q_vol >= 0).sum())
    q_min = np.where(q_vol >= 0, np.minimum(q_vol, q_cme), q_cme)
    return {"exp": q_exp, "exp_eff": q_exp.copy(), "mc": q_mc, "str": q_str,
            "cme": q_cme, "min": q_min, "vol": q_vol, "n_vol": n_vol,
            "n_q": len(q_exp)}


def _refine_real_expiries(cal: dict, real_max: np.ndarray) -> None:
    w = ((real_max >= cal["exp"] - 4 * _NS_PER_DAY)
         & (real_max <= cal["exp"]))
    cal["exp_eff"] = np.where(w, real_max, cal["exp"])


def _contract_ku(und_str, exp, cal: dict) -> np.ndarray:
    n = len(exp)
    ku = np.full(n, -1, dtype=np.int64)
    if not n:
        return ku
    q_exp, q_mc = cal["exp"], cal["mc"]
    mc = np.array([u[-2] if len(u) >= 3 else "?" for u in und_str])
    for ch in np.unique(mc):
        idxs = np.nonzero(q_mc == ch)[0]
        if not len(idxs):
            continue
        exps = q_exp[idxs]
        sel = mc == ch
        pos = np.searchsorted(exps, exp[sel], side="left")
        ok = pos < len(exps)
        res = np.full(int(sel.sum()), -1, dtype=np.int64)
        res[ok] = idxs[np.clip(pos, 0, len(exps) - 1)][ok]
        ku[sel] = res
    return ku


def _significance_rows(ku, always_flag, exp, ts, cal: dict) -> np.ndarray:
    NEG = np.iinfo(np.int64).min
    cur = np.searchsorted(cal["exp_eff"], ts, side="right")
    have = ku >= 0
    is_cur = have & (ku == cur)
    qmin_cur = np.where(cur < cal["n_q"],
                        cal["min"][np.clip(cur, 0, cal["n_q"] - 1)], NEG)
    is_next = have & (ku == cur + 1) & (exp >= qmin_cur)
    return always_flag | is_cur | is_next


# ── sidecar flow state ────────────────────────────────────────────────────────

def _load_state(out_dir: Path, tape_start: int):
    p = out_dir / _STATE_FILE
    if not p.exists():
        return None
    try:
        t = pq.read_table(p)
        meta = t.schema.metadata or {}
        if int(meta.get(b"tape_start", b"-1")) != tape_start:
            return None
        d = t.to_pandas()
        return {"through_day": int(meta[b"through_day"]),
                "iid": d["iid"].to_numpy(np.int64),
                "cum": d["cum"].to_numpy(np.int64),
                "bid": d["bid"].to_numpy(np.float64),
                "ask": d["ask"].to_numpy(np.float64),
                "oi": d["oi"].to_numpy(np.int64),
                "oi_morning": d["oi_morning"].to_numpy(bool),
                "seen": d["seen"].to_numpy(bool),
                "sess_net": d["sess_net"].to_numpy(np.int64),
                "cum_sp": d["cum_sp"].to_numpy(np.int64)}
    except Exception:
        return None


def _save_state(out_dir: Path, tape_start: int, through_day: int,
                iid, cum, bid, ask, oi, oi_morning, seen, sess_net,
                cum_sp) -> None:
    # sorted by iid (searchsorted on load) and deduped keeping the latest def
    # row per iid (a reused id's previous contract has expired anyway)
    order = np.lexsort((np.arange(len(iid)), iid))
    last = np.ones(len(iid), dtype=bool)
    ii = np.asarray(iid)[order]
    last[:-1] = ii[1:] != ii[:-1]
    sel = order[last]
    iid, cum, bid, ask, oi, oi_morning, seen, sess_net, cum_sp = (
        np.asarray(a)[sel] for a in (iid, cum, bid, ask, oi, oi_morning,
                                     seen, sess_net, cum_sp))
    t = pa.table({"iid": iid, "cum": cum, "bid": bid, "ask": ask,
                  "oi": oi, "oi_morning": oi_morning, "seen": seen,
                  "sess_net": sess_net, "cum_sp": cum_sp})
    t = t.replace_schema_metadata({b"tape_start": str(tape_start).encode(),
                                   b"through_day": str(through_day).encode()})
    tmp = out_dir / (_STATE_FILE + ".tmp")
    pq.write_table(t, tmp)
    os.replace(tmp, out_dir / _STATE_FILE)


# ── driver ────────────────────────────────────────────────────────────────────

def run_all(
        input_folder: str,
        output_folder: str,
        skip_existing: bool = True,
        on_progress: callable = None,
        params: dict | None = None,
) -> None:
    def progress(cur: int, total: int, msg: str) -> None:
        if on_progress:
            on_progress(cur, total, msg)

    drop_insignificant = bool((params or {}).get("drop_insignificant", False))

    folders = _resolve_folders(Path(input_folder))
    candle_folder, asset = _find_candle_folder(folders["tbbo"])
    out_dir = Path(output_folder)

    tbbo_files = sorted(folders["tbbo"].glob("*.dbn.zst"), key=_file_day)
    if len(tbbo_files) < 2:
        progress(1, 1, "ERROR: need at least 2 TBBO files to build a session")
        return
    tbbo_days = [_file_day(f) for f in tbbo_files]
    range_lo, range_hi = tbbo_days[0], tbbo_days[-1]

    # tape start = first tradable moment covered = 18:00 NY on the first file's
    # date (files are UTC days; the head of the first file precedes any session)
    tape_start = int(pd.Timestamp(str(pd.Timestamp(range_lo * _NS_PER_DAY).date())
                                  + " 18:00", tz=_NY).value)

    # sessions: curr file = a weekday; window = 18:00 NY prev calendar day ->
    # 17:00 NY on the session day
    sessions = []                     # (session_day, prev_idx, curr_idx)
    for i in range(1, len(tbbo_files)):
        d = tbbo_days[i]
        if pd.Timestamp(d * _NS_PER_DAY).weekday() >= 5:
            continue
        sessions.append((d, i - 1, i))
    if not sessions:
        progress(1, 1, "ERROR: no weekday TBBO sessions found")
        return

    # ── resume bookkeeping ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_days = set()
    for f in out_dir.glob("*.parquet"):
        if f.stem[:1].isdigit():
            try:
                existing_days.add(int(pd.Timestamp(f.stem).value // _NS_PER_DAY))
            except ValueError:
                pass
    state = _load_state(out_dir, tape_start) if skip_existing else None
    state_through = state["through_day"] if state else -1
    # a missing file BEHIND the folded state cannot be rebuilt from the
    # sidecar (it only holds state at through_day) -> re-decode from the start
    if state is not None and any(s[0] <= state_through
                                 and s[0] not in existing_days
                                 for s in sessions):
        state = None
        state_through = -1
    if skip_existing and existing_days and state is None:
        progress(0, 1, "NOTE: flow-state sidecar missing/stale/behind a gap — "
                       "re-decoding the tape from the start to rebuild dealer "
                       "flow (existing daily files are kept, not rewritten)")

    # sessions to DECODE: those not yet folded into state.  Files needed start
    # at the prev file of the first such session.
    todo = [s for s in sessions if s[0] > state_through]
    if not todo:
        progress(1, 1, "↷ Everything up to date")
        return
    first_file_idx = todo[0][1]
    first_day_needed = tbbo_days[first_file_idx]

    def_files = sorted(folders["definition"].glob("*.dbn.zst"), key=_file_day)
    older = [f for f in def_files if _file_day(f) < first_day_needed]
    def_files = [f for f in def_files
                 if first_day_needed <= _file_day(f) <= range_hi]
    if older:
        def_files = [older[-1]] + def_files
    stats_files = sorted(folders["statistics"].glob("*.dbn.zst"), key=_file_day)
    stats_files = [f for f in stats_files
                   if first_day_needed - 6 <= _file_day(f) <= range_hi]
    tbbo_todo = tbbo_files[first_file_idx:]

    total = len(todo) + 1
    if candle_folder is None:
        progress(0, total, f"WARNING: no *ohlcv* candle folder found for "
                           f"{asset} — volume_roll falls back to the "
                           f"scheduled roll")

    # roll calendar (same volume_roll source as transform A: candle footers)
    always = ASSETS_TO_ALWAYS_TRACK.get(asset, ())
    cal = _build_roll_calendar(candle_folder, asset)

    # constant multiplier for the file metadata (modal populated value)
    multiplier = 0
    try:
        d_last = _load_defs(def_files[-1])
        if d_last is not None and (d_last["mult"] != 0).any():
            vals, counts = np.unique(d_last["mult"][d_last["mult"] != 0],
                                     return_counts=True)
            multiplier = int(round(int(vals[np.argmax(counts)]) / 1e9))
    except Exception:
        pass

    defs = _DefStore()
    times = {"decode_wait": 0.0, "defs": 0.0, "oi": 0.0, "session": 0.0,
             "write": 0.0}

    # per-contract (didx) static + rolling state, grown with defs.n
    cap0 = 1 << 16
    ku_by_didx = np.full(cap0, -1, dtype=np.int64)
    always_by_didx = np.zeros(cap0, dtype=bool)
    flow_cum = np.zeros(cap0, dtype=np.int64)       # signed flow since activation
    flow_cum_sp = np.zeros(cap0, dtype=np.int64)    # spread-leg flow since activ.
    q_bid = np.full(cap0, np.nan)                   # last known quote
    q_ask = np.full(cap0, np.nan)
    oi_val = np.zeros(cap0, dtype=np.int64)         # last known OI value
    oi_is_morning = np.zeros(cap0, dtype=bool)
    seen_b = np.zeros(cap0, dtype=bool)             # contract ever emitted a row
    sess_net = np.zeros(cap0, dtype=np.int64)       # prev session net buy-sell
    real_max = np.full(cal["n_q"], np.iinfo(np.int64).min, dtype=np.int64)
    sig_done = 0
    seeded = state is not None

    def _extend_static() -> None:
        nonlocal ku_by_didx, always_by_didx, flow_cum, flow_cum_sp, q_bid, q_ask
        nonlocal oi_val, oi_is_morning, seen_b, sess_net, sig_done
        if defs.n <= sig_done:
            return
        lo, hi = sig_done, defs.n
        if hi > len(ku_by_didx):
            cap = max(hi, 2 * len(ku_by_didx))

            def grow(a, fill):
                g = np.full(cap, fill, dtype=a.dtype)
                g[:sig_done] = a[:sig_done]
                return g
            ku_by_didx = grow(ku_by_didx, -1)
            always_by_didx = grow(always_by_didx, False)
            flow_cum = grow(flow_cum, 0)
            flow_cum_sp = grow(flow_cum_sp, 0)
            q_bid = grow(q_bid, np.nan)
            q_ask = grow(q_ask, np.nan)
            oi_val = grow(oi_val, 0)
            oi_is_morning = grow(oi_is_morning, False)
            seen_b = grow(seen_b, False)
            sess_net = grow(sess_net, 0)
        und_str = [defs.und_cats[c] for c in defs.und_code[lo:hi]]
        root_str = np.asarray([defs.series_cats[c] for c in defs.series_code[lo:hi]])
        ku = _contract_ku(und_str, defs.exp[lo:hi], cal)
        ku_by_didx[lo:hi] = ku
        always_by_didx[lo:hi] = (np.isin(root_str, np.asarray(always))
                                 if always else False)
        okm = ku >= 0
        if okm.any():
            np.maximum.at(real_max, ku[okm], defs.exp[lo:hi][okm])
            _refine_real_expiries(cal, real_max)
        # a redefinition row inherits its predecessor's rolling state ONLY if
        # it is the SAME economic contract (definition amendments — e.g. the
        # Juneteenth expiry shift — or contracts that vanish from a daily
        # snapshot and reappear): same strike/cp/underlying/series and expiry
        # within a week.  Anything else is an id REUSE for a brand-new
        # contract (delisted strikes free their ids) -> fresh state.
        for j in range(lo, hi):
            hist = defs.hist.get(int(defs.iid[j]))
            if hist and len(hist) >= 2 and hist[-1] == j:
                o = hist[-2]
                if (defs.strike[o] == defs.strike[j]
                        and defs.cp[o] == defs.cp[j]
                        and defs.und_code[o] == defs.und_code[j]
                        and defs.series_code[o] == defs.series_code[j]
                        and abs(int(defs.exp[o]) - int(defs.exp[j]))
                        <= 7 * _NS_PER_DAY):
                    flow_cum[j] = flow_cum[o]
                    flow_cum_sp[j] = flow_cum_sp[o]
                    q_bid[j] = q_bid[o]
                    q_ask[j] = q_ask[o]
                    oi_val[j] = oi_val[o]
                    oi_is_morning[j] = oi_is_morning[o]
                    seen_b[j] = seen_b[o]
                    sess_net[j] = sess_net[o]
        # seed rolling state from the sidecar for newly known contracts
        if seeded:
            pos = np.searchsorted(state["iid"], defs.iid[lo:hi])
            pos_c = np.clip(pos, 0, len(state["iid"]) - 1)
            hit = (state["iid"][pos_c] == defs.iid[lo:hi]) if len(state["iid"]) else \
                np.zeros(hi - lo, dtype=bool)
            rows = np.arange(lo, hi)[hit]
            src = pos_c[hit]
            flow_cum[rows] = state["cum"][src]
            flow_cum_sp[rows] = state["cum_sp"][src]
            q_bid[rows] = state["bid"][src]
            q_ask[rows] = state["ask"][src]
            oi_val[rows] = state["oi"][src]
            oi_is_morning[rows] = state["oi_morning"][src]
            seen_b[rows] = state["seen"][src]
            sess_net[rows] = state["sess_net"][src]
        sig_done = hi

    # OI prints buffered per as-of session (ref day), resolved to didx
    oi_buffer: dict[int, list] = {}      # ref_day -> [(didx, ts, qty, ny_day)]

    # strategy legs (v3 defs): spread iid -> (leg_iids, signs*ratios) — daily
    # snapshots, overwritten as they stream (as-of resolution like the defs)
    spread_map: dict[int, tuple] = {}

    def _apply_legs(res: dict) -> None:
        sid = res["sid"]
        order = np.argsort(sid, kind="stable")
        sid = sid[order]
        li = res["leg_iid"][order]
        w = (res["sign"][order] * res["ratio"][order]).astype(np.int64)
        rt = res["ratio"][order]
        starts = np.nonzero(np.r_[True, sid[1:] != sid[:-1]])[0]
        bounds = np.r_[starts, len(sid)]
        for j in range(len(starts)):
            a, b = bounds[j], bounds[j + 1]
            spread_map[int(sid[a])] = (li[a:b], w[a:b], rt[a:b])

    # writer pool
    writer_pool = ThreadPoolExecutor(4)
    write_futs: list = []
    n_written = n_skipped = n_rows_total = n_dropped = 0
    n_unknown_iid = n_tombstones = 0
    prev_payload = None                  # last consumed tbbo payload
    folded_upto = tape_start if state is None else \
        int(pd.Timestamp(str(pd.Timestamp(state_through * _NS_PER_DAY).date())
                         + " 17:00", tz=_NY).value)
    sessions_done = state_through

    def _fold_into(target: np.ndarray, didx: np.ndarray, signed: np.ndarray,
                   ts: np.ndarray, lo_ts: int, hi_ts: int) -> None:
        """Fold trades with lo_ts < ts <= hi_ts into a cumulative state,
        excluding post-expiry trades (mirrors A's suppression)."""
        m = (ts > lo_ts) & (ts <= hi_ts) & (didx >= 0)
        m &= ts <= defs.exp[np.clip(didx, 0, None)]
        if m.any():
            np.add.at(target, didx[m], signed[m])

    def _write_session(day: int, frame: pd.DataFrame, contract_didx) -> None:
        nonlocal n_written, n_rows_total
        meta_contracts = {}
        for r in contract_didx.tolist():
            meta_contracts[str(int(defs.iid[r]))] = {
                "underlying": defs.und_cats[defs.und_code[r]],
                "series": defs.series_cats[defs.series_code[r]],
                "expiry": str(pd.Timestamp(int(defs.exp[r]), tz="UTC")
                              .tz_convert(_NY)),
                "strike": float(defs.strike[r]) / 1e9,
                "cp_flag": "call" if defs.cp[r] else "put",
                "activation_date": str(pd.Timestamp(int(defs.act[r]), tz="UTC")
                                       .tz_convert(_NY)),
            }
        contracts_json = json.dumps(meta_contracts).encode()
        date_str = str(pd.Timestamp(day * _NS_PER_DAY).date())

        def _to_disk(fr=frame, cj=contracts_json, ds=date_str):
            table = pa.Table.from_pandas(fr)   # in the writer thread (GIL-lite)
            meta = dict(table.schema.metadata or {})
            meta[b"contracts"] = cj
            meta[b"multiplier"] = str(multiplier).encode()
            pq.write_table(table.replace_schema_metadata(meta),
                           out_dir / f"{ds}.parquet")
        write_futs.append(writer_pool.submit(_to_disk))
        n_written += 1
        n_rows_total += len(frame)
        while len(write_futs) > 8:
            write_futs.pop(0).result()

    # ── the per-session engine ────────────────────────────────────────────────
    def _process_session(day: int, prev_t: dict, curr_t: dict) -> None:
        nonlocal folded_upto, n_unknown_iid, n_tombstones
        nonlocal n_skipped, n_dropped, sessions_done
        w_start = int(pd.Timestamp(
            str(pd.Timestamp((day - 1) * _NS_PER_DAY).date()) + " 18:00",
            tz=_NY).value)
        w_end = int(pd.Timestamp(
            str(pd.Timestamp(day * _NS_PER_DAY).date()) + " 17:00",
            tz=_NY).value)
        n_k = int((w_end - w_start) // _NS_5MIN)
        _extend_static()

        # stitch + resolve contracts (as-of lookup; unknown iids skipped)
        iid = np.concatenate([prev_t["iid"], curr_t["iid"]])
        ts = np.concatenate([prev_t["ts"], curr_t["ts"]])
        px = np.concatenate([prev_t["px"], curr_t["px"]])
        size = np.concatenate([prev_t["size"], curr_t["size"]])
        bid = np.concatenate([prev_t["bid"], curr_t["bid"]])
        ask = np.concatenate([prev_t["ask"], curr_t["ask"]])
        signed = np.concatenate([prev_t["signed"], curr_t["signed"]])
        didx = defs.lookup(iid, day)

        # ── explode spread trades into their option LEGS (v3 leg maps) ──────
        # a spread trade of size N: each leg's outright gets N × ratio,
        # signed by (trade aggressor × leg side) — same dealer convention
        unk = np.nonzero(didx < 0)[0]
        L_iid = np.empty(0, np.int64)
        L_ts = np.empty(0, np.int64)
        L_signed = np.empty(0, np.int64)
        L_size = np.empty(0, np.int64)
        if len(unk):
            u_order = unk[np.argsort(iid[unk], kind="stable")]
            u_iid = iid[u_order]
            starts_u = np.nonzero(np.r_[True, u_iid[1:] != u_iid[:-1]])[0]
            bounds_u = np.r_[starts_u, len(u_iid)]
            pi, pt, ps, pz = [], [], [], []
            for j in range(len(starts_u)):
                a, b = bounds_u[j], bounds_u[j + 1]
                legs = spread_map.get(int(u_iid[a]))
                if legs is None:
                    n_unknown_iid += b - a
                    continue
                li, w, rt = legs
                tr = u_order[a:b]
                nl = len(li)
                pi.append(np.tile(li, len(tr)))
                pt.append(np.repeat(ts[tr], nl))
                ps.append(np.repeat(signed[tr], nl) * np.tile(w, len(tr)))
                pz.append(np.repeat(size[tr], nl) * np.tile(rt, len(tr)))
            if pi:
                L_iid = np.concatenate(pi)
                L_ts = np.concatenate(pt)
                L_signed = np.concatenate(ps)
                L_size = np.concatenate(pz)
        L_didx = defs.lookup(L_iid, day) if len(L_iid) else L_iid
        if len(L_didx):
            okl = L_didx >= 0
            n_unknown_iid += int((~okl).sum())
            L_didx, L_ts = L_didx[okl], L_ts[okl]
            L_signed, L_size = L_signed[okl], L_size[okl]

        # fold any pre-window leftovers (holiday evenings etc.), keep flow exact
        _fold_into(flow_cum, didx, signed, ts, folded_upto, w_start)
        _fold_into(flow_cum_sp, L_didx, L_signed, L_ts, folded_upto, w_start)

        in_w = (didx >= 0) & (ts > w_start) & (ts <= w_end)
        in_w &= ts <= defs.exp[np.clip(didx, 0, None)]     # no post-expiry rows
        didx_w, ts_w = didx[in_w], ts[in_w]
        px_w, size_w = px[in_w], size[in_w]
        bid_w, ask_w, signed_w = bid[in_w], ask[in_w], signed[in_w]
        k_w = ((ts_w - w_start) // _NS_5MIN).astype(np.int64)
        k_w = np.minimum(k_w, n_k - 1)                     # ts == w_end edge

        # OI regime for this session.  The buffered refs identify themselves:
        # the largest ref < session day IS the previous session; older refs are
        # stale (holiday gaps) and only advance the rolling state.  Evening
        # value = first print per contract, morning value = last print whose
        # NY date is the session day (with its publication ts).
        refs = sorted(r for r in oi_buffer if r < day)
        eve: dict[int, int] = {}
        morn: dict[int, tuple] = {}
        for ref in refs[:-1]:                       # stale -> rolling only
            for (pdx, pts, pqty, _pny) in oi_buffer.pop(ref):
                order = np.argsort(pts, kind="stable")
                oi_val[pdx[order]] = pqty[order]
                oi_is_morning[pdx[order]] = False
        if refs:
            for (pdx, pts, pqty, pny) in oi_buffer.pop(refs[-1]):
                order = np.argsort(pts, kind="stable")
                pdx, pts, pqty, pny = (pdx[order], pts[order],
                                       pqty[order], pny[order])
                for j in range(len(pdx)):
                    r = int(pdx[j])
                    if r not in eve:
                        eve[r] = (int(pqty[j]), int(pts[j]))
                    if pny[j] == day:
                        morn[r] = (int(pqty[j]), int(pts[j]))

        # ── traded aggregation per (contract, bucket) ────────────────────────
        order = np.lexsort((ts_w, k_w, didx_w))
        dxs, ks = didx_w[order], k_w[order]
        pxs, szs = px_w[order], size_w[order]
        bids, asks, sgs = bid_w[order], ask_w[order], signed_w[order]
        newg = np.ones(len(dxs), dtype=bool)
        if len(dxs):
            newg[1:] = (dxs[1:] != dxs[:-1]) | (ks[1:] != ks[:-1])
        starts = np.nonzero(newg)[0]
        g_didx = dxs[starts]
        g_k = ks[starts]
        ends = (np.r_[starts[1:], len(dxs)] - 1) if len(starts) else starts
        g_open = pxs[starts]
        g_close = pxs[ends]
        g_high = np.maximum.reduceat(pxs, starts) if len(starts) else pxs[:0]
        g_low = np.minimum.reduceat(pxs, starts) if len(starts) else pxs[:0]
        g_vol = np.add.reduceat(szs, starts) if len(starts) else szs[:0]
        g_buy = (np.add.reduceat(np.where(sgs < 0, -sgs, 0), starts)
                 if len(starts) else szs[:0])
        g_sell = (np.add.reduceat(np.where(sgs > 0, sgs, 0), starts)
                  if len(starts) else szs[:0])
        g_signed = (np.add.reduceat(sgs, starts) if len(starts) else szs[:0])
        g_bid = bids[ends]
        g_ask = asks[ends]

        # per-contract within-session cumulative signed flow at each group:
        # global cumsum minus the running total just before each contract's
        # first group (repeated across that contract's groups)
        cum = np.cumsum(g_signed)
        gnew_c = np.ones(len(g_didx), dtype=bool)
        if len(g_didx):
            gnew_c[1:] = g_didx[1:] != g_didx[:-1]
        first_idx = np.nonzero(gnew_c)[0]
        if len(first_idx):
            counts = np.diff(np.r_[first_idx, len(g_didx)])
            prior = np.where(first_idx > 0, cum[first_idx - 1], 0)
            g_cumflow = cum - np.repeat(prior, counts)
        else:
            g_cumflow = cum

        # ── spread-leg aggregation per (contract, bucket) ────────────────────
        in_l = (L_ts > w_start) & (L_ts <= w_end)
        in_l &= L_ts <= defs.exp[np.clip(L_didx, 0, None)]
        Ld, Lt = L_didx[in_l], L_ts[in_l]
        Ls, Lz = L_signed[in_l], L_size[in_l]
        Lk = np.minimum((Lt - w_start) // _NS_5MIN, n_k - 1).astype(np.int64)
        order_l = np.lexsort((Lt, Lk, Ld))
        Ld, Lk = Ld[order_l], Lk[order_l]
        Ls, Lz = Ls[order_l], Lz[order_l]
        newl = np.ones(len(Ld), dtype=bool)
        if len(Ld):
            newl[1:] = (Ld[1:] != Ld[:-1]) | (Lk[1:] != Lk[:-1])
        starts_l = np.nonzero(newl)[0]
        s_didx = Ld[starts_l]
        s_k = Lk[starts_l]
        s_vol = np.add.reduceat(Lz, starts_l) if len(starts_l) else Lz[:0]
        s_buy = (np.add.reduceat(np.where(Ls < 0, -Ls, 0), starts_l)
                 if len(starts_l) else Lz[:0])
        s_sell = (np.add.reduceat(np.where(Ls > 0, Ls, 0), starts_l)
                  if len(starts_l) else Lz[:0])
        s_signed = (np.add.reduceat(Ls, starts_l) if len(starts_l) else Lz[:0])
        cum_l = np.cumsum(s_signed)
        snew_c = np.ones(len(s_didx), dtype=bool)
        if len(s_didx):
            snew_c[1:] = s_didx[1:] != s_didx[:-1]
        first_l = np.nonzero(snew_c)[0]
        if len(first_l):
            counts_l = np.diff(np.r_[first_l, len(s_didx)])
            prior_l = np.where(first_l > 0, cum_l[first_l - 1], 0)
            s_cum = cum_l - np.repeat(prior_l, counts_l)
        else:
            s_cum = cum_l
        skey = s_didx * (n_k + 1) + s_k          # sorted (lexsort by didx,k)

        # ── OI-update event rows (sparse output: a row exists only where a
        # trade happened or a CME OI print landed) ────────────────────────────
        def _pairs(d: dict):
            rows = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            vals = np.fromiter((v for v, _t in d.values()),
                               dtype=np.int64, count=len(d))
            tss = np.fromiter((t for _v, t in d.values()),
                              dtype=np.int64, count=len(d))
            return rows, vals, tss

        e_rows, e_vals, e_tss = (_pairs(eve) if eve else
                                 (np.empty(0, np.int64),) * 3)
        m_rows, m_vals, m_tss = (_pairs(morn) if morn else
                                 (np.empty(0, np.int64),) * 3)
        o_didx = np.concatenate([e_rows, m_rows])
        o_ts = np.concatenate([e_tss, m_tss])
        # prints published before the session (Sunday republications) show at
        # the first bucket: the value in force from session start
        o_k = np.clip((o_ts - w_start) // _NS_5MIN, 0, n_k - 1)
        o_label = w_start + o_k * _NS_5MIN
        keep_o = (defs.act[o_didx] <= w_end) & (o_label < defs.exp[o_didx])
        o_didx, o_k = o_didx[keep_o], o_k[keep_o]
        # dedup: one row per (contract, bucket); traded buckets already exist
        okey = o_didx * (n_k + 1) + o_k
        okey, uniq_idx = np.unique(okey, return_index=True)
        o_didx, o_k = o_didx[uniq_idx], o_k[uniq_idx]
        gkey = g_didx * (n_k + 1) + g_k          # sorted (lexsort by didx,k)
        fresh_o = ~np.isin(okey, gkey)
        o_didx, o_k, okey = o_didx[fresh_o], o_k[fresh_o], okey[fresh_o]

        # quote PAIR + session flow as of a bucket, from the contract's last
        # traded group at/before it, else the carried state (used by OI-only
        # rows and expiry tombstones)
        def _carry_at(c_didx, c_key):
            pos = np.searchsorted(gkey, c_key, side="right") - 1
            pos_c = np.clip(pos, 0, max(len(gkey) - 1, 0))
            hit = ((pos >= 0) & (g_didx[pos_c] == c_didx)) if len(gkey) else \
                np.zeros(len(c_didx), dtype=bool)
            c_bid = np.where(hit, g_bid[pos_c] if len(gkey) else np.nan,
                             q_bid[c_didx])
            c_ask = np.where(hit, g_ask[pos_c] if len(gkey) else np.nan,
                             q_ask[c_didx])
            c_flow = np.where(hit, g_cumflow[pos_c] if len(gkey) else 0, 0)
            return c_bid, c_ask, c_flow

        # spread session-cum as of a bucket (exact bucket included): last
        # spread group of the contract at/before the bucket, else 0
        def _sp_cum_at(c_didx, c_key):
            if not len(skey):
                return np.zeros(len(c_didx), dtype=np.int64)
            pos = np.searchsorted(skey, c_key, side="right") - 1
            pos_c = np.clip(pos, 0, len(skey) - 1)
            hit = (pos >= 0) & (s_didx[pos_c] == c_didx)
            return np.where(hit, s_cum[pos_c], 0)

        o_bid, o_ask, o_flow = _carry_at(o_didx, okey)

        # spread-only rows: leg activity in a bucket with no outright trade
        # and no OI print there
        so_mask = ~np.isin(skey, gkey) & ~np.isin(skey, okey)
        so_didx, so_k, so_key = s_didx[so_mask], s_k[so_mask], skey[so_mask]
        so_bid, so_ask, so_flow = _carry_at(so_didx, so_key)

        # contracts appearing this session (rows exist) can later tombstone
        seen_b[g_didx] = True
        seen_b[o_didx] = True
        seen_b[so_didx] = True

        # ── expiry tombstones: ONE final row at the bucket containing the
        # real expiry + 1s, open_interest = -1, for contracts B has shown ────
        texp = defs.exp[:defs.n]
        cand_t = np.nonzero((texp > w_start) & (texp <= w_end)
                            & seen_b[:defs.n])[0]
        if len(cand_t):
            u_t = np.unique(defs.iid[cand_t])
            t_didx = defs.lookup(u_t, day)
            t_didx = t_didx[t_didx >= 0]
            m_t = ((defs.exp[t_didx] > w_start) & (defs.exp[t_didx] <= w_end)
                   & seen_b[t_didx])
            t_didx = t_didx[m_t]
        else:
            t_didx = cand_t
        t_k = np.clip((defs.exp[t_didx] + _NS_1S - w_start) // _NS_5MIN,
                      0, n_k - 1)
        tkey = t_didx * (n_k + 1) + t_k
        clash = (np.isin(tkey, gkey) | np.isin(tkey, okey)
                 | np.isin(tkey, so_key))
        t_didx, t_k, tkey = t_didx[~clash], t_k[~clash], tkey[~clash]
        t_bid, t_ask, t_flow = _carry_at(t_didx, tkey)

        n_g, n_o, n_s, n_t = len(g_didx), len(o_didx), len(so_didx), len(t_didx)
        n_rest = n_o + n_s + n_t
        nanf = np.full(n_rest, np.nan)
        zero = np.zeros(n_rest, np.int64)
        R_didx = np.concatenate([g_didx, o_didx, so_didx, t_didx])
        R_k = np.concatenate([g_k, o_k, so_k, t_k])
        R_ts = w_start + R_k * _NS_5MIN
        R_open = np.concatenate([g_open, nanf])
        R_high = np.concatenate([g_high, nanf])
        R_low = np.concatenate([g_low, nanf])
        R_close = np.concatenate([g_close, nanf])
        R_vol = np.concatenate([g_vol, zero])
        R_buy = np.concatenate([g_buy, zero])
        R_sell = np.concatenate([g_sell, zero])
        R_bid = np.concatenate([g_bid, o_bid, so_bid, t_bid])
        R_ask = np.concatenate([g_ask, o_ask, so_ask, t_ask])
        R_fresh = np.concatenate([np.ones(n_g, bool), np.zeros(n_rest, bool)])
        R_flow_raw = np.concatenate([g_cumflow, o_flow, so_flow, t_flow]) \
            + flow_cum[R_didx]
        R_flow_ok = defs.act[R_didx] >= tape_start

        # spread columns: exact (contract, bucket) match into the leg groups;
        # spread session-cum as-of the bucket for flow/estimate
        row_key = R_didx * (n_k + 1) + R_k
        R_spvol = np.zeros(len(R_didx), np.int64)
        R_spbuy = np.zeros(len(R_didx), np.int64)
        R_spsell = np.zeros(len(R_didx), np.int64)
        if len(skey):
            pos = np.searchsorted(skey, row_key)
            pos_c = np.clip(pos, 0, len(skey) - 1)
            hit = (pos < len(skey)) & (skey[pos_c] == row_key)
            R_spvol[hit] = s_vol[pos_c[hit]]
            R_spbuy[hit] = s_buy[pos_c[hit]]
            R_spsell[hit] = s_sell[pos_c[hit]]
        R_sp_cum = _sp_cum_at(R_didx, row_key)
        R_flow_sp_raw = R_sp_cum + flow_cum_sp[R_didx]

        # OI per row, as-of publication: previous session's rolling value
        # until the evening print's ts, then evening, then morning once its
        # print has been published (is_morning_update flips at that ts)
        eve_v = oi_val[:defs.n].copy()
        eve_t = np.full(defs.n, np.iinfo(np.int64).min, dtype=np.int64)
        if eve:
            eve_v[e_rows] = e_vals
            eve_t[e_rows] = e_tss
        morn_v = np.zeros(defs.n, dtype=np.int64)
        morn_t = np.full(defs.n, _I64_MAX, dtype=np.int64)
        if morn:
            morn_v[m_rows] = m_vals
            morn_t[m_rows] = m_tss
        bucket_end = R_ts + _NS_5MIN
        is_morning = bucket_end > morn_t[R_didx]
        R_oi = np.where(is_morning, morn_v[R_didx],
                        np.where(bucket_end > eve_t[R_didx],
                                 eve_v[R_didx], oi_val[R_didx]))

        # daily_estimated_oi: latest published OI adjusted by the tape's net
        # aggressor volume (buy adds, sell subtracts) accumulated since the
        # session the published number describes.  Once this session's print
        # lands (as-of prev session close = this session's start), the base
        # snaps to it and only THIS session's net applies; before it lands,
        # the base is the older print, so the previous session's net counts
        # too.  Contracts without a print this session keep the older base
        # (+INF switch time keeps them on the pre-print branch all session).
        eve_t_est = np.full(defs.n, _I64_MAX, dtype=np.int64)
        if eve:
            eve_t_est[e_rows] = e_tss
        # session net(buy - sell): outright tape + attributed spread legs
        R_net = -(R_flow_raw - flow_cum[R_didx]) - R_sp_cum
        R_est = R_oi + R_net + np.where(bucket_end > eve_t_est[R_didx],
                                        0, sess_net[R_didx])
        if n_t:                       # tombstones: the -1 sentinel, like A
            R_oi[-n_t:] = -1
            R_est[-n_t:] = -1
            is_morning[-n_t:] = False
            n_tombstones += n_t

        R_sig = _significance_rows(ku_by_didx[R_didx], always_by_didx[R_didx],
                                   defs.exp[R_didx], R_ts, cal)
        R_flow = np.where(R_flow_ok, R_flow_raw.astype(np.float64), np.nan)
        R_flow_sp = np.where(R_flow_ok, R_flow_sp_raw.astype(np.float64),
                             np.nan)

        if drop_insignificant:
            keep = R_sig
            n_dropped += int((~keep).sum())
            (R_didx, R_ts, R_open, R_high, R_low, R_close, R_vol, R_buy,
             R_sell, R_bid, R_ask, R_fresh, R_oi, R_est, is_morning, R_flow,
             R_flow_sp, R_spvol, R_spbuy, R_spsell, R_sig) = (a[keep] for a in (
                 R_didx, R_ts, R_open, R_high, R_low, R_close, R_vol, R_buy,
                 R_sell, R_bid, R_ask, R_fresh, R_oi, R_est, is_morning,
                 R_flow, R_flow_sp, R_spvol, R_spbuy, R_spsell, R_sig))

        # update rolling per-contract state AFTER assembling (session end)
        if len(e_rows):
            oi_val[e_rows] = e_vals
            oi_is_morning[e_rows] = False
        if len(m_rows):
            oi_val[m_rows] = m_vals
            oi_is_morning[m_rows] = True
        _fold_into(flow_cum, didx, signed, ts, w_start, w_end)
        folded_upto = w_end
        lastg = np.zeros(len(g_didx), dtype=bool)
        if len(g_didx):
            lastg[:-1] = g_didx[1:] != g_didx[:-1]
            lastg[-1] = True
            lg = np.nonzero(lastg)[0]
            # quote state updated as a PAIR from the same last record — never
            # mix a fresh bid with an older ask (that fabricates crossed books)
            q_bid[g_didx[lg]] = g_bid[lg]
            q_ask[g_didx[lg]] = g_ask[lg]
        # this session's total net (buy - sell) per contract — outright PLUS
        # attributed spread legs — used by the next session's pre-print
        # estimated-OI branch
        sess_net[:defs.n] = 0
        if len(g_didx):
            sess_net[g_didx[lg]] = -g_cumflow[lg]
        if len(s_didx):
            lastl = np.zeros(len(s_didx), dtype=bool)
            lastl[:-1] = s_didx[1:] != s_didx[:-1]
            lastl[-1] = True
            ls = np.nonzero(lastl)[0]
            np.add.at(sess_net, s_didx[ls], -s_cum[ls])
        _fold_into(flow_cum_sp, L_didx, L_signed, L_ts, w_start, w_end)
        sessions_done = day

        if skip_existing and day in existing_days:
            n_skipped += 1
            return

        order2 = np.lexsort((defs.iid[R_didx], R_ts))  # iid: run-independent order
        index = pd.DatetimeIndex(R_ts[order2], tz="UTC").tz_convert(_NY)
        index.name = "timestamp"
        frame = pd.DataFrame({
            "instrument_id": defs.iid[R_didx[order2]].astype(np.int32),
            "open": R_open[order2], "high": R_high[order2],
            "low": R_low[order2], "close": R_close[order2],
            "volume": R_vol[order2].astype(np.int32),
            "buy_volume": R_buy[order2].astype(np.int32),
            "sell_volume": R_sell[order2].astype(np.int32),
            "spread_volume": R_spvol[order2].astype(np.int32),
            "spread_buy_volume": R_spbuy[order2].astype(np.int32),
            "spread_sell_volume": R_spsell[order2].astype(np.int32),
            "bid": R_bid[order2], "ask": R_ask[order2],
            "is_stale_quote": ~R_fresh[order2],
            "open_interest": R_oi[order2].astype(np.int32),
            "is_morning_update": is_morning[order2],
            "daily_estimated_oi": R_est[order2].astype(np.int32),
            "dealer_oi_flow": R_flow[order2],
            "dealer_oi_flow_spread": R_flow_sp[order2],
            _SIG_COL: R_sig[order2],
        }, index=index)
        if drop_insignificant:
            frame = frame.drop(columns=_SIG_COL)
        _write_session(day, frame, np.unique(R_didx))

    # ── pooled decode + ordered consumption ───────────────────────────────────
    legs_files = []
    if folders["legs"] is not None:
        legs_files = sorted(folders["legs"].glob("*.dbn.zst"), key=_file_day)
        older_l = [f for f in legs_files if _file_day(f) < first_day_needed]
        legs_files = [f for f in legs_files
                      if first_day_needed <= _file_day(f) <= range_hi]
        if older_l:
            legs_files = [older_l[-1]] + legs_files
    else:
        progress(0, total, "NOTE: no DBN v3 definitions folder found — "
                           "spread-leg attribution disabled (spread columns "
                           "will be zero)")

    tasks = ([("defs", p, _file_day(p)) for p in def_files]
             + [("legs", p, _file_day(p)) for p in legs_files]
             + [("oi", p, _file_day(p)) for p in stats_files]
             + [("tbbo", p, _file_day(p)) for p in tbbo_todo])
    kind_rank = {"defs": 0, "legs": 1, "oi": 2, "tbbo": 3}
    tasks.sort(key=lambda t_: (t_[2], kind_rank[t_[0]]))

    n_workers = max(1, min(15, (os.cpu_count() or 8) - 1))
    ex = None
    if n_workers > 1 and len(tasks) > 4:
        try:
            worker_fn = _plant_worker()
            ex = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=runpy.run_path,
                initargs=(str(Path(__file__).resolve()),))
        except Exception as e:
            progress(0, total, f"WARN: process pool unavailable ({e}) — "
                               f"running sequentially")
            ex = None

    window = n_workers * 3
    futs: dict[int, object] = {}
    submitted = 0
    sess_seen = 0
    session_days = {s[0] for s in todo}
    try:
        for i, (kind, path, day) in enumerate(tasks):
            if ex is not None:
                while submitted < len(tasks) and submitted <= i + window:
                    k2, p2, _d2 = tasks[submitted]
                    futs[submitted] = ex.submit(worker_fn, k2, str(p2))
                    submitted += 1
            t = perf_counter()
            try:
                if ex is not None:
                    try:
                        res = futs.pop(i).result()
                    except BrokenProcessPool:
                        progress(sess_seen, total,
                                 "WARN: worker pool died — continuing "
                                 "sequentially")
                        ex.shutdown(wait=False, cancel_futures=True)
                        ex, futs = None, {}
                        res = _worker(kind, str(path))
                else:
                    res = _worker(kind, str(path))
            except Exception as e:
                times["decode_wait"] += perf_counter() - t
                progress(sess_seen, total, f"WARN: skipped {path.name}: {e}")
                continue
            times["decode_wait"] += perf_counter() - t

            t = perf_counter()
            if kind == "defs":
                if res is not None:
                    defs.apply(res)
                times["defs"] += perf_counter() - t
            elif kind == "legs":
                _apply_legs(res)
                times["defs"] += perf_counter() - t
            elif kind == "oi":
                _extend_static()
                pdx = defs.lookup(res["iid"], day)
                okp = pdx >= 0
                for ref in np.unique(res["ref_day"][okp]):
                    m = okp & (res["ref_day"] == ref)
                    oi_buffer.setdefault(int(ref), []).append(
                        (pdx[m], res["ts"][m], res["qty"][m], res["ny_day"][m]))
                times["oi"] += perf_counter() - t
            else:  # tbbo
                if day in session_days and prev_payload is not None:
                    sess_seen += 1
                    progress(sess_seen, total,
                             f"{pd.Timestamp(day * _NS_PER_DAY).date()}")
                    _process_session(day, prev_payload, res)
                prev_payload = res
                times["session"] += perf_counter() - t
                if sess_seen and sess_seen % 25 == 0:
                    _save_state(out_dir, tape_start, sessions_done,
                                defs.iid[:defs.n], flow_cum[:defs.n],
                                q_bid[:defs.n], q_ask[:defs.n],
                                oi_val[:defs.n], oi_is_morning[:defs.n],
                                seen_b[:defs.n], sess_net[:defs.n],
                                flow_cum_sp[:defs.n])
    except BaseException:
        writer_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

    writer_pool.shutdown(wait=True)
    for f in write_futs:
        f.result()
    if sessions_done > state_through:
        _save_state(out_dir, tape_start, sessions_done,
                    defs.iid[:defs.n], flow_cum[:defs.n],
                    q_bid[:defs.n], q_ask[:defs.n],
                    oi_val[:defs.n], oi_is_morning[:defs.n],
                    seen_b[:defs.n], sess_net[:defs.n],
                    flow_cum_sp[:defs.n])

    notes = []
    if n_tombstones:
        notes.append(f"{n_tombstones:,} expiry tombstones (OI = -1)")
    if n_unknown_iid:
        notes.append(f"{n_unknown_iid:,} trades on instruments without a "
                     f"leg map (unattributed spreads/unknowns) skipped")
    if cal["n_vol"] < cal["n_q"]:
        notes.append(f"volume_roll undeterminable for "
                     f"{cal['n_q'] - cal['n_vol']} quarterlies — used the "
                     f"scheduled roll alone")
    if drop_insignificant:
        notes.append(f"dropped {n_dropped:,} insignificant rows")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    skipped = f", {n_skipped} already existed" if n_skipped else ""
    progress(total, total, f"✓ Wrote {n_written} session files — "
                           f"{n_rows_total:,} rows{skipped}{suffix}")

    if TIMING:
        for k, v in times.items():
            _tlog(f"{k:>12}  {v:8.2f}s")
