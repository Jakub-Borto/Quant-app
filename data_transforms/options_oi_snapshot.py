"""
options_eod_chain — Transform A: EOD options chain (ES).

Reads GLBX.MDP3 DEFINITION + STATISTICS daily .dbn.zst files and writes ONE
PARQUET PER NY CALENDAR DAY (YYYY-MM-DD.parquet): up to two open-interest
snapshots per contract per session (evening/morning), each row carrying the
PRIOR session's settlement (prefer-final-else-latest) and the underlying's
spot close at/just before the snapshot time.  Rows land in the file of their
snapshot_time's NY date (a session's evening print is dated the session day,
its morning print the next trading day).  No IV / Greeks math — clean inputs.

When a contract expires, ONE tombstone row is written at its (real) expiry
+ 1 second — snapshot_type "expired", open_interest -1, settlement NaN — and
the contract is never mentioned again: the feed's post-expiry OI republications
(which arrive up to ~a day late) are suppressed.  Tombstones are emitted only
for contracts that actually appeared in the OI stream, so never-traded strikes
don't get one.

skip_existing=True resumes incrementally: finished daily files are kept and
only the input tail is reprocessed, always recomputing the trailing ~2 weeks
(days written at the previous run's end may predate their late OI
republications / weekend settlement finals).  Resume only extends forward —
to fill gaps older than the newest existing file, run with skip_existing off.

Each row also carries significant_options_based_on_front_month, evaluated
PER SESSION (Rules 1/2/3 vs the front-month roll calendar; see
_significance_rows): a row is significant when its root is always-tracked, or
its underlying is the current front quarterly that day, or the next front
quarterly with expiry ≥ the upcoming roll (the incoming book captured through
its pre-roll build-up).  So the same contract is significant near the front and
not while it is far-dated — that is what lets PARAMS.drop_insignificant
(default False) actually shrink the output; with it on, the (then all-True)
column is omitted.  volume_roll is read up front from the candle front_month
footers (whole range already on disk), so the plan's "pass 2" is unnecessary.

The input folder may be either the DEFINITION or the STATISTICS folder; the
sibling is resolved automatically from the schema token in the *.dbn.zst
filenames.  The asset is derived from the folder layout
<root>/raw_dbn/<type>/<ASSET>/<dataset>; spot candles come from
<root>/parquet/Futures/<ASSET>/<*ohlcv*>/YYYY-MM-DD.parquet.

Era semantics (verified on the real ES files, 2010-2026):

- modern era (~2018+): OI (stat 9) publishes twice per session with a valid
  ts_ref = the as-of session date: an evening burst ~21-22:00 NY on the ref
  date itself and a morning burst ~09-10:00 NY on the next trading day.
  Weekend re-publications (Sun ~13:00 for the Friday session) carry the same
  ref; per (contract, session, snapshot_type) the LATEST publication wins,
  so a Friday session resolves to its Monday-morning print.
- old era (2010-2017): no evening burst; up to three pre-open republications
  (~02:00/07:00/09:00 NY) and OI ts_ref is an undefined sentinel.  The as-of
  session is inferred as the latest settled session strictly before the
  publication's NY date; again the latest publication wins ("morning" only).
- snapshot_type: "evening" if the publication's NY date <= the as-of session
  date, else "morning".
- settlement (stat 3) is keyed by its own ts_ref (valid in ALL eras) and
  joined on the OI row's as-of session — which IS the plan's prior-session,
  no-look-ahead settlement (finalized on that session's evening, before the
  first OI publication that references it).  Within a session: prefer records
  with the final bit (stat_flags & 1), then latest ts_event.  Friday finals
  arriving in Sat/Sun files are covered by the ts_ref keying.
- multiplier comes from unit_of_measure_qty (contract_multiplier is an
  INT_MAX sentinel on ES options); definitions keep only instrument_class
  C/P outrights (class 'T' = spreads would double-count the chain).
"""

import builtins
import os
import runpy
import struct
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zstandard

TIMING = True

# ── front-month significance classification ───────────────────────────────────
# Transform B mirrors BOTH dicts below and the significance logic; there is no
# shared module (by request), so keep them in sync manually when either changes.
#
# 1a. Series roots that are ALWAYS significant (structural OI: quarterly / EOM /
#     serial families).  Root-membership only, no date logic.
ASSETS_TO_ALWAYS_TRACK = {
    "ES": ("ES", "EW", "EYC"),
}

# 1b. Per-asset quarterly roll / expiry rule.  Quarterly futures expire on the
#     Nth weekday of the quarter months (ES: 3rd Friday, 09:30 ET, AM-settled);
#     the CME scheduled roll is `roll_offset_days` calendar days earlier
#     (Thursday, 8 days before expiry for the equity indices).
ROLL_RULES = {
    "ES": {"quarter_months": (3, 6, 9, 12), "expiry_weekday": 4,
           "expiry_week": 3, "expiry_time": (9, 30), "roll_offset_days": 8},
}
_DEFAULT_ROLL_RULE = ROLL_RULES["ES"]
_MONTHCHAR = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
              7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

# UI param: default keeps every row (research-safe).  True drops rows whose
# significant_options_based_on_front_month is False.
PARAMS = {"drop_insignificant": False}

_NY = "America/New_York"
_NS_PER_DAY = 86_400_000_000_000
_I64_MAX = np.iinfo(np.int64).max
# ts_ref sanity window: outside [2005, 2200) it's a sentinel (0 / -1 / i64max).
_TS_VALID_LO = pd.Timestamp("2005-01-01").value
_TS_VALID_HI = pd.Timestamp("2200-01-01").value

# How many calendar days after a session we keep it open for late OI
# republications / weekend settlement finals before flushing its rows.
_FLUSH_LAG_DAYS = 7


def _tlog(msg: str) -> None:
    if TIMING:
        print(f"[TIMING] {msg}", flush=True)


# ── fast DBN decode ───────────────────────────────────────────────────────────
# Every file in these datasets is DBN v1 with fixed-width records; decoding is
# a zstd stream + np.frombuffer, skipping databento's metadata (symbology)
# parse entirely (~15ms/file we never use).  Anything unexpected falls back to
# databento's own decoder.

_DBN_V1_STAT_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"),
    ("instrument_id", "<u4"), ("ts_event", "<u8"), ("ts_recv", "<u8"),
    ("ts_ref", "<u8"), ("price", "<i8"), ("quantity", "<i4"),
    ("sequence", "<u4"), ("ts_in_delta", "<i4"), ("stat_type", "<u2"),
    ("channel_id", "<u2"), ("update_action", "u1"), ("stat_flags", "u1"),
    ("_reserved", "S6"),
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
                  "definition": _DBN_V1_DEF_DTYPE}


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
    import databento as db  # fallback only — workers usually never import it
    return db.DBNStore.from_file(str(path)).to_ndarray()


# ── folder / file resolution ──────────────────────────────────────────────────

def _schema_token(folder: Path) -> str | None:
    """'definition' / 'statistics' from the first *.dbn.zst filename, else None."""
    for f in folder.glob("*.dbn.zst"):
        parts = f.name.split(".")
        if len(parts) >= 3:
            return parts[-3].lower()
        return None
    return None


def _resolve_folders(input_folder: Path) -> tuple[Path, Path]:
    """Input may be the DEFINITION or STATISTICS folder — return (defs, stats).

    The sibling is searched in the parent (asset) folder by schema token; when
    several siblings hold the same schema the one with the most files wins
    (daily datasets over monthly leftovers).
    """
    token = _schema_token(input_folder)
    if token not in ("definition", "statistics"):
        raise ValueError(
            f"{input_folder} does not contain .definition/.statistics .dbn.zst files"
        )
    want = "statistics" if token == "definition" else "definition"

    candidates = []
    for sib in input_folder.parent.iterdir():
        if sib.is_dir() and sib != input_folder and _schema_token(sib) == want:
            candidates.append((len(list(sib.glob("*.dbn.zst"))), sib))
    if not candidates:
        raise FileNotFoundError(
            f"No sibling {want.upper()} folder found next to {input_folder}"
        )
    sibling = max(candidates)[1]

    if token == "definition":
        return input_folder, sibling
    return sibling, input_folder


def _find_candle_folder(input_folder: Path) -> tuple[Path | None, str]:
    """(candle folder or None, asset). Root = ancestor holding 'raw_dbn'."""
    asset = input_folder.parent.name
    root = None
    for anc in input_folder.parents:
        if anc.name == "raw_dbn":
            root = anc.parent
            break
    roots = [root] if root is not None else []
    if not roots:
        try:  # fall back to the app's configured data roots
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
            # several *ohlcv* folders -> the one with the most daily files
            best = max(matches, key=lambda d: len(list(d.glob("*.parquet"))))
            return best, asset
    return None, asset


def _file_day(path: Path) -> int:
    """glbx-mdp3-YYYYMMDD.<schema>.dbn.zst -> days since epoch (UTC)."""
    ymd = path.name.split(".")[0].split("-")[-1]
    return int(pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}").value // _NS_PER_DAY)


# ── definitions ───────────────────────────────────────────────────────────────

def _bytes_hash(a: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit polynomial hash of a fixed-width 'S' field."""
    a = np.ascontiguousarray(a)  # structured-array fields are strided views
    m = a.view(np.uint8).reshape(len(a), -1).astype(np.uint64)
    h = np.zeros(len(a), dtype=np.uint64)
    for j in range(m.shape[1]):
        h = h * np.uint64(1099511628211) + m[:, j]
    return h.view(np.int64)


def _load_defs(path: Path) -> dict | None:
    """Decode a definitions file to sorted-by-iid arrays of C/P outrights."""
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
    cp = (a["instrument_class"] == b"C").astype(np.int64)  # 1=call, 0=put
    sig = (
        exp
        ^ (strike * 1000003)
        ^ (act >> 1)
        ^ (mult << 7)
        ^ (cp << 3)
        ^ _bytes_hash(a["underlying"])
        ^ _bytes_hash(a["asset"])
    )
    return {
        "iid": iid, "sig": sig, "exp": exp, "act": act, "strike": strike,
        "mult": mult, "cp": cp, "underlying": a["underlying"], "series": a["asset"],
    }


class _DefStore:
    """Chronological instrument_id -> static-facts store.

    Definitions arrive as full daily snapshots; day-over-day (iid, sig)
    comparison means only new/changed contracts hit the Python loop
    (a few hundred per day).

    Instrument ids are REUSED within days: a weekly option expires Wednesday
    and the same id is re-listed (pre-activation) for the next month's weekly
    while the expired contract's final OI prints are still arriving.  Each id
    therefore keeps its full definition history, and lookups resolve as-of a
    session: the newest definition whose activation does not postdate the
    session wins (falling back to the newest overall for stamping quirks).
    """

    def __init__(self):
        # iid -> newest def row as a direct-address table (ES option ids top
        # out around 43M -> ~170MB int32, and scatter/gather beats any sorted
        # structure).  Only ids that get redefined (a few % — weekly relists)
        # keep a full history list in the dict.
        self.last_row = np.full(1 << 20, -1, dtype=np.int32)
        self.hist: dict[int, list[int]] = {}
        # per-row static facts, capacity-doubled arrays (rows are append-only)
        self.n = 0
        cap = 1 << 16
        self.exp = np.empty(cap, dtype=np.int64)
        self.act = np.empty(cap, dtype=np.int64)
        self.strike = np.empty(cap, dtype=np.int64)
        self.mult = np.empty(cap, dtype=np.int64)
        self.cp = np.empty(cap, dtype=np.int8)
        self.series_code = np.empty(cap, dtype=np.int32)
        self.und_code = np.empty(cap, dtype=np.int32)
        self.iid = np.empty(cap, dtype=np.int64)         # contract identity
        self.seen = np.zeros(cap, dtype=bool)            # appeared in OI stream
        self._series_ids: dict[str, int] = {}
        self._und_ids: dict[str, int] = {}
        self.series_cats: list[str] = []
        self.und_cats: list[str] = []
        self._prev: dict | None = None       # yesterday's decoded file

    def _ensure(self, n: int) -> None:
        if n <= len(self.exp):
            return
        cap = max(n, 2 * len(self.exp))
        for name in ("exp", "act", "strike", "mult", "cp",
                     "series_code", "und_code", "iid", "seen"):
            old = getattr(self, name)
            grown = (np.zeros if old.dtype == bool else np.empty)(cap, old.dtype)
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

        # redefined ids (relists) get a history list; everything else is
        # covered by the vectorized last_row table alone
        old = self.last_row[new_iid]
        for k in np.nonzero(old >= 0)[0]:
            iid_k = int(new_iid[k])
            self.hist.setdefault(iid_k, [int(old[k])]).append(int(new_row[k]))
        self.last_row[new_iid] = new_row
        self._prev = d

    def lookup(self, iids: np.ndarray, session_day: int) -> np.ndarray:
        """Def row per iid, as-of the session (-1 when the iid is unknown).

        Per iid: the newest history row whose activation does not postdate
        the session; if none qualifies, the newest row overall.
        """
        act_cut = (session_day + 1) * _NS_PER_DAY  # activation on session day is ok
        safe = np.minimum(iids, len(self.last_row) - 1)
        out = np.where(iids < len(self.last_row),
                       self.last_row[safe], np.int32(-1)).astype(np.int64)

        # newest def activates in the future (a relist whose predecessor's
        # final prints are still arriving) -> walk that id's history backwards
        redo = out >= 0
        redo[redo] = self.act[out[redo]] >= act_cut
        if redo.any():
            act = self.act
            for k in np.nonzero(redo)[0]:
                hist = self.hist.get(int(iids[k]))
                if hist is None:
                    continue                    # single def -> keep it
                for r in reversed(hist):
                    if act[r] < act_cut:
                        out[k] = r
                        break
        return out


# ── statistics ────────────────────────────────────────────────────────────────

def _ny_day(ts_ns: np.ndarray) -> np.ndarray:
    """UTC epoch-ns -> NY-local day number (days since epoch)."""
    idx = pd.DatetimeIndex(ts_ns, tz="UTC").tz_convert(_NY)
    return idx.tz_localize(None).asi8 // _NS_PER_DAY


def _load_stats(path: Path) -> tuple[dict, dict]:
    """Decode one statistics file -> (settlements, open interest) arrays."""
    arr = _decode_dbn(path, "statistics")
    st = arr["stat_type"]

    s = arr[st == 3]
    ref = s["ts_ref"].astype(np.int64)
    ok = (ref > _TS_VALID_LO) & (ref < _TS_VALID_HI)
    s, ref = s[ok], ref[ok]
    price_raw = s["price"].astype(np.int64)
    settle = {
        "iid": s["instrument_id"].astype(np.int64),
        "ref_day": ref // _NS_PER_DAY,
        "price": np.where(price_raw == _I64_MAX, np.nan, price_raw / 1e9),
        "final": (s["stat_flags"] & 1).astype(bool),
        "ts": s["ts_event"].astype(np.int64),
        "n_bad_ref": int((~ok).sum()),
    }
    # pre-dedup per (session, contract) in the worker: keep the max-rank
    # record (final bit, then latest ts).  Max is associative, so the final
    # cross-file merge at flush picks the same record either way.
    if len(settle["iid"]):
        rank = settle["ts"] + (settle["final"].astype(np.int64) << 62)
        order = np.lexsort((rank, settle["iid"], settle["ref_day"]))
        last = np.ones(len(order), dtype=bool)
        s_iid = settle["iid"][order]
        s_day = settle["ref_day"][order]
        last[:-1] = (s_iid[1:] != s_iid[:-1]) | (s_day[1:] != s_day[:-1])
        keep = order[last]
        for k in ("iid", "ref_day", "price", "final", "ts"):
            settle[k] = settle[k][keep]

    o = arr[st == 9]
    ref = o["ts_ref"].astype(np.int64)
    ts = o["ts_event"].astype(np.int64)
    oi = {
        "iid": o["instrument_id"].astype(np.int64),
        "ref_valid": (ref > _TS_VALID_LO) & (ref < _TS_VALID_HI),
        "ref_day": ref // _NS_PER_DAY,
        "ts": ts,
        "ny_day": _ny_day(ts),  # computed here so pool workers do the tz math
        "qty": o["quantity"].astype(np.int64),
    }
    return settle, oi


class _SessionStore:
    """Per-session accumulation of OI publications and settlement records.

    Sessions are keyed by as-of day (int days since epoch) and flushed once
    the file cursor is _FLUSH_LAG_DAYS past them, so weekend republications
    and late settlement finals are all in before rows are emitted.
    """

    def __init__(self, defs: _DefStore):
        self.defs = defs
        self.oi: dict[int, list] = {}        # day -> [(iid, typ, ts, qty), ...]
        self.settle: dict[int, list] = {}    # day -> [(iid, rank, price), ...]
        self.settled_days: list[int] = []    # sorted, for old-era ref inference
        # finished rows, binned by NY calendar day (-> one output file each)
        self.bins: dict[int, list[tuple]] = {}
        self.n_missing_def = 0
        self.n_unresolved_ref = 0
        self.n_bad_settle_ref = 0
        self.n_post_expiry = 0

    def ready_days(self, force: bool = False) -> list[int]:
        """Bins no future session can still add rows to (all, when forced).

        A session S only emits rows dated >= S (its evening print is dated S),
        so once every open session is > D, day D is final.
        """
        if force or not self.oi:
            return sorted(self.bins)
        open_min = min(self.oi)
        return [d for d in sorted(self.bins) if d < open_min]

    # ── ingest ────────────────────────────────────────────────────────────────
    def add_settle(self, s: dict) -> None:
        self.n_bad_settle_ref += s["n_bad_ref"]
        if not len(s["iid"]):
            return
        # rank: final bit dominates, then latest ts_event (ts < 2^62 -> no overflow)
        rank = s["ts"] + (s["final"].astype(np.int64) << 62)
        for day in np.unique(s["ref_day"]):
            m = s["ref_day"] == day
            self.settle.setdefault(int(day), []).append(
                (s["iid"][m], rank[m], s["price"][m]))
            day = int(day)
            i = np.searchsorted(self.settled_days, day)
            if i == len(self.settled_days) or self.settled_days[i] != day:
                self.settled_days.insert(i, day)

    def add_oi(self, o: dict) -> None:
        if not len(o["iid"]):
            return
        ny = o["ny_day"]
        session = o["ref_day"].copy()
        bad = ~o["ref_valid"]
        if bad.any():
            # old era: as-of session = latest settled session before the NY pub date
            days = np.asarray(self.settled_days, dtype=np.int64)
            pos = np.searchsorted(days, ny[bad], side="left") - 1
            inferred = np.where(pos >= 0, days[np.clip(pos, 0, None)], -1)
            session[bad] = inferred
            unresolved = session == -1
            if unresolved.any():
                self.n_unresolved_ref += int(unresolved.sum())
                keep = ~unresolved
                o = {k: (v[keep] if isinstance(v, np.ndarray) else v)
                     for k, v in o.items()}
                ny, session = ny[keep], session[keep]
                if not len(session):
                    return
        typ = (ny > session).astype(np.int8)  # 0=evening, 1=morning
        for day in np.unique(session):
            m = session == day
            self.oi.setdefault(int(day), []).append(
                (o["iid"][m], typ[m], o["ts"][m], o["qty"][m]))

    # ── flush ─────────────────────────────────────────────────────────────────
    def flush_older_than(self, day_cutoff: int) -> None:
        for day in sorted(self.oi):
            if day >= day_cutoff:
                break
            self._flush(day)
        for day in [d for d in self.settle if d < day_cutoff]:
            del self.settle[day]  # settle sessions that never saw OI

    def flush_all(self) -> None:
        for day in sorted(self.oi):
            self._flush(day)
        self.settle.clear()

    def _flush(self, day: int) -> None:
        batches = self.oi.pop(day)
        iid = np.concatenate([b[0] for b in batches])
        typ = np.concatenate([b[1] for b in batches])
        ts = np.concatenate([b[2] for b in batches])
        qty = np.concatenate([b[3] for b in batches])

        # latest publication wins per (snapshot_type, contract)
        order = np.lexsort((ts, iid, typ))
        iid, typ, ts, qty = iid[order], typ[order], ts[order], qty[order]
        last = np.ones(len(iid), dtype=bool)
        last[:-1] = (iid[1:] != iid[:-1]) | (typ[1:] != typ[:-1])
        iid, typ, ts, qty = iid[last], typ[last], ts[last], qty[last]

        # static facts, resolved as-of this session (instrument ids get reused)
        didx = self.defs.lookup(iid, day)
        known = didx >= 0
        self.n_missing_def += int((~known).sum())
        if not known.all():
            iid, typ, ts, qty, didx = (
                iid[known], typ[known], ts[known], qty[known], didx[known])
        self.defs.seen[didx] = True   # this contract appeared in the OI stream

        # drop prints AFTER the contract's expiry (the feed republishes the
        # expiry session's OI up to ~a day later) — each expired contract is
        # represented exactly once, by its open_interest = -1 tombstone row
        alive = ts <= self.defs.exp[didx]
        if not alive.all():
            self.n_post_expiry += int((~alive).sum())
            iid, typ, ts, qty, didx = (
                iid[alive], typ[alive], ts[alive], qty[alive], didx[alive])
        if not len(iid):
            self.settle.pop(day, None)
            return

        # prior-session settlement: prefer final bit, then latest ts_event
        sp = np.full(len(iid), np.nan)
        sf = np.zeros(len(iid), dtype=bool)
        sb = self.settle.pop(day, None)
        if sb:
            s_iid = np.concatenate([b[0] for b in sb])
            s_rank = np.concatenate([b[1] for b in sb])
            s_price = np.concatenate([b[2] for b in sb])
            order = np.lexsort((s_rank, s_iid))
            s_iid, s_rank, s_price = s_iid[order], s_rank[order], s_price[order]
            last = np.ones(len(s_iid), dtype=bool)
            last[:-1] = s_iid[1:] != s_iid[:-1]
            s_iid, s_rank, s_price = s_iid[last], s_rank[last], s_price[last]

            pos = np.searchsorted(s_iid, iid)
            pos_c = np.clip(pos, 0, len(s_iid) - 1)
            hit = s_iid[pos_c] == iid
            sp[hit] = s_price[pos_c[hit]]
            sf[hit] = (s_rank[pos_c[hit]] >> 62) & 1

        ny = _ny_day(ts)
        for d_ in np.unique(ny):
            m = ny == d_
            self.bins.setdefault(int(d_), []).append(
                (ts[m], typ[m], didx[m].astype(np.int32), qty[m], sp[m], sf[m]))


# ── spot ──────────────────────────────────────────────────────────────────────

def _load_candle_closes(folder: Path,
                        from_day: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """1m closes in the folder as (utc_ns sorted, close), optionally only
    files dated >= from_day (incremental runs need just the tail)."""
    files = [f for f in sorted(folder.glob("*.parquet")) if f.stem[0].isdigit()]
    if from_day is not None:
        cutoff = str(pd.Timestamp(from_day * _NS_PER_DAY).date())
        files = [f for f in files if f.stem >= cutoff]

    def read_one(f: Path):
        df = pd.read_parquet(f, columns=["close"])
        idx = df.index
        if idx.tz is None:  # defensive; candle indexes are tz-aware NY
            idx = idx.tz_localize(_NY)
        # units mix ns/us across files -> normalize to ns
        return idx.as_unit("ns").asi8, df["close"].to_numpy(dtype=np.float64)

    if not files:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    with ThreadPoolExecutor(min(8, len(files))) as ex:  # pyarrow releases the GIL
        parts = list(ex.map(read_one, files))
    ts = np.concatenate([p[0] for p in parts])
    px = np.concatenate([p[1] for p in parts])
    order = np.argsort(ts, kind="stable")
    return ts[order], px[order]


def _attach_spot(snap_ts: np.ndarray, candle_ts: np.ndarray,
                 candle_px: np.ndarray) -> np.ndarray:
    if not len(candle_ts):
        return np.full(len(snap_ts), np.nan)
    pos = np.searchsorted(candle_ts, snap_ts, side="right") - 1
    spot = np.where(pos >= 0, candle_px[np.clip(pos, 0, None)], np.nan)
    return spot


# ── process-pool plumbing ─────────────────────────────────────────────────────
# Plugin files are exec'd without sys.modules registration, so their functions
# cannot be pickled to pool workers by module name.  Instead each worker runs
# this file via runpy.run_path (an importable, picklable initializer); the
# `__name__ == "<run_path>"` guard below then plants the worker entry under
# builtins, where pickle can resolve it on both sides.

_PLANT_NAME = "_options_eod_chain_worker"


def _worker(kind: str, path: str):
    """Decode + reduce one file in a pool worker (returns small arrays)."""
    if kind == "stats":
        return _load_stats(Path(path))
    return _load_defs(Path(path))


def _plant_worker():
    fn = _worker
    fn.__module__ = "builtins"
    fn.__qualname__ = _PLANT_NAME
    setattr(builtins, _PLANT_NAME, fn)
    return fn


if __name__ == "<run_path>":   # executing inside a pool worker's initializer
    _plant_worker()


# ── front-month roll calendar & significance ──────────────────────────────────

def _build_roll_calendar(candle_folder: Path | None, asset: str,
                         y0: int = 2006, y1: int = 2035) -> dict:
    """Quarterly roll calendar for `asset`.

    Futures expiries come from the deterministic rule (Nth weekday of the
    quarter months); `cme_scheduled_roll` = expiry − roll_offset_days; the
    hindsight `volume_roll` is read from candle `front_month` footers (the
    whole range is on disk, so it's available up front — no second pass over
    the option files needed).  `min_roll` = min(volume, scheduled), or the
    scheduled roll alone where volume is undeterminable.
    """
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
            try:                                     # footer only, no data read
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
                # search only near this quarter -> no decade-string collision
                lo = q_exp[k] - 160 * _NS_PER_DAY
                sel = (fdays >= lo) & (fdays < q_exp[k]) & (fcode == q_str[k])
                if sel.any():
                    became[k] = fdays[sel].min()
            q_vol[:-1] = became[1:]          # roll off k = when k+1 became front
            n_vol = int((q_vol >= 0).sum())
    q_min = np.where(q_vol >= 0, np.minimum(q_vol, q_cme), q_cme)
    return {"exp": q_exp, "exp_eff": q_exp.copy(), "mc": q_mc, "str": q_str,
            "cme": q_cme, "min": q_min, "vol": q_vol, "n_vol": n_vol,
            "n_q": len(q_exp)}


def _refine_real_expiries(cal: dict, real_max: np.ndarray) -> None:
    """Snap formula expiries to the REAL futures expiries seen in definitions.

    The 3rd-Friday formula is wrong when the Friday is a holiday (Juneteenth
    2026-06-19: ESM6 actually expired Thursday 06-18 09:30).  The latest option
    expiry mapped to a quarterly IS that future's AM expiry (the ES-root
    quarterly option expires with the future), so accept `real_max` as the
    effective expiry when it sits within [formula − 4d, formula] — holiday
    shifts move expiry exactly one business day earlier, and no other option
    series expires inside that window before the AM quarterly is listed.
    """
    w = ((real_max >= cal["exp"] - 4 * _NS_PER_DAY)
         & (real_max <= cal["exp"]))
    cal["exp_eff"] = np.where(w, real_max, cal["exp"])


def _contract_ku(und_str, exp, cal: dict) -> np.ndarray:
    """Quarterly index per contract: month-code of the underlying, first
    calendar expiry ≥ the option's own expiry (disambiguates the decade).
    -1 when the underlying isn't a quarterly / falls outside the calendar."""
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
    """Per-SESSION significance for a batch of rows (all rules 1/2/3).

    Evaluated at each row's own snapshot time `ts`, so a contract is
    significant only on the sessions when its underlying is actually the
    current or the next front-month quarterly — this is what makes
    drop_insignificant remove the far-dated books.

    Rule 1: always-tracked root.
    Rule 2: underlying is the current front quarterly — the nearest one not yet
            expired — so the expiring front stays significant until its own
            expiry (deterministic, no volume_roll needed).
    Rule 3: underlying is the NEXT front quarterly AND the option's expiry is
            ≥ min(volume_roll, cme_roll) of the current front (the upcoming
            roll) — i.e. the incoming book, captured through its pre-roll
            build-up during the current front's tenure.
    """
    NEG = np.iinfo(np.int64).min
    # current front = nearest quarterly not yet expired at this session's ts
    # (exp_eff = formula expiry snapped to the real, holiday-shifted one)
    cur = np.searchsorted(cal["exp_eff"], ts, side="right")
    have = ku >= 0
    is_cur = have & (ku == cur)
    qmin_cur = np.where(cur < cal["n_q"],
                        cal["min"][np.clip(cur, 0, cal["n_q"] - 1)], NEG)
    is_next = have & (ku == cur + 1) & (exp >= qmin_cur)
    return always_flag | is_cur | is_next


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

    defs_folder, stats_folder = _resolve_folders(Path(input_folder))
    candle_folder, asset = _find_candle_folder(stats_folder)

    out_dir = Path(output_folder)
    def_files = sorted(defs_folder.glob("*.dbn.zst"), key=_file_day)
    stats_files = sorted(stats_folder.glob("*.dbn.zst"), key=_file_day)
    if not stats_files or not def_files:
        progress(1, 1, "ERROR: no .dbn.zst files found")
        return

    # ── incremental resume over daily outputs ─────────────────────────────────
    # skip_existing keeps finished YYYY-MM-DD.parquet files, but the trailing
    # ~2 weeks are always recomputed: sessions near the previous run's end were
    # written before their late OI republications / weekend settlement finals
    # could arrive.  (Resume only extends forward — to fill gaps older than the
    # newest existing file, run with skip_existing off.)
    existing_days = set()
    for f in out_dir.glob("*.parquet"):
        if f.stem[:1].isdigit():
            try:
                existing_days.add(int(pd.Timestamp(f.stem).value // _NS_PER_DAY))
            except ValueError:
                pass
    rewrite_from = None
    if skip_existing and existing_days:
        rewrite_from = max(existing_days) - (_FLUSH_LAG_DAYS + 2)
        input_from = rewrite_from - _FLUSH_LAG_DAYS - 5
        older = [f for f in def_files if _file_day(f) < input_from]
        def_files = [f for f in def_files if _file_day(f) >= input_from]
        if older:  # at least one full definitions snapshot before the start
            def_files = [older[-1]] + def_files
        stats_files = [f for f in stats_files if _file_day(f) >= input_from]
        if not stats_files:
            progress(1, 1, "↷ Everything up to date")
            return

    total = len(stats_files) + 1
    if candle_folder is None:
        progress(0, total, f"WARNING: no *ohlcv* candle folder found for "
                           f"{asset} — spot_{asset.lower()} will be NaN")

    # candle loading is disk/pyarrow-bound — overlap it with the main loop
    candle_from = None if rewrite_from is None else rewrite_from - 5
    spot_pool = ThreadPoolExecutor(1)
    spot_future = (spot_pool.submit(_load_candle_closes, candle_folder,
                                    candle_from)
                   if candle_folder is not None else None)
    spot_pool.shutdown(wait=False)

    # multiplier fill for old-era definitions where unit_of_measure_qty is 0:
    # the multiplier is constant per product, so take the modal populated
    # value from the newest definitions snapshot (50 for ES)
    fill_mult = 0
    try:
        d_last = _load_defs(def_files[-1])
        if d_last is not None and (d_last["mult"] != 0).any():
            vals, counts = np.unique(d_last["mult"][d_last["mult"] != 0],
                                     return_counts=True)
            fill_mult = int(vals[np.argmax(counts)])
    except Exception:
        pass

    # roll calendar for the significance classification (built once, up front:
    # volume_roll comes from candle footers, all on disk, so no second pass over
    # the option files is needed — the flag is a per-contract static property).
    always = ASSETS_TO_ALWAYS_TRACK.get(asset, ())
    cal = _build_roll_calendar(candle_folder, asset)

    defs = _DefStore()
    store = _SessionStore(defs)
    times = {"decode_wait": 0.0, "defs": 0.0, "stats": 0.0, "flush": 0.0,
             "spot": 0.0, "sig": 0.0, "write": 0.0}

    # per-contract STATIC significance inputs (quarterly index + always-track
    # flag), grown as definitions are applied.  The per-session rule itself is
    # applied at write time, using each row's own snapshot date.
    ku_by_didx = np.full(1 << 16, -1, dtype=np.int64)
    always_by_didx = np.zeros(1 << 16, dtype=bool)
    real_max = np.full(cal["n_q"], np.iinfo(np.int64).min, dtype=np.int64)
    tomb_by_day: dict[int, list] = {}   # expiry NY day -> [didx arrays]
    sig_done = 0
    last_task_day = -1                  # newest input day processed so far

    def _extend_static() -> None:
        nonlocal ku_by_didx, always_by_didx, sig_done
        if defs.n <= sig_done:
            return
        t_ = perf_counter()
        lo, hi = sig_done, defs.n
        if hi > len(ku_by_didx):
            cap = max(hi, 2 * len(ku_by_didx))
            gk = np.full(cap, -1, dtype=np.int64); gk[:sig_done] = ku_by_didx[:sig_done]
            ga = np.zeros(cap, dtype=bool); ga[:sig_done] = always_by_didx[:sig_done]
            ku_by_didx, always_by_didx = gk, ga
        und_str = [defs.und_cats[c] for c in defs.und_code[lo:hi]]
        root_str = np.asarray([defs.series_cats[c] for c in defs.series_code[lo:hi]])
        ku = _contract_ku(und_str, defs.exp[lo:hi], cal)
        ku_by_didx[lo:hi] = ku
        always_by_didx[lo:hi] = (np.isin(root_str, np.asarray(always))
                                 if always else False)
        # learn real (holiday-shifted) futures expiries from the definitions;
        # a quarterly's defs appear years before it is ever the front, so its
        # effective expiry is final long before any row near it is written
        okm = ku >= 0
        if okm.any():
            np.maximum.at(real_max, ku[okm], defs.exp[lo:hi][okm])
            _refine_real_expiries(cal, real_max)
        # queue expiry tombstones: bucket new def rows by their expiry NY day
        exp_day = _ny_day(defs.exp[lo:hi])
        rows_new = np.arange(lo, hi, dtype=np.int64)
        for d_ in np.unique(exp_day):
            tomb_by_day.setdefault(int(d_), []).append(rows_new[exp_day == d_])
        sig_done = hi
        times["sig"] += perf_counter() - t_

    # ── per-day writer ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    writer_pool = ThreadPoolExecutor(4)   # to_parquet releases the GIL
    write_futs: list = []
    candle_ts = candle_px = None
    n_written = n_rows = n_skipped = n_backfilled = n_dropped = 0
    n_tombstones = 0

    def _make_tombstones(d_: int):
        """One open_interest = -1 row per OI-seen contract expiring on day d_
        (deduped per contract id, keeping its latest definition row)."""
        nonlocal n_tombstones
        batches = tomb_by_day.pop(d_, None)
        if batches is None:
            return None
        cand = np.concatenate(batches)
        order = np.lexsort((cand, defs.iid[cand]))
        c = cand[order]
        ii = defs.iid[c]
        starts = np.nonzero(np.r_[True, ii[1:] != ii[:-1]])[0]
        seen_any = np.logical_or.reduceat(defs.seen[c], starts)
        latest = c[np.r_[starts[1:] - 1, len(c) - 1]]
        keep = latest[seen_any]
        if not len(keep):
            return None
        n_tombstones += len(keep)
        n = len(keep)
        return (defs.exp[keep] + 1_000_000_000,          # expiry + 1 second
                np.full(n, 2, dtype=np.int8),            # snapshot_type "expired"
                keep.astype(np.int32),
                np.full(n, -1, dtype=np.int64),          # the -1 OI sentinel
                np.full(n, np.nan),
                np.zeros(n, dtype=bool))

    def _drain_bins(force: bool = False) -> None:
        nonlocal candle_ts, candle_px, n_written, n_rows, n_skipped
        nonlocal n_backfilled, n_dropped
        _extend_static()   # ensure every didx we're about to write has ku/always
        days = store.ready_days(force)
        # tombstone-only days (an expiry date whose bin has no OI rows)
        open_min = min(store.oi) if store.oi else None
        ready_tomb = [d for d in tomb_by_day
                      if d <= last_task_day
                      and (force or open_min is None or d < open_min)]
        days = sorted(set(days) | set(ready_tomb))
        if not days:
            return
        if candle_ts is None:
            t_ = perf_counter()
            if spot_future is not None:
                candle_ts, candle_px = spot_future.result()
            else:
                candle_ts = np.empty(0, np.int64)
                candle_px = np.empty(0, np.float64)
            times["spot"] += perf_counter() - t_
        t_ = perf_counter()
        for d_ in days:
            parts = store.bins.pop(d_, [])
            if (rewrite_from is not None and d_ < rewrite_from
                    and d_ in existing_days):
                tomb_by_day.pop(d_, None)
                n_skipped += 1
                continue
            tomb = _make_tombstones(d_)
            if tomb is not None:
                parts.append(tomb)
            if not parts:
                continue
            ts = np.concatenate([p[0] for p in parts])
            order = np.argsort(ts, kind="stable")
            ts = ts[order]
            typ = np.concatenate([p[1] for p in parts])[order]
            didx = np.concatenate([p[2] for p in parts])[order]
            oi_ = np.concatenate([p[3] for p in parts])[order]
            sp_ = np.concatenate([p[4] for p in parts])[order]
            sf_ = np.concatenate([p[5] for p in parts])[order]

            sig = _significance_rows(ku_by_didx[didx], always_by_didx[didx],
                                     defs.exp[didx], ts, cal)
            if drop_insignificant:
                keep = sig
                n_dropped += int((~keep).sum())
                if not keep.all():
                    ts, typ, didx = ts[keep], typ[keep], didx[keep]
                    oi_, sp_, sf_, sig = oi_[keep], sp_[keep], sf_[keep], sig[keep]
                if not len(ts):
                    continue

            mult_raw = defs.mult[didx]
            zero = mult_raw == 0
            if zero.any() and fill_mult:
                n_backfilled += int(zero.sum())
                mult_raw = np.where(zero, fill_mult, mult_raw)
            spot = _attach_spot(ts, candle_ts, candle_px)

            index = pd.DatetimeIndex(ts, tz="UTC").tz_convert(_NY)
            index.name = "snapshot_time"
            df = pd.DataFrame(
                {
                    "snapshot_type": pd.Categorical.from_codes(
                        typ, categories=["evening", "morning", "expired"]),
                    "underlying": pd.Categorical.from_codes(
                        defs.und_code[didx], categories=list(defs.und_cats)),
                    "series": pd.Categorical.from_codes(
                        defs.series_code[didx],
                        categories=list(defs.series_cats)),
                    "expiry": pd.DatetimeIndex(
                        defs.exp[didx], tz="UTC").tz_convert(_NY),
                    "strike": defs.strike[didx] / 1e9,
                    "cp_flag": pd.Categorical.from_codes(
                        defs.cp[didx], categories=["put", "call"]),
                    "multiplier": np.rint(mult_raw / 1e9).astype(np.int32),
                    "open_interest": oi_,
                    "settlement_price": sp_,
                    "settlement_is_final": sf_,
                    "activation_date": pd.DatetimeIndex(
                        defs.act[didx], tz="UTC").tz_convert(_NY),
                    "spot_es": spot,
                    "significant_options_based_on_front_month": sig,
                },
                index=index,
            )
            # with drop on, the column is all-True (redundant) -> drop it
            if drop_insignificant:
                df = df.drop(columns="significant_options_based_on_front_month")
            date_str = str(pd.Timestamp(d_ * _NS_PER_DAY).date())
            write_futs.append(writer_pool.submit(
                df.to_parquet, out_dir / f"{date_str}.parquet",
                engine="pyarrow"))
            n_written += 1
            n_rows += len(df)
        while len(write_futs) > 64:      # bound the backlog, surface errors
            write_futs.pop(0).result()
        times["write"] += perf_counter() - t_

    # One merged, date-ordered task list; a day's definitions land before its
    # statistics, exactly like the sequential apply-defs-then-stats loop.
    tasks = ([("defs", p, _file_day(p)) for p in def_files]
             + [("stats", p, _file_day(p)) for p in stats_files])
    tasks.sort(key=lambda t_: (t_[2], 0 if t_[0] == "defs" else 1))

    # Decode/reduce runs on a process pool; this thread only merges results
    # (and idles waiting on it much of the time, so leave it just one core).
    n_workers = max(1, min(15, (os.cpu_count() or 8) - 1))
    ex = None
    if n_workers > 1 and len(tasks) > 4:
        try:
            worker_fn = _plant_worker()
            ex = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=runpy.run_path,
                initargs=(str(Path(__file__).resolve()),))
        except Exception as e:  # pool unavailable -> sequential fallback
            progress(0, total, f"WARN: process pool unavailable ({e}) — "
                               f"running sequentially")
            ex = None

    window = n_workers * 3
    futs: dict[int, object] = {}
    submitted = 0
    stats_seen = 0
    last_day = None
    try:
        for i, (kind, path, day) in enumerate(tasks):
            if ex is not None:
                while submitted < len(tasks) and submitted <= i + window:
                    k2, p2, _d2 = tasks[submitted]
                    futs[submitted] = ex.submit(worker_fn, k2, str(p2))
                    submitted += 1

            if kind == "stats":
                stats_seen += 1
                progress(stats_seen, total,
                         f"{pd.Timestamp(day * _NS_PER_DAY).date()}")
            last_task_day = max(last_task_day, day)

            # flush BEFORE this day's definitions apply, so emitted rows never
            # see a redefinition (instrument_id reuse) postdating their session
            if day != last_day:
                t = perf_counter()
                store.flush_older_than(day - _FLUSH_LAG_DAYS)
                times["flush"] += perf_counter() - t
                _drain_bins()
                last_day = day

            t = perf_counter()
            try:
                if ex is not None:
                    try:
                        res = futs.pop(i).result()
                    except BrokenProcessPool:
                        progress(stats_seen, total,
                                 "WARN: worker pool died — continuing "
                                 "sequentially")
                        ex.shutdown(wait=False, cancel_futures=True)
                        ex, futs = None, {}
                        res = _worker(kind, str(path))
                else:
                    res = _worker(kind, str(path))
            except Exception as e:
                times["decode_wait"] += perf_counter() - t
                progress(stats_seen, total, f"WARN: skipped {path.name}: {e}")
                continue
            times["decode_wait"] += perf_counter() - t

            t = perf_counter()
            if kind == "defs":
                if res is not None:
                    defs.apply(res)
                times["defs"] += perf_counter() - t
            else:
                settle, oi = res
                store.add_settle(settle)
                store.add_oi(oi)
                times["stats"] += perf_counter() - t
    except BaseException:
        writer_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

    t = perf_counter()
    store.flush_all()
    times["flush"] += perf_counter() - t
    _drain_bins(force=True)

    writer_pool.shutdown(wait=True)
    for f in write_futs:
        f.result()   # surface any writer error

    if not n_written and not n_skipped:
        progress(total, total, "ERROR: no open-interest rows found")
        return

    notes = []
    if store.n_missing_def:
        notes.append(f"{store.n_missing_def:,} OI rows without a definition")
    if store.n_unresolved_ref:
        notes.append(f"{store.n_unresolved_ref:,} OI rows with unresolvable "
                     f"as-of session (dataset start)")
    if store.n_bad_settle_ref:
        notes.append(f"{store.n_bad_settle_ref:,} settlement rows with "
                     f"invalid ts_ref")
    if n_backfilled:
        notes.append(f"{n_backfilled:,} rows had no multiplier — backfilled "
                     f"with the product's modal value")
    if cal["n_vol"] < cal["n_q"]:
        notes.append(f"volume_roll undeterminable for "
                     f"{cal['n_q'] - cal['n_vol']} quarterlies — used the "
                     f"scheduled roll alone")
    if n_tombstones:
        notes.append(f"{n_tombstones:,} expiry tombstones (OI = -1)")
    if store.n_post_expiry:
        notes.append(f"{store.n_post_expiry:,} post-expiry OI prints dropped")
    if drop_insignificant:
        notes.append(f"dropped {n_dropped:,} insignificant rows")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    skipped = f", {n_skipped} already existed" if n_skipped else ""
    progress(total, total, f"✓ Wrote {n_written} daily files — "
                           f"{n_rows:,} rows{skipped}{suffix}")

    if TIMING:
        for k, v in times.items():
            _tlog(f"{k:>10}  {v:8.2f}s")
