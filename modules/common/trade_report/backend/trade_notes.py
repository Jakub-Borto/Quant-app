"""
The trade-notes data model — flattening the strategies' `notes` JSON column
into queryable columns, inferring what each column IS, and evaluating a
grouped AND/OR query over the result.

Pure pandas, NO Qt: the whole point is that the flatten + query semantics are
unit-testable without a QApplication (tests/test_trade_notes.py).

═══════════════════════════════════════════════════════════════════════════
 WHY THIS IS HARDER THAN "json_normalize THE COLUMN"
═══════════════════════════════════════════════════════════════════════════

The notes key universe is OPEN-ENDED and TYPE-INCONSISTENT, verified against
real trades files:

  * ivb_model emits 60+ distinct keys on one run, and its trailing risk script
    generates trail1_*, trail2_*, … per trade, re-prefixing the triggering
    entry finder's ENTIRE note key set. There is no fixed schema to code to.
  * Keys are SPARSE — `trail5_stop` exists on 1 trade of 156.
  * The same key changes type between rows: `absorption_time` is a str for two
    entry finders and a list[str] for two others.
  * Values can be JSON null (vwap_at_entry on a degenerate session).
  * Some strategies (orb, fvg_ifvg) emit NO notes column at all, and an
    optimizer grid that produced zero trades has neither `notes` nor
    `trade_type`. Everything here must degrade, never raise.

So: types are SNIFFED per column, every operator runs on a coerced "view" of
the column rather than on its raw values, and evaluation is defensive by
construction — a condition that cannot be evaluated is dropped or fails
closed, but never propagates an exception into the report's filter chain.

═══════════════════════════════════════════════════════════════════════════
 COLUMN IDS
═══════════════════════════════════════════════════════════════════════════

A trade column's id is its own name. A notes key's id is ALWAYS
"notes.<key>" — even when it doesn't collide — because the key universe is
open-ended and an unprefixed id could silently shadow `direction` or `ticks`.
The DISPLAY label drops the prefix unless that would collide.

That prefix is also the leak guard: flattened columns exist only in
FlatNotes.frame, never on a host's trades frame, so they cannot reach a saved
parquet (see actions_row.DERIVED_COLUMNS, which needs no entry for them).
"""

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .chart_window import _is_timestamp

NOTES_PREFIX = "notes."

# ── column kinds ─────────────────────────────────────────────────────────────
KIND_EMPTY = "empty"
KIND_BOOL = "bool"
KIND_NUMERIC = "numeric"
KIND_TIME = "time"
KIND_TEXT = "text"
KIND_LIST = "list"
KIND_DATETIME = "datetime"

# Strict on purpose. chart_window._is_timestamp is deliberately loose (it
# accepts anything pd.Timestamp parses that contains ':' or '-', so "12-15"
# is a "timestamp"), which is fine for formatting one tile and useless for
# deciding a whole column's type.
_HHMM = re.compile(r"^\s*(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\s*$")


# ══ scalar helpers ═══════════════════════════════════════════════════════════
def _is_null(value) -> bool:
    """NaN/None/NaT for scalars. Containers are never null — an empty list is
    'blank' (see view_blank), which is a different question."""
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def display_text(value) -> str:
    """
    The ONE display rule, shared by the table and by every text/choice
    operator — so "is one of" offers exactly the strings the user can read in
    the cell.

    Kept in lockstep with dataframe_model._fmt (which grew list joining for
    this feature); the only addition here is that lists join recursively.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(display_text(v) for v in value)
    if _is_null(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        return f"{value:g}"
    if isinstance(value, pd.Timestamp):
        if value.tz is None and value == value.normalize():
            return value.date().isoformat()
        return str(value)
    return str(value)


def to_number(value) -> float:
    """Scalar -> float, NaN when it isn't one. Lists are never numbers."""
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return float("nan")
    return float("nan")


def to_minutes(value) -> float:
    """Scalar -> minutes past midnight, NaN when it isn't a time."""
    if isinstance(value, pd.Timestamp):
        return value.hour * 60 + value.minute + value.second / 60.0
    if isinstance(value, str):
        match = _HHMM.match(value)
        if match is None:
            return float("nan")
        hours = int(match.group(1))
        if hours > 23:
            return float("nan")
        seconds = int(match.group(3) or 0)
        return hours * 60 + int(match.group(2)) + seconds / 60.0
    return float("nan")


def to_bool(value):
    """True / False / None (not a boolean). A 1 is NOT True here — a numeric
    column that happens to hold 0/1 gets numeric operators, not is-true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes"):
            return True
        if text in ("false", "no"):
            return False
    return None


def _is_blank(value) -> bool:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return len(value) == 0
    if _is_null(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


# ══ flattening ═══════════════════════════════════════════════════════════════
def _loads(raw):
    """One cell of the notes column -> dict, {} for absent, None for broken."""
    if isinstance(raw, dict):
        return raw
    if _is_null(raw):
        return {}
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception:      # noqa: BLE001 — any parse failure is just "broken"
        return None
    return obj if isinstance(obj, dict) else None


def _canon(value):
    """
    Canonicalise time-looking strings to "HH:MM" ONCE, at flatten time, so
    display, sorting and time comparisons all agree on one representation.

    Stricter than trade_detail.py's formatting, which reuses the loose
    _is_timestamp directly: a ':' is required, so an ISO date like
    "2026-01-05" survives as itself instead of collapsing to "00:00".
    """
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, str) and ":" in value and _is_timestamp(value):
        try:
            return pd.Timestamp(value).strftime("%H:%M")
        except Exception:      # noqa: BLE001
            return value
    return value


@dataclass
class FlatNotes:
    """The notes column, exploded. `frame` holds ONLY notes.* columns and
    shares the source frame's index — the trade columns are never copied."""
    frame: pd.DataFrame
    notes_columns: list = field(default_factory=list)
    kinds: dict = field(default_factory=dict)
    numeric_ok: dict = field(default_factory=dict)
    time_ok: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @classmethod
    def empty(cls, index=None) -> "FlatNotes":
        return cls(pd.DataFrame(index=index if index is not None else pd.Index([])))


def flatten_notes(source: pd.DataFrame | None) -> FlatNotes:
    """Explode `source["notes"]` into one object column per key."""
    if source is None or "notes" not in getattr(source, "columns", ()) \
            or len(source) == 0:
        index = source.index if source is not None else None
        return FlatNotes.empty(index)

    records, broken = [], 0
    for raw in source["notes"].tolist():
        record = _loads(raw)
        if record is None:
            broken += 1
            record = {}
        records.append(record)

    keys: dict = {}                       # first-appearance order
    for record in records:
        for key in record:
            keys.setdefault(key, None)

    # Column-wise, NOT pd.json_normalize: normalize would explode a list value
    # into key.0 / key.1 and lose the "this trade absorbed on 2 bars" fact.
    data = {NOTES_PREFIX + key: [_canon(rec.get(key)) for rec in records]
            for key in keys}
    frame = (pd.DataFrame(data, index=source.index) if data
             else pd.DataFrame(index=source.index))

    notes_columns = list(frame.columns)
    kinds, numeric_ok, time_ok = {}, {}, {}
    for cid in notes_columns:
        kind, num_ok, t_ok = infer_kind(frame[cid])
        kinds[cid] = kind
        numeric_ok[cid] = num_ok
        time_ok[cid] = t_ok

    taken = set(map(str, source.columns))
    labels = {}
    for cid in notes_columns:
        key = cid[len(NOTES_PREFIX):]
        labels[cid] = key if key not in taken else cid
        taken.add(labels[cid])

    warnings = []
    if broken:
        warnings.append(f"{broken} trade(s) had unreadable notes JSON — "
                        f"treated as having no notes.")
    return FlatNotes(frame, notes_columns, kinds, numeric_ok, time_ok,
                     labels, warnings)


def notes_fingerprint(source: pd.DataFrame | None) -> tuple:
    """
    Cache key for a flatten. CONTENT-based, not identity-based: RegimeSection
    .annotate() hands the report a brand-new frame on every source change
    while the notes column is byte-identical, and re-flattening 100k rows for
    that would be pure waste. Hashing 100k strings is ~10ms.
    """
    if source is None:
        return (0, None)
    if "notes" not in getattr(source, "columns", ()):
        return (len(source), None)
    try:
        digest = int(pd.util.hash_pandas_object(source["notes"],
                                                index=False).sum())
    except Exception:      # noqa: BLE001 — an unhashable notes column
        digest = None
    return (len(source), digest)


# ══ type inference ═══════════════════════════════════════════════════════════
def infer_kind(series: pd.Series) -> tuple:
    """
    (kind, numeric_ok, time_ok) sniffed from the non-null values.

    Order matters. LIST wins over everything because of the verified
    absorption_time collision (str for two finders, list[str] for two others)
    — a column that is a list ANYWHERE is a list column, and its scalars are
    treated as one-element lists. BOOL is tested before NUMERIC because
    isinstance(True, int) is True.

    numeric_ok / time_ok are reported separately for text columns, so a key
    holding "1"/"2" still gets > and <, and one holding "10:22"/"n/a" does
    NOT get time comparisons it would silently fail.
    """
    values = [v for v in series.tolist() if not _is_null(v)]
    if not values:
        return KIND_EMPTY, False, False
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in values):
        return KIND_LIST, False, False
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return KIND_BOOL, False, False
    if all(isinstance(v, (int, float, np.integer, np.floating))
           and not isinstance(v, (bool, np.bool_)) for v in values):
        return KIND_NUMERIC, True, False
    if all(isinstance(v, str) for v in values) \
            and all(not math.isnan(to_minutes(v)) for v in values):
        return KIND_TIME, False, True
    numeric_ok = all(not math.isnan(to_number(v)) for v in values)
    time_ok = all(not math.isnan(to_minutes(v)) for v in values)
    return KIND_TEXT, numeric_ok, time_ok


def column_kind(series: pd.Series) -> tuple:
    """Kind of a real trade column — dtype first, sniffing only as fallback."""
    if pd.api.types.is_bool_dtype(series):
        return KIND_BOOL, False, False
    if pd.api.types.is_numeric_dtype(series):
        return KIND_NUMERIC, True, False
    if pd.api.types.is_datetime64_any_dtype(series):
        return KIND_DATETIME, False, True
    return infer_kind(series)


# ══ coerced views ════════════════════════════════════════════════════════════
VIEW_NUMBER = "number"
VIEW_MINUTES = "minutes"
VIEW_TEXT = "text"
VIEW_LISTS = "lists"
VIEW_BLANK = "blank"
VIEW_BOOL = "bool"


def _view(series: pd.Series, name: str) -> pd.Series:
    index = series.index
    if name == VIEW_NUMBER:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(float)
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        return pd.Series([to_number(v) for v in series], index=index,
                         dtype=float)
    if name == VIEW_MINUTES:
        if pd.api.types.is_datetime64_any_dtype(series):
            return (series.dt.hour * 60 + series.dt.minute
                    + series.dt.second / 60.0).astype(float)
        return pd.Series([to_minutes(v) for v in series], index=index,
                         dtype=float)
    if name == VIEW_TEXT:
        return pd.Series([display_text(v) for v in series], index=index,
                         dtype=object)
    if name == VIEW_LISTS:
        return pd.Series([list(v) if isinstance(v, (list, tuple, np.ndarray))
                          else ([] if _is_blank(v) else [v]) for v in series],
                         index=index, dtype=object)
    if name == VIEW_BLANK:
        return pd.Series([_is_blank(v) for v in series], index=index,
                         dtype=bool)
    if name == VIEW_BOOL:
        return pd.Series([to_bool(v) for v in series], index=index,
                         dtype=object)
    raise KeyError(name)


class ColumnSource:
    """
    Reads a column id from EITHER the trades frame or the flattened notes,
    memoizing the coerced views. Nothing is merged: the flat frame stays
    separate so flattened columns can never leak into a host's frame.
    """

    def __init__(self, trades: pd.DataFrame | None,
                 flat: FlatNotes | None = None):
        self._trades = trades if trades is not None else pd.DataFrame()
        self._flat = flat if flat is not None else FlatNotes.empty(
            self._trades.index)
        self._cache: dict = {}
        self._kinds: dict = {}

    # ── structure ─────────────────────────────────────────────────────────────
    @property
    def index(self):
        return self._trades.index

    @property
    def trade_columns(self) -> list:
        return [str(c) for c in self._trades.columns]

    @property
    def notes_columns(self) -> list:
        return list(self._flat.notes_columns)

    @property
    def warnings(self) -> list:
        return list(self._flat.warnings)

    def has(self, cid: str) -> bool:
        return self.series(cid) is not None

    def label(self, cid: str) -> str:
        return self._flat.labels.get(cid, cid)

    def series(self, cid: str):
        if cid in self._flat.frame.columns:
            return self._flat.frame[cid]
        if cid in self._trades.columns:
            return self._trades[cid]
        return None

    def kind(self, cid: str) -> str:
        return self.kind_info(cid)[0]

    def kind_info(self, cid: str) -> tuple:
        if cid in self._flat.kinds:
            return (self._flat.kinds[cid], self._flat.numeric_ok.get(cid, False),
                    self._flat.time_ok.get(cid, False))
        if cid not in self._kinds:
            series = self.series(cid)
            self._kinds[cid] = ((KIND_EMPTY, False, False) if series is None
                                else column_kind(series))
        return self._kinds[cid]

    # ── data ──────────────────────────────────────────────────────────────────
    def view(self, cid: str, name: str):
        key = (cid, name)
        if key not in self._cache:
            series = self.series(cid)
            if series is None:
                return None
            self._cache[key] = _view(series, name)
        return self._cache[key]

    def choices(self, cid: str, limit: int = 500) -> tuple:
        """(values, total) for an 'is one of' picker — the display strings."""
        view = self.view(cid, VIEW_TEXT)
        if view is None:
            return [], 0
        values = sorted({v for v in view if v != ""})
        return values[:limit], len(values)


# ══ operators ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OpSpec:
    op: str
    label: str
    arity: int          # 0 | 1 | 2
    editor: str         # none | number | range | time | time_range | text
                        # | choices | count


_SPECS = (
    OpSpec("gt", ">", 1, "number"),
    OpSpec("ge", "≥", 1, "number"),
    OpSpec("lt", "<", 1, "number"),
    OpSpec("le", "≤", 1, "number"),
    OpSpec("eq", "=", 1, "number"),
    OpSpec("ne", "≠", 1, "number"),
    OpSpec("between", "between", 2, "range"),

    OpSpec("time_gt", "after", 1, "time"),
    OpSpec("time_ge", "at or after", 1, "time"),
    OpSpec("time_lt", "before", 1, "time"),
    OpSpec("time_le", "at or before", 1, "time"),
    OpSpec("time_eq", "at", 1, "time"),
    OpSpec("time_ne", "not at", 1, "time"),
    OpSpec("time_between", "between (time)", 2, "time_range"),

    OpSpec("is_true", "is true", 0, "none"),
    OpSpec("is_false", "is false", 0, "none"),

    OpSpec("one_of", "is one of…", 1, "choices"),
    OpSpec("contains", "contains", 1, "text"),
    OpSpec("matches", "matches (regex)", 1, "text"),

    OpSpec("has_item", "contains item", 1, "text"),
    OpSpec("count_gt", "item count >", 1, "count"),
    OpSpec("count_lt", "item count <", 1, "count"),
    OpSpec("count_eq", "item count =", 1, "count"),

    OpSpec("is_empty", "is empty", 0, "none"),
    OpSpec("not_empty", "is not empty", 0, "none"),
)

OPS = {spec.op: spec for spec in _SPECS}

NUMERIC_OPS = ("gt", "ge", "lt", "le", "eq", "ne", "between")
TIME_COMPARE_OPS = ("time_gt", "time_ge", "time_lt", "time_le", "time_eq",
                    "time_ne", "time_between")
BOOL_OPS = ("is_true", "is_false")
TEXT_OPS = ("one_of", "contains", "matches")
LIST_OPS = ("has_item", "count_gt", "count_lt", "count_eq")
UNIVERSAL_OPS = ("is_empty", "not_empty")


def operators_for(kind: str, numeric_ok: bool = False,
                  time_ok: bool = False) -> list:
    """The operator ids offered for a column, in display order."""
    if kind == KIND_NUMERIC:
        ops = list(NUMERIC_OPS)
    elif kind == KIND_TIME:
        ops = list(TIME_COMPARE_OPS)
    elif kind == KIND_DATETIME:
        ops = list(TIME_COMPARE_OPS) + list(TEXT_OPS)
    elif kind == KIND_BOOL:
        ops = list(BOOL_OPS)
    elif kind == KIND_LIST:
        ops = list(LIST_OPS) + list(TEXT_OPS)
    elif kind == KIND_TEXT:
        ops = list(TEXT_OPS)
        if numeric_ok:
            ops += list(NUMERIC_OPS)
        if time_ok:
            ops += list(TIME_COMPARE_OPS)
    else:                                    # KIND_EMPTY
        ops = []
    return ops + list(UNIVERSAL_OPS)


# ══ the query ════════════════════════════════════════════════════════════════
JOIN_AND = "and"
JOIN_OR = "or"


@dataclass(frozen=True)
class Condition:
    """One [column][operator][value] row. Frozen + hashable so a draft query
    can be compared to the committed one with a plain ==."""
    column: str
    op: str
    value: object = None

    def is_complete(self) -> bool:
        spec = OPS.get(self.op)
        if not self.column or spec is None:
            return False
        if spec.arity == 0:
            return True
        if self.value is None:
            return False
        if spec.arity == 2:
            return (isinstance(self.value, (tuple, list))
                    and len(self.value) == 2
                    and all(v is not None and v != "" for v in self.value))
        if isinstance(self.value, (tuple, list)):
            return len(self.value) > 0
        return self.value != ""


@dataclass(frozen=True)
class Group:
    conditions: tuple = ()
    join: str = JOIN_AND


@dataclass(frozen=True)
class Query:
    groups: tuple = ()
    join: str = JOIN_AND

    def is_empty(self) -> bool:
        return not any(any(c.is_complete() for c in g.conditions)
                       for g in self.groups)


def _compare(numbers: pd.Series, op: str, value) -> pd.Series:
    """Numeric/time comparison. NaN never matches — a missing value is not
    'less than 100'. `!=` is written out because pandas has NaN != v as True,
    which would quietly match every trade the key is absent on."""
    if op in ("between", "time_between"):
        low, high = value
        if low > high:
            low, high = high, low
        return numbers.between(low, high, inclusive="both")
    if op in ("gt", "time_gt"):
        return numbers > value
    if op in ("ge", "time_ge"):
        return numbers >= value
    if op in ("lt", "time_lt"):
        return numbers < value
    if op in ("le", "time_le"):
        return numbers <= value
    if op in ("eq", "time_eq"):
        return numbers == value
    return numbers.notna() & (numbers != value)      # ne / time_ne


def _operand_numbers(op: str, value):
    """The condition's operand(s), coerced the same way the column is."""
    parse = to_minutes if op in TIME_COMPARE_OPS else to_number
    if op in ("between", "time_between"):
        low, high = parse(value[0]), parse(value[1])
        if math.isnan(low) or math.isnan(high):
            return None
        return (low, high)
    parsed = parse(value)
    return None if math.isnan(parsed) else parsed


def _eval_condition(condition: Condition, columns: ColumnSource):
    """
    (mask, warning). A mask of None means DROP the condition (identity), not
    'nothing matches' — the Optimizer re-slices constantly, and a cell whose
    trades never set `trail5_stop` must not silently empty the whole report.
    """
    spec = OPS.get(condition.op)
    if spec is None:
        return None, f"unknown operator '{condition.op}' — condition ignored"
    if not condition.is_complete():
        return None, None
    if not columns.has(condition.column):
        return None, (f"'{condition.column}' is not in this trade set — "
                      f"condition ignored")

    op = condition.op
    if op == "is_empty":
        return columns.view(condition.column, VIEW_BLANK), None
    if op == "not_empty":
        return ~columns.view(condition.column, VIEW_BLANK), None

    if op in BOOL_OPS:
        truth = columns.view(condition.column, VIEW_BOOL)
        wanted = op == "is_true"
        return pd.Series([v is wanted for v in truth], index=truth.index,
                         dtype=bool), None

    if op in NUMERIC_OPS or op in TIME_COMPARE_OPS:
        operand = _operand_numbers(op, condition.value)
        if operand is None:
            return None, (f"'{display_text(condition.value)}' is not a valid "
                          f"{'time' if op in TIME_COMPARE_OPS else 'number'} "
                          f"— condition ignored")
        view = VIEW_MINUTES if op in TIME_COMPARE_OPS else VIEW_NUMBER
        return _compare(columns.view(condition.column, view), op, operand), None

    if op == "one_of":
        wanted = set(condition.value if isinstance(condition.value, (list, tuple))
                     else [condition.value])
        text = columns.view(condition.column, VIEW_TEXT)
        return text.isin(wanted), None

    if op in ("contains", "matches"):
        text = columns.view(condition.column, VIEW_TEXT)
        try:
            return text.str.contains(str(condition.value), case=False,
                                     regex=(op == "matches"), na=False), None
        except re.error as exc:
            return (pd.Series(False, index=text.index),
                    f"invalid regex '{condition.value}': {exc}")

    if op == "has_item":
        needle = str(condition.value).strip().lower()
        lists = columns.view(condition.column, VIEW_LISTS)
        return pd.Series(
            [any(display_text(x).strip().lower() == needle for x in items)
             for items in lists], index=lists.index, dtype=bool), None

    if op in ("count_gt", "count_lt", "count_eq"):
        wanted = to_number(condition.value)
        if math.isnan(wanted):
            return None, (f"'{display_text(condition.value)}' is not a valid "
                          f"count — condition ignored")
        counts = columns.view(condition.column, VIEW_LISTS).map(len)
        if op == "count_gt":
            return counts > wanted, None
        if op == "count_lt":
            return counts < wanted, None
        return counts == wanted, None       # count_eq 0 also matches a missing
                                            # key, which flattens to []

    return None, f"unhandled operator '{op}' — condition ignored"


def evaluate(query: Query | None, columns: ColumnSource) -> tuple:
    """
    (boolean mask indexed like the source, warnings).

    NEVER raises: every condition is wrapped, and a failing one fails closed
    with a warning rather than taking the report's filter chain down with it.
    An empty query, an empty group and a group whose every condition was
    dropped all reduce to 'no opinion' — an empty group under a top-level OR
    must not make everything match.
    """
    index = columns.index
    all_true = pd.Series(True, index=index)
    warnings: list = []
    if query is None or query.is_empty():
        return all_true, warnings

    group_masks = []
    for group in query.groups:
        masks = []
        for condition in group.conditions:
            try:
                mask, warning = _eval_condition(condition, columns)
            except Exception as exc:      # noqa: BLE001 — fail closed, loudly
                mask = pd.Series(False, index=index)
                warning = (f"{columns.label(condition.column)} "
                           f"{condition.op}: {exc}")
            if warning:
                warnings.append(warning)
            if mask is None:
                continue
            masks.append(mask.reindex(index, fill_value=False).astype(bool))
        if not masks:
            continue
        combined = masks[0]
        for mask in masks[1:]:
            combined = (combined & mask if group.join == JOIN_AND
                        else combined | mask)
        group_masks.append(combined)

    if not group_masks:
        return all_true, warnings
    result = group_masks[0]
    for mask in group_masks[1:]:
        result = result & mask if query.join == JOIN_AND else result | mask
    return result, warnings
