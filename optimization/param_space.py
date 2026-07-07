"""
Sweep-axis handling for the optimizer. Strategies declare nothing — which
params are sweepable is inferred from each PARAMS default's type, and the
min/max/step (or the explicit value list) is chosen by the user in the
optimizer UI:

  int / float default   -> swept over arange(min, max+step, step), max included
  str default           -> swept over a comma-separated value list
  bool / anything else  -> not sweepable (held fixed)

This module also enumerates the cartesian product of the chosen axes.
"""

from itertools import product

import numpy as np

# Axis roles in auto-assignment order (selection order: 1st checked -> x, ...).
ROLES = ["x", "y", "slider"]
ROLE_LABELS = {"x": "X axis", "y": "Y axis", "slider": "Slider"}
MAX_SWEPT = 3


def sweep_kind(default) -> str | None:
    """
    How a param can be swept, from its default value: 'int' / 'float' (min/
    max/step range), 'categorical' (value list), None = not sweepable.
    """
    if isinstance(default, bool):        # bool is an int subclass — check first
        return None
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "categorical"
    return None


def build_range(lo, hi, step, kind: str) -> list:
    """
    arange(lo, hi+step, step) with hi included and float dust rounded away.
    Raises ValueError with a readable message on bad input.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if hi < lo:
        raise ValueError("max < min")
    # explicit count: hi included when it sits on the grid (float-dust
    # tolerant), never exceeded when it doesn't
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    values = [lo + i * step for i in range(n)]
    if kind == "int":
        deduped = []
        for v in (int(round(v)) for v in values):
            if v not in deduped:
                deduped.append(v)
        return deduped
    return [float(np.round(v, 10)) for v in values]


def parse_values(text: str) -> list:
    """
    Comma-separated text -> de-duplicated string value list (categorical
    sweeps). Raises ValueError when empty.
    """
    tokens = [t.strip() for t in str(text).split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError("no values given")
    deduped = []
    for tok in tokens:
        if tok not in deduped:
            deduped.append(tok)
    return deduped


def enumerate_combos(axes: list) -> list:
    """
    Cartesian product of the swept axes -> list of {param: value} dicts.
    `axes` is an ordered list of {"param": str, "values": list}; the first
    axis varies slowest (row-major), so combo count == product of sizes.
    """
    if not axes:
        return []
    names = [a["param"] for a in axes]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate swept param in axes: {names}")
    value_sets = [a["values"] for a in axes]
    if any(len(vs) == 0 for vs in value_sets):
        raise ValueError("every swept axis needs at least one value")
    return [dict(zip(names, combo)) for combo in product(*value_sets)]


def combo_count(axes: list) -> int:
    n = 1
    for a in axes:
        n *= len(a["values"])
    return n if axes else 0
