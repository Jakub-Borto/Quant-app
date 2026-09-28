"""
Entry finders registry.

Each finder module exposes find_entry(win, params) taking the shared EntryWindow context
(see _daydata) and returns:
    (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)
with window-relative bar indices (the dispatcher maps them back to timestamps/positions),
plus its own declarations: PARAMS, SECTION, DEFAULT_ON, BASELINES (collected by
..params and core — a finder is fully described by its own file).

The order of FINDER_MODULES is the bit order of the valid_entries / trail_entries flag
strings. To add a finder: drop a module here and append it to FINDER_MODULES. To remove
one: delete its file and its line below.
"""

from . import absorption_delta
from . import consecutive_absorption
from . import two_bar_absorption
from . import passive_absorption_size_only
from . import passive_wall
from . import cvd_divergence_absorption
from . import cvd_divergence_exhaustion

FINDER_MODULES = [
    absorption_delta,
    consecutive_absorption,
    two_bar_absorption,
    passive_absorption_size_only,
    passive_wall,
    cvd_divergence_absorption,
    cvd_divergence_exhaustion,
]

FINDER_REGISTRY = [m.find_entry for m in FINDER_MODULES]
FINDER_NAMES    = [m.__name__.rsplit(".", 1)[-1] for m in FINDER_MODULES]

# default flag string: one bit per finder, from each finder's DEFAULT_ON
DEFAULT_FLAGS = "".join("1" if m.DEFAULT_ON else "0" for m in FINDER_MODULES)
