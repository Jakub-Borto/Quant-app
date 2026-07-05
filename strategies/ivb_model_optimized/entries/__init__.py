"""
Entry finders registry.

Each finder exposes find_entry(win, params) taking the shared EntryWindow context
(see _daydata) and returns:
    (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type)
with window-relative bar indices (the dispatcher maps them back to timestamps/positions).

The order here maps to the valid_entries flag string ("1111111" = all on).
To add a new entry type: drop a module in this folder and append its find_entry (and name)
here.
"""

from .absorption_delta             import find_entry as absorption_delta
from .consecutive_absorption       import find_entry as consecutive_absorption
from .two_bar_absorption           import find_entry as two_bar_absorption
from .passive_absorption_size_only import find_entry as passive_absorption_size_only
from .passive_wall                 import find_entry as passive_wall
from .cvd_divergence_absorption    import find_entry as cvd_divergence_absorption
from .cvd_divergence_exhaustion    import find_entry as cvd_divergence_exhaustion

FINDER_REGISTRY = [
    absorption_delta,
    consecutive_absorption,
    two_bar_absorption,
    passive_absorption_size_only,
    passive_wall,
    cvd_divergence_absorption,
    cvd_divergence_exhaustion,
]

FINDER_NAMES = [
    "absorption_delta",
    "consecutive_absorption",
    "two_bar_absorption",
    "passive_absorption_size_only",
    "passive_wall",
    "cvd_divergence_absorption",
    "cvd_divergence_exhaustion",
]
