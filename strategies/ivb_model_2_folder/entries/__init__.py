"""
Entry finders registry.

Each finder exposes find_entry(...) with the shared signature and returns:
    (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)

The order here maps to the valid_entries flag string ("1111" = all on).
To add a new entry type: drop a module in this folder and append its find_entry here.
"""

from .pure_absorption        import find_entry as pure_absorption
from .consecutive_absorption import find_entry as consecutive_absorption
from .two_bar_absorption     import find_entry as two_bar_absorption
from .passive_absorption     import find_entry as passive_absorption

FINDER_REGISTRY = [
    pure_absorption,
    consecutive_absorption,
    two_bar_absorption,
    passive_absorption,
]
