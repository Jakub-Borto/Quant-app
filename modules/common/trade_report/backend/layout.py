"""
The trade report's section registry — what sections exist, in what order, and
how each is shown (always visible / collapsible dropdown / hidden).

Pure: the registry and the saved-layout reconciliation are data, so they live
here rather than next to the widgets that render them (see ui/stack.py for the
Qt half).

═══════════════════════════════════════════════════════════════════════════
 SECTION ORDER IS A VIEW CONCERN. IT NEVER MOVES A COMPUTATION.
═══════════════════════════════════════════════════════════════════════════

TradeReport._run_filters is the model and its order is FIXED:

    trade-type filter -> recompute cumulative_ticks
      -> News table (deliberately sees PRE-day-filter trades)
      -> day-type filter -> recompute cumulative_ticks
      -> regime filter -> recompute cumulative_ticks
      -> trade-notes query -> recompute cumulative_ticks
      -> panel.set_trades()

Dragging News below the trades table in the layout dialog moves a widget. It
does not change which trades that widget was handed. There is exactly ONE
implementation of that order (ui/report.py) and this is the only place the
sentence needs to be written down.

Sections come from TWO places and land in ONE ordered stack: the panel
registers its own (metrics, equity, RR, ...) and the report contributes the
ones it owns (the filter rows, news, the trades table, the actions row). That
is what lets a single flat user-ordered list span widgets that live in
different files.

Layout is persisted per user in settings.json under
ui_prefs[UI_PREF_TRADE_REPORT] and shared by every window that shows a report
— the Backtester and both Optimizer drill-downs.
"""

from dataclasses import dataclass

from modules.common.backend.settings import UI_PREF_TRADE_REPORT

# ── modes ────────────────────────────────────────────────────────────────────
MODE_VISIBLE = "visible"        # bare content, with the header above it
MODE_COLLAPSED = "collapsed"    # inside a CollapsibleSection, starts closed
MODE_HIDDEN = "hidden"

MODES = (MODE_VISIBLE, MODE_COLLAPSED, MODE_HIDDEN)
MODE_LABELS = {MODE_VISIBLE: "Always visible",
               MODE_COLLAPSED: "Dropdown",
               MODE_HIDDEN: "Hidden"}


@dataclass(frozen=True)
class SectionSpec:
    key: str                     # stable settings key — never rename
    title: str                   # dialog label + the header the stack supplies
    default_mode: str
    lockable: bool = False       # mode pinned to visible; still reorderable
    owns_header: bool = False    # widget draws its own header
    self_visible: bool = False   # widget manages its own visibility


# The shipped order: everything always-visible first, dropdowns after.
#
# THE `key` STRINGS ARE PERSISTENCE. A user's saved order lives in
# settings.json keyed by them, and resolve_layout silently drops keys it does
# not recognise — so renaming one does not raise, it quietly resets that
# user's layout. tests/test_trade_report.py pins the whole set.
DEFAULT_SECTIONS = (
    SectionSpec("trade_type_filter", "Filter by trade type", MODE_VISIBLE,
                lockable=True, owns_header=True),
    SectionSpec("day_type_filter", "Filter by day type", MODE_VISIBLE,
                lockable=True, owns_header=True),
    SectionSpec("metrics", "Performance", MODE_VISIBLE, lockable=True),
    SectionSpec("equity", "Equity Curve", MODE_VISIBLE, lockable=True),
    SectionSpec("chart_controls", "Chart View Settings", MODE_VISIBLE),
    SectionSpec("trade_detail", "Trade Detail", MODE_VISIBLE,
                owns_header=True),
    SectionSpec("regime", "Strategy Performance by Regime", MODE_COLLAPSED),
    SectionSpec("entry_breakdown", "Entry Breakdown", MODE_COLLAPSED),
    SectionSpec("news", "News & Holiday Exposure", MODE_COLLAPSED),
    SectionSpec("exposure", "Market Exposure (α/β regression)", MODE_COLLAPSED),
    SectionSpec("exit_breakdown", "Exit Breakdown", MODE_COLLAPSED),
    SectionSpec("rr", "RR Distribution", MODE_COLLAPSED),
    SectionSpec("trades_table", "Trades & Notes", MODE_COLLAPSED),
    SectionSpec("actions", "Save / Go to…", MODE_VISIBLE),
)

SPEC_BY_KEY = {s.key: s for s in DEFAULT_SECTIONS}
DEFAULT_ORDER = [s.key for s in DEFAULT_SECTIONS]

# Exactly the set the report's _set_report_visible() hides when a filter
# selects nothing — kept identical to the pre-refactor behaviour, so News and
# the regime tables stay on screen with their pre-filter contents.
REPORT_KEYS = frozenset({"metrics", "equity", "chart_controls", "trade_detail",
                         "exposure", "exit_breakdown", "rr", "trades_table",
                         "actions"})


def resolve_layout(saved: dict | None) -> tuple[list[str], dict[str, str]]:
    """
    (order, modes) from a saved preference blob, reconciled against the
    current registry: unknown keys are dropped, and any registry key the save
    predates is re-inserted at its default position — so shipping a new
    section never wipes a user's order. Lockable specs are forced visible
    whatever is on disk.
    """
    saved = saved or {}
    seen = set()
    order = []
    for key in saved.get("order", []):
        if key in SPEC_BY_KEY and key not in seen:
            order.append(key)
            seen.add(key)
    # Re-insert registry keys the save predates, each one anchored just after
    # its nearest preceding DEFAULT_ORDER neighbour that is already placed —
    # so a newly shipped section lands where it belongs without disturbing the
    # user's existing relative order.
    for index, key in enumerate(DEFAULT_ORDER):
        if key in seen:
            continue
        anchor = -1
        for earlier in reversed(DEFAULT_ORDER[:index]):
            if earlier in seen:
                anchor = order.index(earlier)
                break
        order.insert(anchor + 1, key)
        seen.add(key)

    saved_modes = saved.get("modes", {}) or {}
    modes = {}
    for key in order:
        spec = SPEC_BY_KEY[key]
        mode = saved_modes.get(key, spec.default_mode)
        if mode not in MODES:
            mode = spec.default_mode
        modes[key] = MODE_VISIBLE if spec.lockable else mode
    return order, modes


def layout_from_settings(settings) -> tuple[list[str], dict[str, str]]:
    saved = settings.ui_pref(UI_PREF_TRADE_REPORT) if settings else None
    return resolve_layout(saved)
