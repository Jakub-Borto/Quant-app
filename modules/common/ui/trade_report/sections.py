"""
The trade report's section registry — what sections exist, in what order, and
how each is shown (always visible / collapsible dropdown / hidden).

═══════════════════════════════════════════════════════════════════════════
 SECTION ORDER IS A VIEW CONCERN. IT NEVER MOVES A COMPUTATION.
═══════════════════════════════════════════════════════════════════════════

The host window's _apply_filters is the model and its order is FIXED:

    trade-type filter -> recompute cumulative_ticks
      -> News table (deliberately sees PRE-day-filter trades)
      -> day-type filter -> recompute cumulative_ticks
      -> regime filter -> recompute cumulative_ticks
      -> panel.set_trades()

Dragging News below the trades table in the layout dialog moves a widget. It
does not change which trades that widget was handed.

Sections come from TWO places and land in ONE ordered stack: the panel
registers its own (metrics, equity, RR, ...) and the host window contributes
the ones it owns (the filter rows, news, the trades table, the actions row).
That is what lets a single flat user-ordered list span widgets that live in
different files.

Layout is persisted per user in settings.json under
ui_prefs[UI_PREF_TRADE_REPORT] and shared by every window that shows a
report — the Backtester and the Optimizer's cell drill-down.
"""

import weakref
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from shiboken6 import isValid

from modules.common.backend.settings import UI_PREF_TRADE_REPORT
from ..widgets import CollapsibleSection, SectionHeader, pin_minimum_height

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
    SectionSpec("news", "News & Holiday Exposure", MODE_COLLAPSED),
    SectionSpec("exposure", "Market Exposure (α/β regression)", MODE_COLLAPSED),
    SectionSpec("exit_breakdown", "Exit Breakdown", MODE_COLLAPSED),
    SectionSpec("rr", "RR Distribution", MODE_COLLAPSED),
    SectionSpec("trades_table", "Trades", MODE_COLLAPSED),
    SectionSpec("actions", "Save / Go to…", MODE_VISIBLE),
)

SPEC_BY_KEY = {s.key: s for s in DEFAULT_SECTIONS}
DEFAULT_ORDER = [s.key for s in DEFAULT_SECTIONS]

# Exactly the set the hosts' _set_report_visible() hides when a filter selects
# nothing — kept identical to the pre-refactor behaviour, so News and the
# regime tables stay on screen with their pre-filter contents.
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


class _LayoutBus(QObject):
    """App-wide 'the layout changed' ping so every open report restacks at
    once (module-level singleton, same precedent as actions_row's
    _SPAWNED_WINDOWS)."""
    changed = Signal()


LAYOUT_BUS = _LayoutBus()

# Live stacks, held WEAKLY. A closed Backtester window must not keep its
# stack alive, and — the part that actually bites — the bus must never call
# back into a stack whose C++ side is already gone: connecting each stack's
# bound method to a module-level singleton outlives the widget and segfaults
# on the next gear save. WeakSet handles the Python side, isValid() the C++
# side (they die independently).
_LIVE_STACKS: "weakref.WeakSet" = weakref.WeakSet()


def _broadcast_layout() -> None:
    for stack in list(_LIVE_STACKS):
        if isValid(stack):
            stack.reload_layout()


LAYOUT_BUS.changed.connect(_broadcast_layout)


class ReportSection(QWidget):
    """Base for section content widgets: the content ONLY — the stack supplies
    the header and the collapsible chrome, because the same widget renders
    both ways depending on the user's setting."""


class _SectionFrame(QWidget):
    """
    One section's stable slot in the stack.

    Stability is the point: reordering moves FRAMES, never rebuilds content,
    so a chart's zoom, a checkbox row's selection and a spin box's value all
    survive a trip through the layout dialog.
    """

    def __init__(self, spec: SectionSpec, content: QWidget, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.content = content
        self._mode = None

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        # a frame must never be sized below the section it holds — otherwise
        # the stack squeezes it and the content is clipped (see SectionStack)
        pin_minimum_height(self)

        self._header = None
        if not spec.owns_header:
            self._header = SectionHeader(spec.title)
            self._lay.addWidget(self._header)

        self._box = CollapsibleSection(spec.title)
        self._lay.addWidget(self._box)
        self._box.setVisible(False)

        self._lay.addWidget(content)
        self.set_mode(spec.default_mode)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            mode = self.spec.default_mode
        if mode == self._mode:
            return
        self._mode = mode

        # move the content between the frame's own layout and the collapsible
        self.content.setParent(None)
        if mode == MODE_COLLAPSED:
            self._box.add_widget(self.content)
        else:
            self._lay.addWidget(self.content)
        self.content.setVisible(True)

        self._box.setVisible(mode == MODE_COLLAPSED)
        if self._header is not None:
            self._header.setVisible(mode == MODE_VISIBLE)
        self.setVisible(mode != MODE_HIDDEN)

    @property
    def mode(self) -> str:
        return self._mode


class SectionStack(QWidget):
    """The ordered column of sections. Hosts and the panel both register into
    one of these; apply_layout() then orders and styles it from settings."""

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._frames: dict[str, _SectionFrame] = {}
        self._pending: dict[str, QWidget] = {}
        self._has_tail = False          # the surplus-absorbing trailing stretch
        self._hidden_by_host: set[str] = set()
        self._report_visible = True

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(10)
        # THE dropdown-jitter fix. Expanding a section makes the stack's
        # children need more room, but the stack widget itself is only resized
        # by its parent LATER — traced order: regime frame grows to 313, then
        # equity is crushed 572->313 and metrics 330->313, and only then does
        # the stack grow 1432->1849. Qt violates the children's own minimums
        # to absorb that temporary deficit, which is the visible squash
        # (clipped tiles, half-height chart). Pinning the stack's minimum to
        # its content makes it grow FIRST, so the deficit never exists.
        pin_minimum_height(self)
        _LIVE_STACKS.add(self)      # weak — see _broadcast_layout

    # ── registration ──────────────────────────────────────────────────────────
    def register(self, key: str, widget: QWidget) -> None:
        """Contribute a section's content widget. Unknown keys are ignored so
        a host can offer a section this build doesn't know about."""
        if key in SPEC_BY_KEY:
            self._pending[key] = widget

    def build(self) -> None:
        """Create frames for everything registered, then lay them out."""
        for key, widget in self._pending.items():
            if key not in self._frames:
                frame = _SectionFrame(SPEC_BY_KEY[key], widget)
                self._frames[key] = frame
                self._lay.addWidget(frame)
        self._pending.clear()
        # A trailing stretch to absorb SURPLUS height. Collapsing a section
        # frees its height inside a stack that is still its old size for one
        # pass, and QVBoxLayout hands that surplus to whichever child will
        # take it — pyqtgraph plots are Expanding, so the equity chart
        # ballooned by exactly the freed amount and snapped back (traced:
        # 539 -> 958 -> 539). The stretch has stretch factor 1 so it wins the
        # surplus instead, and costs nothing at rest because the stack's
        # minimum equals its content.
        if not self._has_tail:
            self._lay.addStretch(1)
            self._has_tail = True
        self.reload_layout()

    # ── layout ────────────────────────────────────────────────────────────────
    def reload_layout(self) -> None:
        order, modes = layout_from_settings(self.settings)
        # one atomic relayout — without this the intermediate states of the
        # remove/insert loop are painted and the page visibly jumps
        self.setUpdatesEnabled(False)
        try:
            for position, key in enumerate([k for k in order
                                            if k in self._frames]):
                frame = self._frames[key]
                self._lay.removeWidget(frame)
                self._lay.insertWidget(position, frame)
                frame.set_mode(modes[key])
                self._apply_visibility(key)
        finally:
            self.setUpdatesEnabled(True)

    def set_settings(self, settings) -> None:
        self.settings = settings
        self.reload_layout()

    # ── visibility (host-driven, on top of the layout mode) ──────────────────
    def _apply_visibility(self, key: str) -> None:
        frame = self._frames.get(key)
        if frame is None:
            return
        visible = (frame.mode != MODE_HIDDEN
                   and key not in self._hidden_by_host
                   and (self._report_visible or key not in REPORT_KEYS))
        frame.setVisible(visible)

    def set_report_visible(self, visible: bool) -> None:
        """Hide/show the report proper (everything except the filters and the
        breakdown tables) — what the hosts call when a filter selects nothing."""
        self._report_visible = visible
        for key in self._frames:
            self._apply_visibility(key)

    def set_frame_visible(self, key: str, visible: bool) -> None:
        """Host override for one section (e.g. no trade_type column this run,
        or the trade detail before a point is clicked)."""
        if visible:
            self._hidden_by_host.discard(key)
        else:
            self._hidden_by_host.add(key)
        self._apply_visibility(key)

    def frame(self, key: str) -> _SectionFrame | None:
        return self._frames.get(key)

    def keys(self) -> list[str]:
        """Registered section keys in current visual order."""
        order, _modes = layout_from_settings(self.settings)
        return [k for k in order if k in self._frames]
