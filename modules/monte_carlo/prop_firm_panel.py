"""
Prop-firm Monte Carlo panel — the dedicated UI shown when the selected method
module sets PROP_FIRM = True (e.g. methods/prop_firm.py).

The PySide6 port of _render_prop_firm / _render_prop_sim / _render_combined_sim
from legacy_streamlit/views/monte_carlo.py: General inputs, the cap-wins
toggle, the 6 challenge + 7 payout rule widgets (checkbox + value, greyed
when off, % rules stored as fractions), costs controls, Run on a worker, and
the three result blocks (two truncated fan charts + stats tables, then the
challenge->reset->funded combined chart with the funnel tiles). The params
dict passed to mc_module.run() is verbatim.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from modules.common.ui.charts.base import make_plot
from modules.common.ui.charts.fan_chart import FanChart
from modules.common.ui.dataframe_model import make_table_view
from modules.common.ui.widgets import Banner, Caption, MetricTile, SectionHeader
from modules.common.ui.workers import FunctionWorker
from modules.monte_carlo.backend.cost_ctx import build_cost_ctx
from modules.monte_carlo.backend.stats import (SAMPLE_PATH_COUNT,
                                               _compute_metrics,
                                               _select_featured_paths)

_PROP_OPTIMISM_CAPTION = (
    "Breach is checked on **closed-trade** equity — an intra-trade dip below the "
    "floor that recovers to a green close is not counted. Every P(pass)/P(payout) "
    "here is therefore an **upper bound**."
)


# ── formatters (verbatim) ─────────────────────────────────────────────────────

def _fmt_pct01(v):    return f"{v*100:.1f}%" if v is not None else "—"
def _fmt_dollar(v):   return f"${v:,.0f}"    if v is not None else "—"


def _fmt_pctiles(p: dict | None) -> str:
    if p is None:
        return "— (none)"
    return f"{p['median']:.0f}  (25–75: {p['p25']:.0f}–{p['p75']:.0f}, 95th: {p['p95']:.0f})"


class RuleWidget(QWidget):
    """Checkbox + value input for an {enabled, value} rule; the value greys
    out when the checkbox is off. % rules display ×100 and store /100."""

    def __init__(self, label: str, default: dict, *, step: float = 100.0,
                 pct: bool = False, help_text: str | None = None, parent=None):
        super().__init__(parent)
        self._pct = pct
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._check = QCheckBox(label)
        self._check.setChecked(bool(default["enabled"]))
        if help_text:
            self._check.setToolTip(help_text)
        self._value = QDoubleSpinBox()
        self._value.setDecimals(2)
        if pct:
            self._value.setRange(0.0, 100.0)
            self._value.setSingleStep(5.0)
            self._value.setValue(float(default["value"]) * 100.0)
            self._value.setSuffix("  % of profit from one day")
        else:
            self._value.setRange(-1e12, 1e12)
            self._value.setSingleStep(step)
            self._value.setValue(float(default["value"]))
        self._value.setEnabled(self._check.isChecked())
        self._check.toggled.connect(self._value.setEnabled)
        row.addWidget(self._check, stretch=3)
        row.addWidget(self._value, stretch=2)

    def value(self) -> dict:
        v = float(self._value.value())
        if self._pct:
            v = v / 100.0
        return {"enabled": bool(self._check.isChecked()), "value": v}


# rule label / defaults-key / step / pct / help — the verbatim widget list
_CHALLENGE_RULES = [
    ("Profit Target ($)", "profit_target", 500.0, False, None),
    ("Max Loss Limit (EOD, trailing $)", "challenge_max_loss_eod", 500.0, False,
     "Trailing from the highest end-of-day balance; ratchets up only, then "
     "locks once the floor reaches the starting balance (never above it)."),
    ("Static Loss Limit ($)", "challenge_static_loss", 500.0, False,
     "Fixed floor at (starting balance − value) that never trails. Breach "
     "when closed equity drops to it. Independent of the EOD limit — run "
     "either or both; the nearer floor binds."),
    ("Daily Loss Limit ($)", "challenge_daily_loss", 250.0, False,
     "Risk cap, not a breach — caps size so one day can't lose more than this."),
    ("Consistency Rule", "challenge_consistency", 5.0, True,
     "Max % of total profit allowed from a single day. A pass gate, not a breach."),
    ("Contract Limit", "challenge_contract_limit", 1.0, False,
     "Hard size cap, full-contract units (3.0 = 3 minis = 30 micros)."),
]
_PAYOUT_RULES = [
    ("Targeted Payout ($)", "targeted_payout", 500.0, False, None),
    ("Max Loss Limit (EOD, trailing $)", "payout_max_loss_eod", 500.0, False,
     "Trailing from the highest end-of-day balance; ratchets up only, then "
     "locks once the floor reaches the starting balance (never above it)."),
    ("Static Loss Limit ($)", "payout_static_loss", 500.0, False,
     "Fixed floor at (starting balance − value) that never trails. "
     "Independent of the EOD limit — run either or both."),
    ("Daily Loss Limit ($)", "payout_daily_loss", 250.0, False, None),
    ("Consistency Rule", "payout_consistency", 5.0, True, None),
    ("Contract Limit", "payout_contract_limit", 1.0, False, None),
    ("Maximum Withdrawal ($/payout)", "maximum_withdrawal", 500.0, False,
     "Caps the dollar amount per payout: realized = min(profit, this)."),
]


class PropFirmPanel(QWidget):
    """context_fn() -> {trades_ref, sizer_module, sizer_params, account_size}
    (provided by the Monte Carlo window; read at Run click)."""

    def __init__(self, mc_module, context_fn, track_worker, parent=None):
        super().__init__(parent)
        self._mc_module = mc_module
        self._context_fn = context_fn
        self._track_worker = track_worker
        self._results: dict | None = None    # the old st.session_state.pf_results
        self._costs_on = False               # the old pf_costs

        defaults = getattr(mc_module, "PARAMS", {})
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ── general ───────────────────────────────────────────────────────────
        lay.addWidget(SectionHeader("General"))
        g = QGridLayout()
        self._n_paths = QSpinBox()
        self._n_paths.setRange(100, 1_000_000)
        self._n_paths.setSingleStep(500)
        self._n_paths.setValue(int(defaults.get("n_paths", 5000)))
        self._max_trades = QSpinBox()
        self._max_trades.setRange(10, 1_000_000)
        self._max_trades.setSingleStep(50)
        self._max_trades.setValue(int(defaults.get("max_trades", 500)))
        self._max_trades.setToolTip(
            "Trades before an unresolved path is stopped and counted as "
            "'unresolved' (~trading days at 1 trade/day).")
        self._seed = QSpinBox()
        self._seed.setRange(-1_000_000_000, 1_000_000_000)
        self._seed.setValue(int(defaults.get("seed", 42)))
        for col, (label, w) in enumerate([("Paths", self._n_paths),
                                          ("Max trades (horizon)", self._max_trades),
                                          ("Seed", self._seed)]):
            g.addWidget(QLabel(label), 0, col)
            g.addWidget(w, 1, col)
        lay.addLayout(g)

        self._cap_wins = QCheckBox("Cap daily wins at the consistency limit")
        self._cap_wins.setChecked(bool(defaults.get("cap_wins_to_consistency", False)))
        self._cap_wins.setToolTip(
            "Model a trader who stops/reduces once the day's profit hits the "
            "consistency daily threshold (consistency% × target), so a single "
            "day never exceeds it and the consistency recalculation is never "
            "triggered. Applies to whichever phase has its consistency rule "
            "enabled.")
        lay.addWidget(self._cap_wins)

        # ── rulesets ──────────────────────────────────────────────────────────
        self._rules: dict[str, RuleWidget] = {}
        for title, spec in (("Challenge — passing ruleset", _CHALLENGE_RULES),
                            ("Payout — funded ruleset", _PAYOUT_RULES)):
            lay.addWidget(SectionHeader(title))
            for label, key, step, pct, help_text in spec:
                rule = RuleWidget(label, defaults[key], step=step, pct=pct,
                                  help_text=help_text)
                self._rules[key] = rule
                lay.addWidget(rule)

        # ── costs ─────────────────────────────────────────────────────────────
        self._apply_costs = QCheckBox("Apply commissions && slippage")
        self._apply_costs.setChecked(True)
        lay.addWidget(self._apply_costs)
        slip_row = QHBoxLayout()
        self._slip_label = QLabel("Slippage (ticks/side)")
        self._slippage = QSlider(Qt.Horizontal)
        self._slippage.setRange(1, 5)
        self._slippage.setValue(1)
        self._slippage.setMaximumWidth(220)
        self._slippage.setToolTip("Entry-side ticks slipped per trade; market "
                                  "exits (losers) slip 2×.")
        self._slip_value = QLabel("1")
        self._slippage.valueChanged.connect(lambda v: self._slip_value.setText(str(v)))
        self._apply_costs.toggled.connect(self._slippage.setVisible)
        self._apply_costs.toggled.connect(self._slip_label.setVisible)
        self._apply_costs.toggled.connect(self._slip_value.setVisible)
        slip_row.addWidget(self._slip_label)
        slip_row.addWidget(self._slippage)
        slip_row.addWidget(self._slip_value)
        slip_row.addStretch()
        lay.addLayout(slip_row)

        # ── run ───────────────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Prop-Firm Simulation")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(260)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addStretch()
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        lay.addLayout(run_row)
        self._status = Caption("")
        lay.addWidget(self._status)
        self._banner = Banner()
        lay.addWidget(self._banner)

        self._results_holder = QVBoxLayout()
        lay.addLayout(self._results_holder)

    # ── run flow ──────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        ctx = self._context_fn()
        if ctx is None:
            self._banner.show_message("error", "Pick a trades file and sizer first.")
            return
        trades_ref, sizer_module = ctx["trades_ref"], ctx["sizer_module"]
        sizer_params, account_size = ctx["sizer_params"], ctx["account_size"]

        from modules.common.backend.asset_info import get_dollars_per_tick
        try:
            dollars_per_tick = get_dollars_per_tick(trades_ref.filename)
        except ValueError as e:
            self._banner.show_message("error", str(e))
            return
        try:
            trades = pd.read_parquet(trades_ref.path)
        except Exception as e:  # noqa: BLE001
            self._banner.show_message("error", f"Could not load trades: {e}")
            return

        apply_costs = self._apply_costs.isChecked()
        slippage_n = int(self._slippage.value()) if apply_costs else 1
        cost_ctx, warn_missing = build_cost_ctx(trades_ref.filename,
                                                apply_costs, slippage_n)
        if warn_missing:
            self._banner.show_message(
                "warning",
                f"No commission rate for asset "
                f"'{trades_ref.filename.split('_')[0]}' — commissions billed "
                f"at 0; slippage still applies.")

        final_sizer_params = {**sizer_params, "dollars_per_tick": dollars_per_tick}
        params = {
            "n_paths": int(self._n_paths.value()),
            "max_trades": int(self._max_trades.value()),
            "seed": int(self._seed.value()),
            "account_size": account_size,
            "increment": sizer_params.get("contract_increment", 1.0),
            "cost_ctx": cost_ctx,
            "cap_wins_to_consistency": bool(self._cap_wins.isChecked()),
            "profit_target": self._rules["profit_target"].value(),
            "challenge_max_loss_eod": self._rules["challenge_max_loss_eod"].value(),
            "challenge_static_loss": self._rules["challenge_static_loss"].value(),
            "challenge_daily_loss": self._rules["challenge_daily_loss"].value(),
            "challenge_consistency": self._rules["challenge_consistency"].value(),
            "challenge_contract_limit": self._rules["challenge_contract_limit"].value(),
            "targeted_payout": self._rules["targeted_payout"].value(),
            "payout_max_loss_eod": self._rules["payout_max_loss_eod"].value(),
            "payout_static_loss": self._rules["payout_static_loss"].value(),
            "payout_daily_loss": self._rules["payout_daily_loss"].value(),
            "payout_consistency": self._rules["payout_consistency"].value(),
            "payout_contract_limit": self._rules["payout_contract_limit"].value(),
            "maximum_withdrawal": self._rules["maximum_withdrawal"].value(),
        }

        self._run_btn.setEnabled(False)
        self._status.setText(
            f"Running prop-firm MC — {int(self._n_paths.value()):,} paths × 2 sims…")
        self._pending_costs_on = apply_costs
        worker = FunctionWorker(self._mc_module.run, trades=trades,
                                sizer_module=sizer_module,
                                sizer_params=final_sizer_params, params=params)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._track_worker(worker)

    def _on_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._banner.show_message("error", f"Simulation error: {message}")

    def _on_finished(self, results: dict) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._results = results
        self._costs_on = self._pending_costs_on
        for w in results.get("warnings", []):
            self._banner.show_message("warning", w)
        self._render_results()

    # ── results ───────────────────────────────────────────────────────────────
    def _clear_results(self) -> None:
        while self._results_holder.count():
            item = self._results_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _render_results(self) -> None:
        self._clear_results()
        results = self._results
        if results is None:
            return
        account_size = results["account_size"]
        self._add_prop_sim(results["sim1"], account_size)
        self._add_prop_sim(results["sim2"], account_size)
        self._add_combined_sim(results["sim3"], account_size)

    def _add_prop_sim(self, sim: dict, account_size: float) -> None:
        lay = self._results_holder
        lay.addWidget(SectionHeader(sim["title"]))
        caption = _PROP_OPTIMISM_CAPTION
        if self._costs_on:
            caption += "  Equity is net of commissions & slippage."
        lab = Caption(caption)
        lab.setTextFormat(Qt.MarkdownText)
        lay.addWidget(lab)

        # fan chart with NaN-truncated lines past each path's stop step
        eqm = sim["equity_matrix"]
        featured = _select_featured_paths(eqm)      # needs real finals → full matrix
        metrics = _compute_metrics(eqm, account_size, None)
        n_steps = eqm.shape[1]
        cols = np.arange(n_steps)
        display = np.where(cols[None, :] > sim["stop_step"][:, None], np.nan, eqm)
        fan = FanChart()
        fan.set_data(eqm, account_size, featured, None, y_max=None,
                     band_finals=metrics["band_finals"], line_matrix=display,
                     target=(account_size + sim["target"]) if sim.get("target") else None,
                     x_label="Trade # (path stops at pass/fail, then holds flat)")
        lay.addWidget(fan)

        # stats table (verbatim rows per challenge/payout key set)
        s = sim["stats"]
        rows = []
        if "p_pass" in s:        # challenge
            rows.append(("P(pass)", _fmt_pct01(s["p_pass"])))
            rows.append(("Trades to pass", _fmt_pctiles(s["trades_to_pass"])))
            fb = s["failure_breakdown"]
            rows.append(("Failure — max-loss breach (trailing)", _fmt_pct01(fb["max_loss"])))
            rows.append(("Failure — static-loss breach", _fmt_pct01(fb.get("static_loss", 0.0))))
            rows.append(("Failure — unresolved at horizon", _fmt_pct01(fb["unresolved"])))
            rows.append(("Consistency hold rate", _fmt_pct01(s["consistency_hold_rate"])))
            rows.append(("Median final equity (passers)", _fmt_dollar(s["median_final_equity_passers"])))
            rows.append(("Worst peak-to-trough (passers)", _fmt_dollar(s["worst_peak_to_trough_passers"])))
        else:                    # payout
            bb = s["breach_breakdown"]
            rows.append(("P(payout | funded)", _fmt_pct01(s["p_payout"])))
            rows.append(("Trades to payout", _fmt_pctiles(s["trades_to_payout"])))
            rows.append(("Funded breach — max-loss (trailing)", _fmt_pct01(bb["max_loss"])))
            rows.append(("Funded breach — static-loss", _fmt_pct01(bb.get("static_loss", 0.0))))
            rows.append(("Unresolved at horizon", _fmt_pct01(bb["unresolved"])))
            rows.append(("Held-then-diluted-and-paid", _fmt_pct01(s["held_then_paid"])))
            rows.append(("Held-then-breached-while-grinding", _fmt_pct01(s["held_then_breached"])))
            rows.append(("Consistency hold rate", _fmt_pct01(s["consistency_hold_rate"])))
        lay.addWidget(make_table_view(
            pd.DataFrame(rows, columns=["Statistic", "Value"]),
            height=40 + 30 * len(rows)))

    def _add_combined_sim(self, sim: dict, account_size: float) -> None:
        lay = self._results_holder
        lay.addWidget(SectionHeader(sim["title"]))
        lay.addWidget(Caption(
            "End-to-end: challenge phase, then a fresh funded phase concatenated "
            "at the reset (yellow dot). Funded accounts reset to the starting "
            "balance — challenge profit is not carried. Green lines went on to "
            "get paid."))

        mat, reset_x, paid = sim["equity_matrix"], sim["reset_x"], sim["paid_mask"]
        if mat.shape[0] == 0:
            lay.addWidget(Banner("info", "No paths passed the challenge — "
                                         "nothing to chart."))
        else:
            plot = make_plot("Trade # (challenge → reset → funded)", "Equity ($)")
            plot.setMinimumHeight(440)
            x = np.arange(mat.shape[1], dtype=float)
            rng_ = np.random.default_rng(0)
            sample = rng_.choice(mat.shape[0],
                                 size=min(SAMPLE_PATH_COUNT, mat.shape[0]),
                                 replace=False)
            for mask, color in ((paid, (80, 200, 120, 77)),
                                (~paid, (180, 180, 180, 46))):
                idxs = [int(r) for r in sample if mask[r]]
                if not idxs:
                    continue
                xs = np.concatenate([np.append(x, np.nan) for _ in idxs])
                ys = np.concatenate([np.append(mat[r], np.nan) for r in idxs])
                plot.addItem(pg.PlotDataItem(xs, ys, connect="finite",
                                             pen=pg.mkPen(*color, width=1)))
            rxs = np.array([int(reset_x[r]) for r in sample], dtype=float)
            rys = np.array([float(mat[r, int(reset_x[r])]) for r in sample])
            plot.addItem(pg.ScatterPlotItem(x=rxs, y=rys, size=4,
                                            brush=pg.mkBrush("#ffcc00"), pen=None))
            plot.addItem(pg.InfiniteLine(
                pos=account_size, angle=0,
                pen=pg.mkPen(255, 255, 255, 64, width=1, style=pg.QtCore.Qt.DotLine),
                label=f"Start ${account_size:,.0f}",
                labelOpts={"color": "#98a0b3", "position": 0.03}))
            lay.addWidget(plot)

        s = sim["stats"]
        p_pass, p_po, p_paid = s["funnel"]
        tiles = QGridLayout()
        tiles.addWidget(MetricTile("P(pass)", f"{p_pass*100:.1f}%"), 0, 0)
        tiles.addWidget(MetricTile("P(payout | passed)", f"{p_po*100:.1f}%"), 0, 1)
        tiles.addWidget(MetricTile("P(paid) end-to-end", f"{p_paid*100:.2f}%"), 0, 2)
        lay.addLayout(tiles)

        rows = [
            ("Total trades to payout (challenge + funded)", _fmt_pctiles(s["total_trades_to_payout"])),
            ("Realized payout per paid account", _fmt_dollar(s["realized_payout_per_paid"])),
            ("Expected payout value  =  P(paid) × realized", _fmt_dollar(s["expected_payout_value"])),
        ]
        lay.addWidget(make_table_view(
            pd.DataFrame(rows, columns=["Statistic", "Value"]), height=140))
