"""
views/ohlcv_chart.py  —  Clean OHLCV chart
- Multi-timeframe (1m source, resamples to 1/5/15/30m, 1h/4h/1D/1W/1M)
- Volume sub-panel
- Drawing tools: H-line, V-line, Diagonal, Delete, Clear
- Dual-chart mode with optional drawing sync
- No indicators, no footprint — plain OHLCV
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

TICK_SIZES = {
    "ES": 0.25, "NQ": 0.25, "RTY": 0.10, "YM": 1.00,
    "MES": 0.25, "MNQ": 0.25, "M2K": 0.10, "MYM": 1.00,
    "ZN": 0.015625, "ZB": 0.03125, "ZF": 0.0078125, "ZT": 0.00390625, "SR3": 0.0025,
    "CL": 0.01, "QM": 0.025, "NG": 0.001, "RB": 0.0001, "HO": 0.0001,
    "GC": 0.10, "MGC": 0.10, "SI": 0.005, "HG": 0.0005,
    "ZC": 0.25, "ZS": 0.25, "ZW": 0.25,
    "6E": 0.00005, "6J": 0.0000005, "6B": 0.0001, "6C": 0.00005,
    "BTC": 5.00,
}

# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------
def go_page(page: str):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Folder discovery (same pattern as rest of app)
# ---------------------------------------------------------------------------
def get_parquet_structure() -> dict:
    parquet_path = Path("data/parquet")
    structure = {}
    if not parquet_path.exists():
        return structure
    for type_dir in sorted(parquet_path.iterdir()):
        if not type_dir.is_dir():
            continue
        structure[type_dir.name] = {}
        for asset_dir in sorted(type_dir.iterdir()):
            if not asset_dir.is_dir():
                continue
            datasets = sorted([f.name for f in asset_dir.iterdir() if f.is_dir()])
            if datasets:
                structure[type_dir.name][asset_dir.name] = datasets
    return structure


def resolve_folder_path(structure: dict, key_prefix: str):
    asset_types = list(structure.keys())
    if not asset_types:
        st.error("No data found in data/parquet")
        return None, "", "", ""

    asset_type = st.selectbox("Type", asset_types, key=f"{key_prefix}_type")
    assets = list(structure.get(asset_type, {}).keys())
    if not assets:
        st.error(f"No assets under {asset_type}")
        return None, asset_type, "", ""
    asset = st.selectbox("Asset", assets, key=f"{key_prefix}_asset_{asset_type}")
    datasets = structure[asset_type].get(asset, [])
    if not datasets:
        st.error(f"No datasets under {asset_type}/{asset}")
        return None, asset_type, asset, ""
    dataset = st.selectbox("Dataset", datasets, key=f"{key_prefix}_dataset_{asset_type}_{asset}")
    folder_path = Path("data/parquet") / asset_type / asset / dataset
    return folder_path, asset_type, asset, dataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_days(folder_path: Path, end_date: pd.Timestamp, n_days: int) -> pd.DataFrame:
    """Load up to n_days of 1m parquet files ending on end_date."""
    files = sorted(folder_path.glob("*.parquet"))
    selected = []
    for f in reversed(files):
        try:
            file_date = pd.Timestamp(f.stem)
        except ValueError:
            continue
        if file_date.date() <= end_date.date():
            selected.append(f)
        if len(selected) == n_days:
            break
    if not selected:
        return pd.DataFrame()
    dfs = []
    for f in reversed(selected):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
TIMEFRAME_RULES = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1D":  "D",
    "1W":  "W",
    "1M":  "ME",
}

def resample_candles(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 1m DataFrame to target timeframe. Returns OHLCV DataFrame."""
    if df.empty:
        return df

    rule = TIMEFRAME_RULES.get(tf, "1min")

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Build volume safely (may not exist in all transforms)
    vol_col = "volume" if "volume" in df.columns else None
    buy_col = "buy_volume" if "buy_volume" in df.columns else None
    sell_col = "sell_volume" if "sell_volume" in df.columns else None

    agg: dict = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }
    if vol_col:
        agg["volume"] = "sum"
    if buy_col:
        agg["buy_volume"] = "sum"
    if sell_col:
        agg["sell_volume"] = "sum"

    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}

    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return out


# ---------------------------------------------------------------------------
# Serialise for JS
# ---------------------------------------------------------------------------
def df_to_chart_data(df: pd.DataFrame) -> list:
    """Convert OHLCV DataFrame to list of dicts for JS."""
    result = []
    for ts, row in df.iterrows():
        item = {
            "t":  ts.isoformat(),
            "o":  float(row["open"]),
            "h":  float(row["high"]),
            "l":  float(row["low"]),
            "c":  float(row["close"]),
            "vol": int(row["volume"])      if "volume"      in row.index else 0,
            "buy": int(row["buy_volume"])  if "buy_volume"  in row.index else 0,
            "sell":int(row["sell_volume"]) if "sell_volume" in row.index else 0,
        }
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Chart HTML/JS
# ---------------------------------------------------------------------------
CHART_JS_CLASS = r"""
// ============================================================
//  OHLCVChart — one self-contained canvas chart instance
// ============================================================
class OHLCVChart {
    constructor(containerId, candles, opts = {}) {
        this.id       = containerId;
        this.CANDLES  = candles;
        this.opts     = Object.assign({
            label:        '',
            sharedLines:  null,
            useShared:    false,
            tickSize:     0.25,
        }, opts);
        this.tickSize = this.opts.tickSize;

        // Drawing state — either local or pointing at shared store
        this._localLines = { h: [], v: [], d: [] };  // d = diagonal
        this.lines = this._localLines;

        this.drawMode    = null;
        this.diagStart   = null;   // { idx, price } for diagonal first click
        this.hoveredLine = null;

        // View state
        this.candleW   = 12;
        this.offsetX   = 0;
        this.scaleY    = 1.0;
        this.centerPrice = 0;
        this.priceMin  = 0;
        this.priceMax  = 0;

        // Layout constants
        this.VOL_PANEL_RATIO = 0.12;  // fraction of chart height
        this.PRICE_AXIS_W    = 65;
        this.TIME_AXIS_H     = 20;
        this.HIT             = 6;

        this._build();
        this._bindEvents();
        this.initView();
        this.resize();
    }

    // ----------------------------------------------------------------
    //  DOM construction
    // ----------------------------------------------------------------
    _build() {
        const container = document.getElementById(this.id);
        container.style.cssText = 'display:flex;flex-direction:column;height:100%;background:#111;';

        // Toolbar
        this.toolbar = document.createElement('div');
        this.toolbar.className = 'chart-toolbar';
        this.toolbar.innerHTML = this._toolbarHTML();
        container.appendChild(this.toolbar);

        // Canvas wrap
        this.wrap = document.createElement('div');
        this.wrap.style.cssText = 'flex:1;position:relative;overflow:hidden;';
        container.appendChild(this.wrap);

        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = 'display:block;width:100%;height:100%;cursor:default;';
        this.wrap.appendChild(this.canvas);

        this.ctx = this.canvas.getContext('2d');

        // Tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'chart-tooltip';
        this.tooltip.style.display = 'none';
        this.wrap.appendChild(this.tooltip);

        this._bindToolbar();
    }

    _toolbarHTML() {
        const label = this.opts.label ? `<span class="chart-label">${this.opts.label}</span><div class="tsep"></div>` : '';
        return `
        ${label}
        <span class="tlabel">scroll:pan  ctrl+scroll:zoom-x  shift+scroll:zoom-y  drag:pan</span>
        <div class="tsep"></div>
        <button class="tbtn" data-mode="hline">── H</button>
        <button class="tbtn" data-mode="vline">│ V</button>
        <button class="tbtn" data-mode="diag">╱ diag</button>
        <button class="tbtn danger" data-mode="delete">✕ del</button>
        <button class="tbtn" onclick="this.closest('[id]') && window.__charts__[this.closest('[id]').id] && window.__charts__[this.closest('[id]').id].clearAll()">clear</button>
        <div class="tsep"></div>
        <button class="tbtn" onclick="this.closest('[id]') && window.__charts__[this.closest('[id]').id] && window.__charts__[this.closest('[id]').id].resetView()">reset</button>
        `;
    }

    _bindToolbar() {
        this.toolbar.querySelectorAll('[data-mode]').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                this.setDrawMode(this.drawMode === mode ? null : mode);
            });
        });
        // Store reference globally so toolbar buttons can find this instance
        if (!window.__charts__) window.__charts__ = {};
        window.__charts__[this.id] = this;
    }

    // ----------------------------------------------------------------
    //  Drawing mode
    // ----------------------------------------------------------------
    setDrawMode(mode) {
        this.drawMode  = mode;
        this.diagStart = null;
        this.hoveredLine = null;
        this.toolbar.querySelectorAll('[data-mode]').forEach(btn => {
            btn.classList.remove('active', 'delete-mode');
            if (btn.dataset.mode === mode) {
                btn.classList.add(mode === 'delete' ? 'delete-mode' : 'active');
            }
        });
        this.canvas.style.cursor = mode ? 'crosshair' : 'default';
        this.draw();
    }

    clearAll() {
        this.lines.h.length = 0;
        this.lines.v.length = 0;
        this.lines.d.length = 0;
        this.diagStart = null;
        this.hoveredLine = null;
        this.draw();
    }

    // ----------------------------------------------------------------
    //  Shared drawings toggle
    // ----------------------------------------------------------------
    setShared(useShared, sharedStore) {
        this.opts.useShared  = useShared;
        this.opts.sharedLines = sharedStore;
        if (useShared && sharedStore) {
            this.lines = sharedStore;
        } else {
            this.lines = this._localLines;
        }
        this.draw();
    }

    // ----------------------------------------------------------------
    //  View helpers
    // ----------------------------------------------------------------
    get W() { return this.canvas.width; }
    get H() { return this.canvas.height; }

    volH()    { return Math.floor(this.H * this.VOL_PANEL_RATIO); }
    chartTop(){ return this.volH(); }
    chartBot(){ return this.H - this.TIME_AXIS_H; }
    chartH()  { return this.chartBot() - this.chartTop(); }

    candleX(i) { return this.PRICE_AXIS_W + i * (this.candleW + 2) - this.offsetX; }
    xToCandle(cx){ return Math.round((cx + this.offsetX - this.PRICE_AXIS_W) / (this.candleW + 2)); }

    priceToY(p) {
        const vis  = (this.priceMax - this.priceMin) / this.scaleY;
        const vMin = this.centerPrice - vis/2;
        const vMax = this.centerPrice + vis/2;
        return this.chartTop() + this.chartH() * (1 - (p - vMin) / (vMax - vMin));
    }

    yToPrice(cy) {
        const vis  = (this.priceMax - this.priceMin) / this.scaleY;
        const vMin = this.centerPrice - vis/2;
        const vMax = this.centerPrice + vis/2;
        return vMax - (cy - this.chartTop()) / this.chartH() * vis;
    }

    canvasXY(e) {
        const r = this.canvas.getBoundingClientRect();
        return { cx: e.clientX - r.left, cy: e.clientY - r.top };
    }

    initView() {
        if (!this.CANDLES.length) return;
        const all = [];
        this.CANDLES.forEach(c => all.push(c.h, c.l));
        this.priceMin    = Math.min(...all);
        this.priceMax    = Math.max(...all);
        this.centerPrice = (this.priceMin + this.priceMax) / 2;
        this.scaleY      = 1.0;
        const wrapW = this.wrap.clientWidth || 800;
        this.candleW  = Math.max(2, Math.min(80, wrapW / this.CANDLES.length * 0.85));
        this.offsetX  = 0;
    }

    resetView() { this.initView(); this.draw(); }

    resize() {
        this.canvas.width  = this.wrap.clientWidth;
        this.canvas.height = this.wrap.clientHeight;
        this.draw();
    }

    // ----------------------------------------------------------------
    //  Main draw
    // ----------------------------------------------------------------
    draw() {
        const ctx = this.ctx;
        if (!this.W || !this.H) return;
        ctx.clearRect(0, 0, this.W, this.H);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, this.W, this.H);

        this._drawVolumePanel();
        this._drawPriceAxis();
        this._drawTimeAxis();
        this._drawCandles();
        this._drawLines();
        if (this.drawMode === 'diag' && this.diagStart) this._drawDiagPreview();
    }

    // ----------------------------------------------------------------
    //  Volume panel
    // ----------------------------------------------------------------
    _drawVolumePanel() {
        const ctx = this.ctx;
        const ph  = this.volH();
        if (!ph) return;
        ctx.fillStyle = '#161616';
        ctx.fillRect(0, 0, this.W, ph);
        ctx.beginPath();
        ctx.moveTo(0, ph); ctx.lineTo(this.W, ph);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.stroke();

        const vols = this.CANDLES.map(c => c.vol);
        const maxV = Math.max(...vols) || 1;
        const barH = ph - 6;
        const pad  = 1;

        this.CANDLES.forEach((c, i) => {
            const x = this.candleX(i);
            if (x + this.candleW < this.PRICE_AXIS_W || x > this.W) return;
            const h   = Math.max(1, (c.vol / maxV) * barH);
            const bull = c.c >= c.o;
            ctx.fillStyle = bull ? 'rgba(60,160,100,0.6)' : 'rgba(180,60,60,0.6)';
            ctx.fillRect(
                x + pad,
                ph - h - 2,
                Math.max(1, this.candleW - pad*2),
                h
            );
        });

        // Volume label
        ctx.fillStyle = '#555';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('VOL', this.PRICE_AXIS_W + 4, 10);
    }

    // ----------------------------------------------------------------
    //  Price axis
    // ----------------------------------------------------------------
    _drawPriceAxis() {
        const ctx = this.ctx;
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, this.chartTop(), this.PRICE_AXIS_W, this.chartH());

        const vis  = (this.priceMax - this.priceMin) / this.scaleY;
        const vMin = this.centerPrice - vis/2;
        const vMax = this.centerPrice + vis/2;
        const step = this._niceStep(vis / 10);

        for (let p = Math.ceil(vMin/step)*step; p <= vMax; p += step) {
            const y = this.priceToY(p);
            if (y < this.chartTop() || y > this.chartBot()) continue;
            ctx.beginPath();
            ctx.moveTo(this.PRICE_AXIS_W, y); ctx.lineTo(this.W, y);
            ctx.strokeStyle = '#1e1e1e'; ctx.lineWidth = 0.5; ctx.stroke();
            ctx.fillStyle = '#666';
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.fillText(parseFloat(p.toPrecision(10)).toString(), this.PRICE_AXIS_W - 4, y + 3);
        }
    }

    // ----------------------------------------------------------------
    //  Time axis
    // ----------------------------------------------------------------
    _drawTimeAxis() {
        const ctx = this.ctx;
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, this.H - this.TIME_AXIS_H, this.W, this.TIME_AXIS_H);

        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';

        const step = Math.max(1, Math.floor(80 / this.candleW));
        this.CANDLES.forEach((c, i) => {
            if (i % step !== 0) return;
            const x = this.candleX(i) + this.candleW/2;
            if (x < this.PRICE_AXIS_W || x > this.W) return;
            const d   = new Date(c.t);
            const lbl = d.toLocaleString('en-US', {
                month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit',
                hour12: false, timeZone: 'America/New_York'
            });
            ctx.fillText(lbl, x, this.H - 5);
        });
    }

    // ----------------------------------------------------------------
    //  Candles
    // ----------------------------------------------------------------
    _drawCandles() {
        const ctx = this.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.PRICE_AXIS_W, this.chartTop(), this.W - this.PRICE_AXIS_W, this.chartH());
        ctx.clip();

        this.CANDLES.forEach((c, i) => {
            const x = this.candleX(i);
            if (x + this.candleW < this.PRICE_AXIS_W || x > this.W) return;
            const oy = this.priceToY(c.o), hy = this.priceToY(c.h);
            const ly = this.priceToY(c.l), cy = this.priceToY(c.c);
            const bull = c.c >= c.o;
            const midX = x + this.candleW/2;

            // Wick
            ctx.strokeStyle = bull ? '#3a9a5a' : '#aa4040';
            ctx.lineWidth = Math.max(0.5, this.candleW * 0.08);
            ctx.beginPath(); ctx.moveTo(midX, hy); ctx.lineTo(midX, ly); ctx.stroke();

            // Body
            ctx.fillStyle = bull ? '#2a7a4a' : '#8a3030';
            const bodyTop = Math.min(oy, cy);
            const bodyH   = Math.max(1, Math.abs(cy - oy));
            const bodyW   = Math.max(1, this.candleW * 0.7);
            const bodyX   = x + (this.candleW - bodyW) / 2;
            ctx.fillRect(bodyX, bodyTop, bodyW, bodyH);
        });

        ctx.restore();
    }

    // ----------------------------------------------------------------
    //  Lines drawing
    // ----------------------------------------------------------------
    _drawLines() {
        const ctx = this.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.PRICE_AXIS_W, this.chartTop(), this.W - this.PRICE_AXIS_W, this.chartH());
        ctx.clip();
        ctx.setLineDash([4, 4]);

        // Horizontal lines
        this.lines.h.forEach((line, i) => {
            const y   = this.priceToY(line.price);
            const hov = this.hoveredLine && this.hoveredLine.type === 'h' && this.hoveredLine.index === i;
            ctx.strokeStyle = hov ? '#f44' : '#ff0';
            ctx.lineWidth   = hov ? 2 : 1;
            ctx.beginPath(); ctx.moveTo(this.PRICE_AXIS_W, y); ctx.lineTo(this.W, y); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = hov ? '#f44' : '#cc0';
            ctx.font = '10px monospace'; ctx.textAlign = 'left';
            ctx.fillText(parseFloat(line.price.toPrecision(10)).toString(), this.PRICE_AXIS_W + 4, y - 3);
            ctx.setLineDash([4, 4]);
        });

        // Vertical lines
        this.lines.v.forEach((line, i) => {
            const x   = this.candleX(line.idx) + this.candleW/2;
            const hov = this.hoveredLine && this.hoveredLine.type === 'v' && this.hoveredLine.index === i;
            ctx.strokeStyle = hov ? '#f44' : '#f80';
            ctx.lineWidth   = hov ? 2 : 1;
            ctx.beginPath(); ctx.moveTo(x, this.chartTop()); ctx.lineTo(x, this.chartBot()); ctx.stroke();
        });

        // Diagonal lines
        this.lines.d.forEach((line, i) => {
            const x1 = this.candleX(line.idx1) + this.candleW/2;
            const y1 = this.priceToY(line.price1);
            const x2 = this.candleX(line.idx2) + this.candleW/2;
            const y2 = this.priceToY(line.price2);
            const hov = this.hoveredLine && this.hoveredLine.type === 'd' && this.hoveredLine.index === i;
            ctx.strokeStyle = hov ? '#f44' : '#0cf';
            ctx.lineWidth   = hov ? 2 : 1;

            // Extend line across full visible chart
            const dx = x2 - x1, dy = y2 - y1;
            if (Math.abs(dx) < 0.001) {
                ctx.beginPath(); ctx.moveTo(x1, this.chartTop()); ctx.lineTo(x1, this.chartBot()); ctx.stroke();
            } else {
                const slope = dy / dx;
                const xLeft  = this.PRICE_AXIS_W;
                const xRight = this.W;
                const yLeft  = y1 + slope * (xLeft  - x1);
                const yRight = y1 + slope * (xRight - x1);
                ctx.beginPath(); ctx.moveTo(xLeft, yLeft); ctx.lineTo(xRight, yRight); ctx.stroke();
            }
        });

        ctx.setLineDash([]);
        ctx.restore();
    }

    _drawDiagPreview() {
        if (!this.diagStart || !this._mousePosCanvas) return;
        const ctx = this.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.PRICE_AXIS_W, this.chartTop(), this.W - this.PRICE_AXIS_W, this.chartH());
        ctx.clip();
        const x1 = this.candleX(this.diagStart.idx) + this.candleW/2;
        const y1 = this.priceToY(this.diagStart.price);
        ctx.strokeStyle = 'rgba(0,200,255,0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(this._mousePosCanvas.cx, this._mousePosCanvas.cy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
    }

    // ----------------------------------------------------------------
    //  Hit detection
    // ----------------------------------------------------------------
    _getHoveredLine(cx, cy) {
        for (let i = 0; i < this.lines.h.length; i++) {
            if (Math.abs(cy - this.priceToY(this.lines.h[i].price)) <= this.HIT)
                return { type: 'h', index: i };
        }
        for (let i = 0; i < this.lines.v.length; i++) {
            const x = this.candleX(this.lines.v[i].idx) + this.candleW/2;
            if (Math.abs(cx - x) <= this.HIT)
                return { type: 'v', index: i };
        }
        for (let i = 0; i < this.lines.d.length; i++) {
            const ln = this.lines.d[i];
            const x1 = this.candleX(ln.idx1) + this.candleW/2;
            const y1 = this.priceToY(ln.price1);
            const x2 = this.candleX(ln.idx2) + this.candleW/2;
            const y2 = this.priceToY(ln.price2);
            if (this._distToSegment(cx, cy, x1, y1, x2, y2) <= this.HIT)
                return { type: 'd', index: i };
        }
        return null;
    }

    _distToSegment(px, py, x1, y1, x2, y2) {
        const dx = x2-x1, dy = y2-y1, len2 = dx*dx+dy*dy;
        if (len2 === 0) return Math.hypot(px-x1, py-y1);
        let t = ((px-x1)*dx+(py-y1)*dy)/len2;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px-(x1+t*dx), py-(y1+t*dy));
    }

    // ----------------------------------------------------------------
    //  Events
    // ----------------------------------------------------------------
    _bindEvents() {
        const C = this.canvas;
        let isDragging = false, dragStartX = 0, dragOffsetX = 0,
            dragStartY = 0, dragCenterPrice = 0;

        C.addEventListener('mousedown', e => {
            if (this.drawMode) return;
            isDragging = true;
            dragStartX = e.clientX; dragOffsetX = this.offsetX;
            dragStartY = this.canvasXY(e).cy; dragCenterPrice = this.centerPrice;
        });

        C.addEventListener('click', e => {
            if (!this.drawMode) return;
            const { cx, cy } = this.canvasXY(e);
            const idx   = Math.max(0, Math.min(this.CANDLES.length-1, this.xToCandle(cx)));
            const price = this.yToPrice(cy);

            if (this.drawMode === 'hline') {
                const tickSize = this.tickSize;
                const snapped  = parseFloat((Math.round(price / tickSize) * tickSize).toPrecision(10));
                this.lines.h.push({ price: snapped });
                this.setDrawMode(null);

            } else if (this.drawMode === 'vline') {
                if (idx >= 0 && idx < this.CANDLES.length)
                    this.lines.v.push({ idx });
                this.setDrawMode(null);

            } else if (this.drawMode === 'diag') {
                if (!this.diagStart) {
                    this.diagStart = { idx, price };
                } else {
                    this.lines.d.push({
                        idx1: this.diagStart.idx, price1: this.diagStart.price,
                        idx2: idx,                price2: price,
                    });
                    this.diagStart = null;
                    this.setDrawMode(null);
                }

            } else if (this.drawMode === 'delete') {
                const hit = this._getHoveredLine(cx, cy);
                if (hit) {
                    this.lines[hit.type].splice(hit.index, 1);
                    this.hoveredLine = null;
                }
            }
            this.draw();
        });

        C.addEventListener('mousemove', e => {
            const { cx, cy } = this.canvasXY(e);
            this._mousePosCanvas = { cx, cy };

            if (isDragging) {
                this.offsetX = dragOffsetX - (e.clientX - dragStartX);
                const dy = cy - dragStartY;
                this.centerPrice = dragCenterPrice + dy * (this.priceMax - this.priceMin) / this.scaleY / this.chartH();
                this.draw();
                return;
            }

            if (this.drawMode === 'delete') {
                const hit = this._getHoveredLine(cx, cy);
                if (hit !== this.hoveredLine) {
                    this.hoveredLine = hit;
                    C.style.cursor = hit ? 'pointer' : 'crosshair';
                    this.draw();
                }
                return;
            }

            if (this.drawMode === 'diag' && this.diagStart) {
                this.draw();
                return;
            }

            this._showTooltip(e, cx, cy);
        });

        C.addEventListener('mouseup', () => { isDragging = false; });
        C.addEventListener('mouseleave', () => {
            isDragging = false;
            this.tooltip.style.display = 'none';
            this._mousePosCanvas = null;
            if (this.hoveredLine) { this.hoveredLine = null; this.draw(); }
        });

        C.addEventListener('wheel', e => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.88 : 1.14;
            const { cx }  = this.canvasXY(e);
            if (e.ctrlKey) {
                const idx   = this.xToCandle(cx);
                const prevX = this.candleX(idx);
                this.candleW = Math.max(1, Math.min(300, this.candleW * factor));
                this.offsetX += this.candleX(idx) - prevX;
            } else if (e.shiftKey) {
                const mp = this.yToPrice(this.canvasXY(e).cy);
                this.scaleY = Math.max(0.1, Math.min(50, this.scaleY * factor));
                this.centerPrice += (this.canvasXY(e).cy - this.priceToY(mp))
                    * (this.priceMax - this.priceMin) / this.scaleY / this.chartH();
            } else {
                this.offsetX += e.deltaY > 0 ? -70 : 70;
            }
            this.draw();
        }, { passive: false });
    }

    // ----------------------------------------------------------------
    //  Tooltip
    // ----------------------------------------------------------------
    _showTooltip(e, cx, cy) {
        const idx = this.xToCandle(cx);
        if (idx < 0 || idx >= this.CANDLES.length) {
            this.tooltip.style.display = 'none'; return;
        }
        const c  = this.CANDLES[idx];
        const d  = new Date(c.t);
        const ds = d.toLocaleString('en-US', {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit',
            hour12: false, timeZone: 'America/New_York'
        });

        const fmt = v => parseFloat(v.toPrecision(10)).toString();

        let buyHtml = '', sellHtml = '';
        if (c.buy  > 0) buyHtml  = `<div style="color:#5c5">Buy: <b>${c.buy.toLocaleString()}</b></div>`;
        if (c.sell > 0) sellHtml = `<div style="color:#c55">Sell: <b>${c.sell.toLocaleString()}</b></div>`;

        this.tooltip.innerHTML = `
            <div style="color:#888;margin-bottom:4px">${ds}</div>
            <div>O: <b>${fmt(c.o)}</b></div>
            <div>H: <b>${fmt(c.h)}</b></div>
            <div>L: <b>${fmt(c.l)}</b></div>
            <div>C: <b>${fmt(c.c)}</b></div>
            <div style="color:#9af">Vol: <b>${c.vol.toLocaleString()}</b></div>
            ${buyHtml}${sellHtml}
        `;
        this.tooltip.style.display = 'block';
        const W = this.W, H = this.H;
        const tw = 160, th = 140;
        this.tooltip.style.left = (Math.min(cx + 14, W - tw)) + 'px';
        this.tooltip.style.top  = (Math.max(Math.min(cy - 10, H - th), this.chartTop())) + 'px';
    }

    // ----------------------------------------------------------------
    //  Utils
    // ----------------------------------------------------------------
    _niceStep(rough) {
        const mag  = Math.pow(10, Math.floor(Math.log10(rough)));
        const norm = rough / mag;
        if (norm < 1.5) return mag;
        if (norm < 3)   return 2 * mag;
        if (norm < 7)   return 5 * mag;
        return 10 * mag;
    }
}
"""

CHART_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #111; color: #ccc; font-family: monospace; height: 100%; overflow: hidden; }

#root {
    display: flex;
    flex-direction: column;
    height: 100vh;
    gap: 0;
}

/* ── sync toggle bar ── */
#sync-bar {
    display: __SYNC_DISPLAY__;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    background: #181818;
    border-bottom: 1px solid #2a2a2a;
    flex-shrink: 0;
}
#sync-bar button {
    font-size: 11px; padding: 3px 10px; cursor: pointer;
    background: #2a2a2a; border: 1px solid #444; color: #ccc; border-radius: 3px;
}
#sync-bar button.active { background: #1a3a1a; border-color: #4a9a4a; color: #8aca8a; }
#sync-bar span { font-size: 10px; color: #555; }

/* ── chart slots ── */
#charts {
    flex: 1;
    display: flex;
    flex-direction: __CHART_DIRECTION__;
    min-height: 0;
    gap: __CHART_GAP__;
}

.chart-slot {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    border: 1px solid #222;
}

/* ── toolbar ── */
.chart-toolbar {
    display: flex; align-items: center; flex-wrap: wrap; gap: 5px;
    padding: 4px 8px; background: #1a1a1a; border-bottom: 1px solid #2a2a2a;
    flex-shrink: 0;
}
.chart-label {
    font-size: 11px; color: #9af; font-weight: bold; white-space: nowrap;
}
.tsep { width: 1px; height: 16px; background: #333; margin: 0 2px; }
.tlabel { font-size: 9px; color: #444; white-space: nowrap; }
.tbtn {
    font-size: 10px; padding: 2px 7px; cursor: pointer;
    background: #252525; border: 1px solid #3a3a3a; color: #aaa; border-radius: 2px;
    white-space: nowrap;
}
.tbtn:hover  { background: #333; border-color: #555; }
.tbtn.active { background: #2a3a5a; border-color: #5577bb; color: #99bbff; }
.tbtn.danger { color: #f88; border-color: #522; }
.tbtn.danger:hover { background: #3a1a1a; border-color: #844; }
.tbtn.delete-mode { background: #3a1a1a; border-color: #cc4444; color: #ff8888; }

/* ── canvas tooltip ── */
.chart-tooltip {
    position: absolute; pointer-events: none;
    background: #1c1c1c; border: 1px solid #444;
    padding: 7px 10px; border-radius: 3px;
    font-size: 11px; color: #ddd; z-index: 10;
    min-width: 150px; line-height: 1.75;
}
</style>
</head>
<body>
<div id="root">
    <div id="sync-bar">
        <span>Drawings:</span>
        <button id="sync-btn" onclick="toggleSync()">⬡ sync off</button>
        <span id="sync-hint">Enable to share H/V/diagonal lines between both charts</span>
    </div>
    <div id="charts">
        <div class="chart-slot" id="chart-a"></div>
        __CHART_B_SLOT__
    </div>
</div>

<script>
__CHART_JS__

// ── data injected from Python ──────────────────────────────────
const CANDLES_A = __CANDLES_A__;
const CANDLES_B = __CANDLES_B__;
const DUAL      = __DUAL__;

// Shared drawing store (used when sync is on)
const sharedStore = { h: [], v: [], d: [] };
let syncOn = false;

// ── instantiate charts ─────────────────────────────────────────
const chartA = new OHLCVChart('chart-a', CANDLES_A, { label: __LABEL_A__, tickSize: __TICK_SIZE_A__ });
let   chartB = null;
if (DUAL && CANDLES_B.length) {
    chartB = new OHLCVChart('chart-b', CANDLES_B, { label: __LABEL_B__, tickSize: __TICK_SIZE_B__ });
}

// ── sync toggle ────────────────────────────────────────────────
function toggleSync() {
    syncOn = !syncOn;
    const btn = document.getElementById('sync-btn');
    btn.classList.toggle('active', syncOn);
    btn.textContent = syncOn ? '⬡ sync ON' : '⬡ sync off';
    chartA.setShared(syncOn, syncOn ? sharedStore : null);
    if (chartB) chartB.setShared(syncOn, syncOn ? sharedStore : null);
}

// ── resize observer ────────────────────────────────────────────
const ro = new ResizeObserver(() => {
    chartA.resize();
    if (chartB) chartB.resize();
});
ro.observe(document.getElementById('charts'));
</script>
</body>
</html>
"""


def build_html(
    candles_a: list,
    candles_b: list,
    label_a: str = "",
    label_b: str = "",
    dual: bool = False,
    tick_size_a=0.25,
    tick_size_b=0.25
) -> str:
    html = CHART_HTML_TEMPLATE
    html = html.replace("__CHART_JS__",       CHART_JS_CLASS)
    html = html.replace("__CANDLES_A__",      json.dumps(candles_a))
    html = html.replace("__TICK_SIZE_A__", str(tick_size_a))
    html = html.replace("__CANDLES_B__",      json.dumps(candles_b))
    html = html.replace("__TICK_SIZE_B__", str(tick_size_b))
    html = html.replace("__DUAL__",           "true" if dual else "false")
    html = html.replace("__LABEL_A__",        json.dumps(label_a))
    html = html.replace("__LABEL_B__",        json.dumps(label_b))
    html = html.replace("__SYNC_DISPLAY__",   "flex" if dual else "none")
    html = html.replace("__CHART_DIRECTION__","row")
    html = html.replace("__CHART_GAP__",      "2px" if dual else "0")
    if dual:
        html = html.replace("__CHART_B_SLOT__", '<div class="chart-slot" id="chart-b"></div>')
    else:
        html = html.replace("__CHART_B_SLOT__", "")
    return html


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"]


def _chart_controls(structure: dict, prefix: str, default_tf: str = "1m") -> tuple:
    """Render controls for one chart pane. Returns (candles_data, label_str)."""
    col_path, col_date, col_days, col_tf = st.columns([3, 1.5, 1, 2])

    with col_path:
        folder_path, asset_type, asset, dataset = resolve_folder_path(structure, key_prefix=prefix)
    if folder_path is None:
        return [], ""

    available_dates = sorted([
        pd.Timestamp(f.stem) for f in folder_path.glob("*.parquet")
        if f.stem[:4].isdigit()
    ])
    if not available_dates:
        st.error("No parquet files found")
        return [], ""

    with col_date:
        end_date = st.date_input(
            "End date",
            value=available_dates[-1].date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            key=f"{prefix}_end",
        )

    with col_days:
        n_days = st.number_input("Days", min_value=1, max_value=365, value=1, key=f"{prefix}_ndays")

    with col_tf:
        # Timeframe buttons
        st.write("Timeframe")
        if f"{prefix}_tf" not in st.session_state:
            st.session_state[f"{prefix}_tf"] = default_tf
        btn_cols = st.columns(len(TIMEFRAMES))
        for j, tf in enumerate(TIMEFRAMES):
            with btn_cols[j]:
                active = st.session_state[f"{prefix}_tf"] == tf
                style  = "primary" if active else "secondary"
                if st.button(tf, key=f"{prefix}_tf_{tf}", type=style):
                    st.session_state[f"{prefix}_tf"] = tf
                    st.rerun()

    selected_tf = st.session_state[f"{prefix}_tf"]

    with st.spinner("Loading…"):
        df_raw = load_days(folder_path, pd.Timestamp(end_date), n_days)

    if df_raw.empty:
        st.warning("No data found for selected range.")
        return [], ""

    df = resample_candles(df_raw, selected_tf)
    if df.empty:
        st.warning("No candles after resampling.")
        return [], ""

    candles   = df_to_chart_data(df)
    label     = f"{asset} · {dataset} · {selected_tf}"
    tick_size = TICK_SIZES.get(asset.upper(), 0.25)
    return candles, label, tick_size


def render():
    if st.button("← Back"):
        go_page("home")

    st.title("OHLCV Chart")

    structure = get_parquet_structure()
    if not structure:
        st.error("No data found in data/parquet")
        return

    # ── Dual chart toggle ────────────────────────────────────────────────────
    dual = st.toggle("Add second chart", value=False, key="ohlcv_dual")
    st.divider()

    # ── Chart A ──────────────────────────────────────────────────────────────
    st.markdown("**Chart A**")
    candles_a, label_a, tick_size_a = _chart_controls(structure, prefix="oa", default_tf="1m")

    candles_b, label_b, tick_size_b = [], "", 0.25
    if dual:
        st.divider()
        st.markdown("**Chart B**")
        candles_b, label_b, tick_size_b = _chart_controls(structure, prefix="ob", default_tf="5m")

    st.divider()

    if not candles_a:
        return

    # ── Render ───────────────────────────────────────────────────────────────
    chart_height = 1200 if not dual else 1200
    html = build_html(candles_a, candles_b, label_a, label_b, dual=dual, tick_size_a=tick_size_a, tick_size_b=tick_size_b)
    st.iframe(html, height=chart_height)