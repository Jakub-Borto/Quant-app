import streamlit as st
import pandas as pd
import json
from pathlib import Path


CVD_PANEL_RATIO = 0.20

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

def go_page(page: str):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Folder discovery
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


def resolve_folder_path(structure: dict, key_prefix: str) -> tuple:
    asset_types = list(structure.keys())
    asset_type  = st.selectbox("Type", asset_types, key=f"{key_prefix}_type")

    assets = list(structure.get(asset_type, {}).keys())
    if not assets:
        st.error(f"No assets found under {asset_type}")
        return None, asset_type, "", ""
    asset = st.selectbox("Asset", assets, key=f"{key_prefix}_asset_{asset_type}")

    datasets = structure[asset_type].get(asset, [])
    if not datasets:
        st.error(f"No datasets found under {asset_type}/{asset}")
        return None, asset_type, asset, ""
    dataset = st.selectbox("Dataset", datasets, key=f"{key_prefix}_dataset_{asset_type}_{asset}")

    folder_path = Path("data/parquet") / asset_type / asset / dataset
    return folder_path, asset_type, asset, dataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_days(folder_path: Path, end_date: pd.Timestamp, n_days: int) -> pd.DataFrame:
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
    return pd.concat([pd.read_parquet(f) for f in reversed(selected)])


def load_big_trades(folder_path: Path, end_date: pd.Timestamp, n_days: int) -> list:
    try:
        df = load_days(folder_path, end_date, n_days)
        if df.empty:
            return []
        return [
            {"t": ts.isoformat(), "p": float(row["price"]),
             "s": int(row["size"]), "d": str(row["side"])}
            for ts, row in df.iterrows()
        ]
    except Exception:
        return []


VWAP_GROUPS = {
    "bar_globex": {
        "vwap": "vwap_bar_globex",
        "bands": [
            (1, "vwap_bar_globex_std1_up", "vwap_bar_globex_std1_dn"),
            (2, "vwap_bar_globex_std2_up", "vwap_bar_globex_std2_dn"),
            (3, "vwap_bar_globex_std3_up", "vwap_bar_globex_std3_dn"),
        ],
    },
    "bar_rth": {
        "vwap": "vwap_bar_rth",
        "bands": [
            (1, "vwap_bar_rth_std1_up", "vwap_bar_rth_std1_dn"),
            (2, "vwap_bar_rth_std2_up", "vwap_bar_rth_std2_dn"),
            (3, "vwap_bar_rth_std3_up", "vwap_bar_rth_std3_dn"),
        ],
    },
    "tick_globex": {
        "vwap": "vwap_tick_globex",
        "bands": [
            (1, "vwap_tick_globex_std1_up", "vwap_tick_globex_std1_dn"),
            (2, "vwap_tick_globex_std2_up", "vwap_tick_globex_std2_dn"),
            (3, "vwap_tick_globex_std3_up", "vwap_tick_globex_std3_dn"),
        ],
    },
    "tick_rth": {
        "vwap": "vwap_tick_rth",
        "bands": [
            (1, "vwap_tick_rth_std1_up", "vwap_tick_rth_std1_dn"),
            (2, "vwap_tick_rth_std2_up", "vwap_tick_rth_std2_dn"),
            (3, "vwap_tick_rth_std3_up", "vwap_tick_rth_std3_dn"),
        ],
    },
}


def load_indicators(folder_path: Path, end_date: pd.Timestamp, n_days: int) -> list:
    try:
        df = load_days(folder_path, end_date, n_days)
        if df.empty:
            return []
        all_cols = ["cumulative_delta", "absorption_score", "beta", "residual"]
        for group in VWAP_GROUPS.values():
            all_cols.append(group["vwap"])
            for _, up, dn in group["bands"]:
                all_cols += [up, dn]
        present_cols = [c for c in all_cols if c in df.columns]
        result = []
        for ts, row in df.iterrows():
            entry = {"t": ts.isoformat()}
            for col in present_cols:
                val = row[col]
                entry[col] = float(val) if pd.notna(val) else None
            result.append(entry)
        return result
    except Exception:
        return []


def df_to_chart_data(df: pd.DataFrame) -> list:
    candles = []
    for ts, row in df.iterrows():
        tick_vol = json.loads(row["tick_volume"]) if row["tick_volume"] else {}
        passive  = json.loads(row["passive_orders"]) if row["passive_orders"] else {}
        candles.append({
            "t":              ts.isoformat(),
            "o":              round(float(row["open"]),   8),
            "h":              round(float(row["high"]),   8),
            "l":              round(float(row["low"]),    8),
            "c":              round(float(row["close"]),  8),
            "vol":            int(row["volume"]),
            "buy_vol":        int(row["buy_volume"]),
            "sell_vol":       int(row["sell_volume"]),
            "delta":          int(row["volume_delta"]),
            "delta_pct":      round(float(row["volume_delta_pct"]), 4),
            "tick_volume":    tick_vol,
            "passive_orders": passive,
        })
    return candles


# ---------------------------------------------------------------------------
# HTML / JS chart
# ---------------------------------------------------------------------------
CHART_HTML = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ccc; font-family: monospace; overflow: hidden; }
#wrap { display: flex; flex-direction: column; height: 100vh; }

#toolbar {
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    padding: 5px 10px; background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0;
}
#toolbar span.label { font-size: 10px; color: #666; white-space: nowrap; }
#toolbar button {
    font-size: 11px; padding: 3px 8px; cursor: pointer;
    background: #2a2a2a; border: 1px solid #444; color: #ccc; border-radius: 3px;
}
#toolbar button.active      { background: #3a3a6a; border-color: #6666cc; color: #aaaaff; }
#toolbar button.delete-mode { background: #3a1a1a; border-color: #cc4444; color: #ff8888; }
#toolbar button.edit-mode   { background: #3a2a1a; border-color: #cc8844; color: #ffcc88; }
#toolbar button.vp-mode     { background: #1a3a1a; border-color: #44cc44; color: #88ff88; }
#toolbar button:disabled    { opacity: 0.35; cursor: not-allowed; }
#toolbar input[type=number] {
    width: 46px; font-size: 11px; font-family: monospace;
    background: #2a2a2a; border: 1px solid #444; color: #ccc;
    border-radius: 3px; padding: 2px 4px; text-align: center;
}
#toolbar select {
    font-size: 11px; font-family: monospace;
    background: #2a2a2a; border: 1px solid #444; color: #ccc;
    border-radius: 3px; padding: 2px 4px;
}
#toolbar .sep { width: 1px; height: 16px; background: #333; margin: 0 2px; }
.std-group { display: flex; gap: 3px; align-items: center; }
.std-group button { padding: 2px 6px; font-size: 10px; }
.std-group button.active { background: #2a4a2a; border-color: #4a9a4a; color: #8aca8a; }

#canvas-wrap { flex: 1; position: relative; overflow: hidden; }
canvas { display: block; width: 100%; height: 100%; }
#tooltip {
    position: absolute; pointer-events: none; display: none;
    background: #1e1e1e; border: 1px solid #555; padding: 8px 10px;
    border-radius: 4px; font-size: 11px; color: #ddd; z-index: 10;
    min-width: 170px; line-height: 1.8;
}
#vp-hint {
    position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
    background: #1a3a1a; border: 1px solid #44cc44; color: #88ff88;
    font-size: 11px; padding: 4px 12px; border-radius: 3px;
    pointer-events: none; display: none;
}
</style>

<div id="wrap">
  <div id="toolbar">
    <span class="label">scroll:pan | ctrl+scroll:zoom x | shift+scroll:zoom y | drag:pan</span>
    <div class="sep"></div>
    <button id="btn-footprint" onclick="toggleLayer('footprint')">footprint</button>
    <button id="btn-passive"   onclick="toggleLayer('passive')">passive</button>
    <span class="label">highlight≥</span>
    <input type="number" id="passive-thresh" value="100" min="1" oninput="draw()">
    <div class="sep"></div>
    <button id="btn-bigTrades" onclick="toggleLayer('bigTrades')" __BIG_TRADES_DISABLED__>big trades</button>
    <span class="label">ETH≥</span>
    <input type="number" id="eth-thresh" value="30" min="1" oninput="onThresholdChange()">
    <span class="label">RTH≥</span>
    <input type="number" id="rth-thresh" value="50" min="1" oninput="onThresholdChange()">
    <div class="sep"></div>
    <button id="btn-vwap" onclick="toggleLayer('vwap')" __INDICATORS_DISABLED__>VWAP</button>
    <select id="vwap-select" onchange="draw()" __INDICATORS_DISABLED__>
      <option value="bar_globex">bar · globex</option>
      <option value="bar_rth">bar · RTH</option>
      <option value="tick_globex">tick · globex</option>
      <option value="tick_rth">tick · RTH</option>
    </select>
    <span class="label">std:</span>
    <div class="std-group">
      <button id="btn-std1" onclick="toggleStd(1)" __INDICATORS_DISABLED__>1σ</button>
      <button id="btn-std2" onclick="toggleStd(2)" __INDICATORS_DISABLED__>2σ</button>
      <button id="btn-std3" onclick="toggleStd(3)" __INDICATORS_DISABLED__>3σ</button>
    </div>
    <div class="sep"></div>
    <button id="btn-cvd" onclick="toggleLayer('cvd')" __INDICATORS_DISABLED__>CVD</button>
    <button id="btn-absorption" onclick="toggleLayer('absorption')" __INDICATORS_DISABLED__>absorption</button>
    <span class="label">thresh≥</span>
    <input type="number" id="absorption-thresh" value="2.0" min="0.1" step="0.1" oninput="draw()">
    <div class="sep"></div>
    <button id="btn-vp"     onclick="setDrawMode('vp')">+ VP</button>
    <span class="label">VP vol</span>
    <input type="range" id="vp-vol-width"   min="20" max="1000" value="80"  style="width:100px" oninput="draw()">
    <span class="label">VP delta</span>
    <input type="range" id="vp-delta-width" min="20" max="1800" value="80"  style="width:100px" oninput="draw()">
    <button id="btn-hline"  onclick="setDrawMode('hline')">+ H line</button>
    <button id="btn-vline"  onclick="setDrawMode('vline')">+ V line</button>
    <button id="btn-edit"   onclick="setDrawMode('edit')">✎ edit</button>
    <button id="btn-delete" onclick="setDrawMode('delete')">delete</button>
    <button onclick="clearAll()">clear all</button>
    <div class="sep"></div>
    <button onclick="resetView()">reset</button>
  </div>

  <div id="canvas-wrap">
    <canvas id="c"></canvas>
    <div id="tooltip"></div>
    <div id="vp-hint"></div>
  </div>
</div>

<script>
const CANDLES    = __CANDLES__;
const BIG_TRADES = __BIG_TRADES__;
const INDICATORS = __INDICATORS__;
const CVD_RATIO  = __CVD_RATIO__;
const fmt = v => parseFloat(v.toPrecision(10)).toString();

const VWAP_COLS = {
    bar_globex:  { vwap: "vwap_bar_globex",  bands: [[1,"vwap_bar_globex_std1_up","vwap_bar_globex_std1_dn"],[2,"vwap_bar_globex_std2_up","vwap_bar_globex_std2_dn"],[3,"vwap_bar_globex_std3_up","vwap_bar_globex_std3_dn"]] },
    bar_rth:     { vwap: "vwap_bar_rth",     bands: [[1,"vwap_bar_rth_std1_up","vwap_bar_rth_std1_dn"],[2,"vwap_bar_rth_std2_up","vwap_bar_rth_std2_dn"],[3,"vwap_bar_rth_std3_up","vwap_bar_rth_std3_dn"]] },
    tick_globex: { vwap: "vwap_tick_globex", bands: [[1,"vwap_tick_globex_std1_up","vwap_tick_globex_std1_dn"],[2,"vwap_tick_globex_std2_up","vwap_tick_globex_std2_dn"],[3,"vwap_tick_globex_std3_up","vwap_tick_globex_std3_dn"]] },
    tick_rth:    { vwap: "vwap_tick_rth",    bands: [[1,"vwap_tick_rth_std1_up","vwap_tick_rth_std1_dn"],[2,"vwap_tick_rth_std2_up","vwap_tick_rth_std2_dn"],[3,"vwap_tick_rth_std3_up","vwap_tick_rth_std3_dn"]] },
};
const VWAP_STYLE = {
    bar_globex:  { line: '#f90',  band: ['#f907','#f904','#f902'] },
    bar_rth:     { line: '#0cf',  band: ['#0cf7','#0cf4','#0cf2'] },
    tick_globex: { line: '#f66',  band: ['#f667','#f664','#f662'] },
    tick_rth:    { line: '#6f6',  band: ['#6f67','#6f64','#6f62'] },
};

const C       = document.getElementById('c');
const ctx     = C.getContext('2d');
const wrap    = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');
const vpHint  = document.getElementById('vp-hint');

const layers    = { footprint: false, passive: false, bigTrades: false, vwap: false, cvd: false, absorption: false };
const stdActive = { 1: false, 2: false, 3: false };

function toggleLayer(name) {
    layers[name] = !layers[name];
    const btn = document.getElementById('btn-' + name);
    if (btn) btn.classList.toggle('active', layers[name]);
    resize();
}
function toggleStd(n) {
    stdActive[n] = !stdActive[n];
    document.getElementById('btn-std' + n).classList.toggle('active', stdActive[n]);
    draw();
}
function getVwapKey() { return document.getElementById('vwap-select').value; }

let W = 0, H = 0;
const DELTA_H = 28, PRICE_AXIS_W = 60, TIME_AXIS_H = 20, HIT_THRESHOLD = 6;
const RTH_HOUR = 9, RTH_MINUTE = 30;
const VP_OPACITY = 0.75;
const HANDLE_R = 5; // px radius of VP drag handles

let cvdH = 0, offsetX = 0, candleW = 80;
let priceMin = 0, priceMax = 0, tickSize = __TICK_SIZE__;
let scaleY = 1.0, centerPrice = 0;
let drawMode = null, hoveredLine = null;
const hLines = [], vLines = [], profiles = [];
let vpStartIdx = null;
let _mouseX = 0, _mouseY = 0;

// ---------------------------------------------------------------------------
// Edit state
// ---------------------------------------------------------------------------
// dragging: { type: 'h'|'v'|'vp_start'|'vp_end', index: N }
let editDrag = null;
// which handle is hovered in edit mode
let editHover = null;

// ---------------------------------------------------------------------------
// Threshold helpers
// ---------------------------------------------------------------------------
function getEthThresh() { return parseInt(document.getElementById('eth-thresh').value) || 30; }
function getRthThresh() { return parseInt(document.getElementById('rth-thresh').value) || 50; }
function isRthTime(iso) {
    const t = new Date(iso);
    const h = +t.toLocaleTimeString('en-US', { hour: '2-digit', hour12: false, timeZone: 'America/New_York' });
    const m = +t.toLocaleTimeString('en-US', { minute: '2-digit', timeZone: 'America/New_York' });
    return h > RTH_HOUR || (h === RTH_HOUR && m >= RTH_MINUTE);
}
function tradePassesThreshold(trade) {
    return trade.s >= (isRthTime(trade.t) ? getRthThresh() : getEthThresh());
}
function onThresholdChange() { rebuildTradeIndex(); draw(); }

// ---------------------------------------------------------------------------
// Big trades index
// ---------------------------------------------------------------------------
let tradesByCandleIdx = {}, maxTradeSize = 1;
function rebuildTradeIndex() {
    tradesByCandleIdx = {};
    if (!BIG_TRADES.length || !CANDLES.length) return;
    const candleTimes = CANDLES.map(c => new Date(c.t).getTime());
    let localMax = 1;
    BIG_TRADES.forEach(trade => {
        if (!tradePassesThreshold(trade)) return;
        const ms = new Date(trade.t).getTime();
        let lo = 0, hi = candleTimes.length - 1, idx = 0;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (candleTimes[mid] <= ms) { idx = mid; lo = mid + 1; } else hi = mid - 1;
        }
        if (idx + 1 < candleTimes.length &&
            Math.abs(candleTimes[idx+1]-ms) < Math.abs(candleTimes[idx]-ms)) idx++;
        if (!tradesByCandleIdx[idx]) tradesByCandleIdx[idx] = [];
        tradesByCandleIdx[idx].push(trade);
        if (trade.s > localMax) localMax = trade.s;
    });
    maxTradeSize = localMax;
}
rebuildTradeIndex();

// ---------------------------------------------------------------------------
// Volume profile computation
// ---------------------------------------------------------------------------
function computeProfile(startIdx, endIdx) {
    if (startIdx > endIdx) [startIdx, endIdx] = [endIdx, startIdx];
    const levelMap = {};
    for (let i = startIdx; i <= endIdx; i++) {
        const c = CANDLES[i];
        if (!c || !c.tick_volume) continue;
        Object.entries(c.tick_volume).forEach(([priceStr, val]) => {
            if (!Array.isArray(val) || val.length < 2) return;
            const p   = parseFloat(priceStr);
            const key = p.toPrecision(10);
            const buy = val[0], sell = val[1];
            if (!levelMap[key]) levelMap[key] = { p, total: 0, buy: 0, sell: 0 };
            levelMap[key].total += buy + sell;
            levelMap[key].buy   += buy;
            levelMap[key].sell  += sell;
        });
    }
    const entries = Object.values(levelMap);
    if (!entries.length) return null;
    entries.sort((a, b) => a.p - b.p);
    const prices  = entries.map(e => e.p);
    const volumes = entries.map(e => e.total);
    const n       = prices.length;
    const totalVolume = volumes.reduce((s, v) => s + v, 0);
    if (totalVolume === 0) return null;

    const smoothed = new Array(n);
    for (let i = 0; i < n; i++) {
        const lo = Math.max(0, i - 1), hi = Math.min(n - 1, i + 1);
        let sum = 0, count = 0;
        for (let j = lo; j <= hi; j++) { sum += volumes[j]; count++; }
        smoothed[i] = sum / count;
    }

    const peakIndices = [];
    for (let i = 1; i < n - 1; i++)
        if (smoothed[i] > smoothed[i-1] && smoothed[i] > smoothed[i+1]) peakIndices.push(i);
    if (n >= 2 && smoothed[0]   > smoothed[1])   peakIndices.unshift(0);
    if (n >= 2 && smoothed[n-1] > smoothed[n-2]) peakIndices.push(n-1);
    if (n === 1) peakIndices.push(0);
    if (!peakIndices.length) peakIndices.push(volumes.indexOf(Math.max(...volumes)));

    peakIndices.sort((a, b) => a - b);
    const clusters = [];
    for (const idx of peakIndices) {
        const last = clusters[clusters.length - 1];
        if (last && idx - last.bestIdx <= 4) {
            if (smoothed[idx] > smoothed[last.bestIdx]) last.bestIdx = idx;
        } else {
            clusters.push({ bestIdx: idx });
        }
    }
    clusters.sort((a, b) => smoothed[b.bestIdx] - smoothed[a.bestIdx]);
    const topClusters = clusters.slice(0, 5);

    const candidates = topClusters.map(cl => {
        const lo = Math.max(0, cl.bestIdx - 3), hi = Math.min(n-1, cl.bestIdx + 3);
        let bestRawIdx = lo;
        for (let j = lo+1; j <= hi; j++)
            if (volumes[j] > volumes[bestRawIdx]) bestRawIdx = j;
        return bestRawIdx;
    });

    const target = totalVolume * 0.70;
    function expandVA(pocIdx) {
        let lo = pocIdx, hi = pocIdx, cumVol = volumes[pocIdx];
        while (cumVol < target) {
            const downVol = lo > 0     ? volumes[lo-1] : 0;
            const upVol   = hi < n-1   ? volumes[hi+1] : 0;
            if (!downVol && !upVol) break;
            if (!downVol || (upVol && upVol >= downVol)) { hi++; cumVol += upVol; }
            else                                          { lo--; cumVol += downVol; }
        }
        return { loIdx: lo, hiIdx: hi };
    }

    let bestResult = null, tightestRange = Infinity;
    for (const pocIdx of candidates) {
        const { loIdx, hiIdx } = expandVA(pocIdx);
        const range = prices[hiIdx] - prices[loIdx];
        if (range < tightestRange) {
            tightestRange = range;
            let truePoc = prices[pocIdx], truePocVol = 0;
            for (let j = loIdx; j <= hiIdx; j++) {
                if (volumes[j] > truePocVol) { truePocVol = volumes[j]; truePoc = prices[j]; }
            }
            bestResult = {
                startIdx, endIdx,
                levels: levelMap,
                poc:    truePoc,
                vah:    prices[hiIdx],
                val:    prices[loIdx],
                maxTotal: Math.max(...volumes),
                totalVolume,
            };
        }
    }
    return bestResult;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------
function chartTop()    { return DELTA_H; }
function chartBottom() { return H - TIME_AXIS_H - cvdH; }
function chartH()      { return chartBottom() - chartTop(); }
function cvdTop()      { return H - TIME_AXIS_H - cvdH; }
function getCanvasY(clientY) { return clientY - C.getBoundingClientRect().top; }
function getCanvasX(clientX) { return clientX - C.getBoundingClientRect().left; }

function setDrawMode(mode) {
    if (drawMode === 'vp' && mode !== 'vp') vpStartIdx = null;
    drawMode  = drawMode === mode ? null : mode;
    vpStartIdx = null;
    editDrag   = null;
    editHover  = null;

    document.getElementById('btn-hline').classList.toggle('active',      drawMode === 'hline');
    document.getElementById('btn-vline').classList.toggle('active',      drawMode === 'vline');
    document.getElementById('btn-delete').classList.toggle('delete-mode',drawMode === 'delete');
    document.getElementById('btn-edit').classList.toggle('edit-mode',    drawMode === 'edit');
    document.getElementById('btn-vp').classList.toggle('vp-mode',        drawMode === 'vp');

    if      (drawMode === 'delete') C.style.cursor = 'pointer';
    else if (drawMode === 'edit')   C.style.cursor = 'default';
    else if (drawMode)              C.style.cursor = 'crosshair';
    else                            C.style.cursor = 'default';

    vpHint.style.display = drawMode === 'vp' ? 'block' : 'none';
    vpHint.textContent   = 'Click start candle';
    hoveredLine = null;
    draw();
}

function clearAll() {
    hLines.length = 0; vLines.length = 0; profiles.length = 0;
    hoveredLine = null; vpStartIdx = null;
    editDrag = null; editHover = null;
    draw();
}

function resize() {
    W = wrap.clientWidth; H = wrap.clientHeight;
    C.width = W; C.height = H;
    cvdH = layers.cvd ? Math.floor(H * CVD_RATIO) : 0;
    draw();
}

function initView() {
    if (!CANDLES.length) return;
    const all = []; CANDLES.forEach(c => all.push(c.h, c.l));
    priceMin    = Math.min(...all); priceMax = Math.max(...all);
    centerPrice = (priceMin + priceMax) / 2;
    scaleY      = 1.0;
    candleW     = Math.max(0.1, Math.min(120, W / CANDLES.length * 0.8));
    offsetX     = 0;
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------
function priceToY(price) {
    const ch = chartH(), vis = (priceMax - priceMin) / scaleY;
    const vMin = centerPrice - vis/2, vMax = centerPrice + vis/2;
    return chartTop() + ch * (1 - (price - vMin) / (vMax - vMin));
}
function yToPrice(cy) {
    const ch = chartH(), vis = (priceMax - priceMin) / scaleY;
    const vMin = centerPrice - vis/2, vMax = centerPrice + vis/2;
    return vMax - (cy - chartTop()) / ch * vis;
}
function candleX(i)    { return PRICE_AXIS_W + i * (candleW + 2) - offsetX; }
function xToCandle(cx) { return Math.round((cx + offsetX - PRICE_AXIS_W) / (candleW + 2)); }

// ---------------------------------------------------------------------------
// Edit mode hit detection
// ---------------------------------------------------------------------------
// Returns { type: 'h'|'v'|'vp_start'|'vp_end', index: N } or null
function getEditTarget(cx, cy) {
    // H-lines — drag up/down
    for (let i = 0; i < hLines.length; i++) {
        if (Math.abs(cy - priceToY(hLines[i])) <= HIT_THRESHOLD)
            return { type: 'h', index: i };
    }
    // V-lines — drag left/right
    for (let i = 0; i < vLines.length; i++) {
        const x = candleX(vLines[i]) + candleW/2;
        if (Math.abs(cx - x) <= HIT_THRESHOLD)
            return { type: 'v', index: i };
    }
    // VP handles — start (left edge) and end (right edge) of each profile
    for (let i = 0; i < profiles.length; i++) {
        const prof   = profiles[i];
        const startX = candleX(prof.startIdx) + candleW/2;
        const endX   = candleX(prof.endIdx)   + candleW/2;
        const midY   = (priceToY(prof.vah) + priceToY(prof.val)) / 2;
        if (Math.hypot(cx - startX, cy - midY) <= HANDLE_R + 4)
            return { type: 'vp_start', index: i };
        if (Math.hypot(cx - endX,   cy - midY) <= HANDLE_R + 4)
            return { type: 'vp_end',   index: i };
    }
    return null;
}

function editCursorFor(target) {
    if (!target) return 'default';
    if (target.type === 'h')        return 'ns-resize';
    if (target.type === 'v')        return 'ew-resize';
    if (target.type === 'vp_start') return 'ew-resize';
    if (target.type === 'vp_end')   return 'ew-resize';
    return 'default';
}

// ---------------------------------------------------------------------------
// Main draw
// ---------------------------------------------------------------------------
function draw() {
    if (!W || !H) return;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#111'; ctx.fillRect(0, 0, W, H);

    drawPriceAxis();
    drawTimeAxis();
    drawDeltaPanel();
    drawCandles();
    drawProfiles();
    if (layers.vwap)       drawVwap();
    if (layers.bigTrades)  drawBigTrades();
    drawLines();
    if (drawMode === 'edit') drawEditHandles();
    drawHlinePreview();
    drawVlinePreview();
    if (layers.cvd)        drawCvdPanel();
    if (layers.absorption) drawAbsorption();
    if (drawMode === 'vp' && vpStartIdx !== null) drawVpPreview();
}

// ---------------------------------------------------------------------------
// Price axis
// ---------------------------------------------------------------------------
function drawPriceAxis() {
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, chartTop(), PRICE_AXIS_W, chartH());
    const vis  = (priceMax - priceMin) / scaleY;
    const vMin = centerPrice - vis/2, vMax = centerPrice + vis/2;
    const step = niceStep(vis / 10);
    for (let p = Math.ceil(vMin/step)*step; p <= vMax; p += step) {
        const y = priceToY(p);
        if (y < chartTop() || y > chartBottom()) continue;
        ctx.beginPath(); ctx.moveTo(PRICE_AXIS_W, y); ctx.lineTo(W, y);
        ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5; ctx.stroke();
        ctx.fillStyle = '#888'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
        ctx.fillText(fmt(p), PRICE_AXIS_W - 4, y + 3);
    }
}

// ---------------------------------------------------------------------------
// Time axis
// ---------------------------------------------------------------------------
function drawTimeAxis() {
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, H - TIME_AXIS_H, W, TIME_AXIS_H);
    ctx.fillStyle = '#666'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(60 / candleW));
    CANDLES.forEach((c, i) => {
        if (i % step !== 0) return;
        const x = candleX(i) + candleW / 2;
        if (x < PRICE_AXIS_W || x > W) return;
        ctx.fillText(
            new Date(c.t).toLocaleTimeString('en-US', {
                hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'America/New_York'
            }),
            x, H - 5
        );
    });
}

// ---------------------------------------------------------------------------
// Delta panel
// ---------------------------------------------------------------------------
function drawDeltaPanel() {
    ctx.fillStyle = '#161616';
    ctx.fillRect(PRICE_AXIS_W, 0, W - PRICE_AXIS_W, DELTA_H);
    ctx.beginPath(); ctx.moveTo(PRICE_AXIS_W, DELTA_H); ctx.lineTo(W, DELTA_H);
    ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.stroke();
    CANDLES.forEach((c, i) => {
        const x = candleX(i);
        if (x + candleW < PRICE_AXIS_W || x > W) return;
        ctx.fillStyle = c.delta >= 0 ? '#4a9' : '#c55';
        ctx.font = `${Math.max(8, Math.min(11, candleW / 7))}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText((c.delta >= 0 ? '+' : '') + c.delta, x + candleW / 2, DELTA_H - 8);
    });
}

// ---------------------------------------------------------------------------
// Candles
// ---------------------------------------------------------------------------
function drawCandles() {
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    CANDLES.forEach((c, i) => {
        const x = candleX(i);
        if (x + candleW < PRICE_AXIS_W || x > W) return;
        const oy = priceToY(c.o), hy = priceToY(c.h), ly = priceToY(c.l), cy = priceToY(c.c);
        const bull = c.c >= c.o, midX = x + candleW / 2;
        ctx.strokeStyle = bull ? '#3a9a5a' : '#aa4040'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(midX, hy); ctx.lineTo(midX, ly); ctx.stroke();
        ctx.fillStyle = bull ? '#2a7a4a' : '#8a3030';
        ctx.fillRect(x + candleW*0.2, Math.min(oy, cy), candleW*0.6, Math.max(1, Math.abs(cy-oy)));
        if (layers.footprint) drawFootprint(c, x);
        if (layers.passive)   drawPassive(c, x);
    });
    ctx.restore();
}

// ---------------------------------------------------------------------------
// Footprint
// ---------------------------------------------------------------------------
function drawFootprint(c, x) {
    const keys = Object.keys(c.tick_volume); if (!keys.length) return;
    const cellH = Math.abs(priceToY(0) - priceToY(tickSize)); if (cellH < 4) return;
    ctx.font = `${Math.max(6, Math.min(10, cellH * 0.7))}px monospace`;
    const colW = Math.max(25, candleW * 0.45), rightX = x + candleW * 0.5;
    ctx.save(); ctx.beginPath(); ctx.rect(x, chartTop(), candleW, chartH()); ctx.clip();
    keys.forEach(k => {
        const val = c.tick_volume[k]; if (!Array.isArray(val) || val.length < 2) return;
        const [buy, sell] = val, y = priceToY(parseFloat(k)) - cellH / 2;
        ctx.fillStyle = '#1a2a1a'; ctx.fillRect(rightX,        y, colW/2-1, cellH);
        ctx.fillStyle = '#1a1a2a'; ctx.fillRect(rightX+colW/2, y, colW/2-1, cellH);
        ctx.fillStyle = '#5c5'; ctx.textAlign = 'center';
        ctx.fillText(buy,  rightX + colW/4,   y + cellH*0.65);
        ctx.fillStyle = '#c55';
        ctx.fillText(sell, rightX + colW*3/4, y + cellH*0.65);
    });
    ctx.restore();
}

// ---------------------------------------------------------------------------
// Passive orders
// ---------------------------------------------------------------------------
function drawPassive(c, x) {
    const keys = Object.keys(c.passive_orders); if (!keys.length) return;
    const cellH = Math.abs(priceToY(0) - priceToY(tickSize)); if (cellH < 4) return;
    ctx.font = `${Math.max(6, Math.min(10, cellH * 0.7))}px monospace`;
    const colW   = Math.max(25, candleW * 0.45);
    const thresh = parseInt(document.getElementById('passive-thresh').value) || 20;
    ctx.save(); ctx.beginPath(); ctx.rect(x, chartTop(), candleW, chartH()); ctx.clip();
    keys.forEach(k => {
        const val = c.passive_orders[k]; if (!Array.isArray(val) || val.length < 2) return;
        const [size, count] = val, p = parseFloat(k), y = priceToY(p) - cellH/2, above = p > c.o;
        ctx.fillStyle = above ? '#2a1a1a' : '#1a2a1a';
        ctx.fillRect(x, y, colW, cellH);
        ctx.fillStyle = above ? '#c88' : '#8c8'; ctx.textAlign = 'center';
        ctx.fillText(`${size}(${count})`, x + colW/2, y + cellH*0.65);
        if (size >= thresh) {
            ctx.strokeStyle = '#f33'; ctx.lineWidth = 1.5;
            ctx.strokeRect(x, y, colW, Math.max(1, cellH));
        }
    });
    ctx.restore();
}

// ---------------------------------------------------------------------------
// VWAP
// ---------------------------------------------------------------------------
function drawVwap() {
    if (!INDICATORS.length) return;
    const key   = getVwapKey();
    const group = VWAP_COLS[key];
    const style = VWAP_STYLE[key];
    if (!group) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    group.bands.forEach(([n, upCol, dnCol]) => {
        if (!stdActive[n]) return;
        ctx.strokeStyle = style.band[n-1]; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
        ctx.beginPath();
        let started = false;
        INDICATORS.forEach((ind, i) => {
            const v = ind[upCol]; if (v == null) { started = false; return; }
            const x = candleX(i) + candleW/2, y = priceToY(v);
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.beginPath(); started = false;
        INDICATORS.forEach((ind, i) => {
            const v = ind[dnCol]; if (v == null) { started = false; return; }
            const x = candleX(i) + candleW/2, y = priceToY(v);
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });
    ctx.strokeStyle = style.line; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    INDICATORS.forEach((ind, i) => {
        const v = ind[group.vwap]; if (v == null) { started = false; return; }
        const x = candleX(i) + candleW/2, y = priceToY(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
}

// ---------------------------------------------------------------------------
// Big trades
// ---------------------------------------------------------------------------
function drawBigTrades() {
    if (!BIG_TRADES.length) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    CANDLES.forEach((c, i) => {
        const trades = tradesByCandleIdx[i]; if (!trades) return;
        const x = candleX(i); if (x + candleW < PRICE_AXIS_W || x > W) return;
        trades.forEach(trade => {
            const cx = x + candleW/2, cy = priceToY(trade.p);
            if (cy < chartTop() || cy > chartBottom()) return;
            const r = Math.max(3, Math.min(18, (trade.s / maxTradeSize) * 18));
            const isBuy = trade.d === 'B';
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2);
            ctx.fillStyle   = isBuy ? 'rgba(60,120,255,0.7)' : 'rgba(160,60,255,0.7)';
            ctx.fill();
            ctx.strokeStyle = isBuy ? '#6af' : '#c6f'; ctx.lineWidth = 1; ctx.stroke();
        });
    });
    ctx.restore();
}

// ---------------------------------------------------------------------------
// CVD panel
// ---------------------------------------------------------------------------
function drawCvdPanel() {
    if (!cvdH || !INDICATORS.length) return;
    const top = cvdTop();
    ctx.fillStyle = '#161616'; ctx.fillRect(0, top, W, cvdH);
    ctx.beginPath(); ctx.moveTo(0, top); ctx.lineTo(W, top);
    ctx.strokeStyle = '#444'; ctx.lineWidth = 0.5; ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
    ctx.fillText('CVD', PRICE_AXIS_W + 4, top + 10);
    const vals = INDICATORS.map(ind => ind.cumulative_delta).filter(v => v != null);
    if (!vals.length) return;
    const cvdMin = Math.min(...vals), cvdMax = Math.max(...vals);
    const cvdRange = cvdMax - cvdMin || 1;
    const pad = 4;
    function cvdToY(v) { return top + pad + (cvdH - pad*2) * (1 - (v - cvdMin) / cvdRange); }
    const zeroY = cvdToY(0);
    if (zeroY >= top && zeroY <= top + cvdH) {
        ctx.beginPath(); ctx.moveTo(PRICE_AXIS_W, zeroY); ctx.lineTo(W, zeroY);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.stroke();
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, top, W - PRICE_AXIS_W, cvdH); ctx.clip();
    ctx.beginPath();
    let started = false;
    INDICATORS.forEach((ind, i) => {
        if (ind.cumulative_delta == null) return;
        const x = candleX(i) + candleW/2, y = cvdToY(ind.cumulative_delta);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#5af'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#555'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText(cvdMax.toFixed(0), PRICE_AXIS_W - 2, top + 10);
    ctx.fillText(cvdMin.toFixed(0), PRICE_AXIS_W - 2, top + cvdH - 2);
}

// ---------------------------------------------------------------------------
// Absorption
// ---------------------------------------------------------------------------
function drawAbsorption() {
    if (!INDICATORS.length) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    CANDLES.forEach((c, i) => {
        const ind = INDICATORS[i];
        if (!ind || ind.absorption_score == null) return;
        const score    = ind.absorption_score;
        const absThresh = parseFloat(document.getElementById('absorption-thresh').value) || 2.0;
        if (score < absThresh) return;
        const x    = candleX(i);
        const barH = Math.abs(priceToY(c.l) - priceToY(c.h));
        const y    = priceToY(c.h);
        const color = c.delta < 0
            ? `rgba(160,60,255,${Math.min(0.15 + (score-2)*0.1, 0.5)})`
            : `rgba(255,140,0,${Math.min(0.15 + (score-2)*0.1, 0.5)})`;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, candleW, Math.max(1, barH));
    });
    ctx.restore();
}

// ---------------------------------------------------------------------------
// Volume profiles
// ---------------------------------------------------------------------------
function drawProfiles() { profiles.forEach(p => drawOneProfile(p)); }

function drawOneProfile(prof) {
    const { startIdx, endIdx, levels, poc, vah, val, maxTotal } = prof;
    const startX  = candleX(startIdx), endX = candleX(endIdx) + candleW;
    const rangeW  = endX - startX;
    const volMaxW  = parseInt(document.getElementById('vp-vol-width').value)   || 80;
    const deltaMaxW= parseInt(document.getElementById('vp-delta-width').value) || 80;
    const barMaxW  = Math.min(volMaxW, rangeW * 0.4);
    if (barMaxW < 4) return;

    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    const cellH = Math.abs(priceToY(0) - priceToY(tickSize));

    Object.values(levels).forEach(lv => {
        const y    = priceToY(lv.p) - cellH / 2;
        const inVA = lv.p >= val && lv.p <= vah;
        const isPoc = lv.p === poc;
        ctx.globalAlpha = VP_OPACITY;
        ctx.fillStyle   = isPoc ? '#ff69b4' : inVA ? '#e07020' : '#b89020';
        ctx.fillRect(startX, y, (lv.total / maxTotal) * barMaxW, Math.max(1, cellH));
        const delta = lv.buy - lv.sell;
        ctx.fillStyle = delta >= 0 ? '#3a8a3a' : '#6a2a9a';
        ctx.fillRect(startX - (Math.abs(delta)/maxTotal)*deltaMaxW, y,
                     (Math.abs(delta)/maxTotal)*deltaMaxW, Math.max(1, cellH));
        ctx.globalAlpha = 1;
    });

    const pocY = priceToY(poc);
    ctx.strokeStyle = '#ff69b4'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(startX, pocY); ctx.lineTo(endX, pocY); ctx.stroke();
    const vahY = priceToY(vah), valY = priceToY(val);
    ctx.strokeStyle = '#888'; ctx.lineWidth = 0.5; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(startX, vahY); ctx.lineTo(endX, vahY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(startX, valY); ctx.lineTo(endX, valY); ctx.stroke();
    ctx.setLineDash([]);
    if (barMaxW > 20) {
        ctx.font = '9px monospace'; ctx.textAlign = 'left';
        ctx.fillStyle = '#ff69b4';
        ctx.fillText(`POC ${fmt(poc)}`, startX+2, pocY-2);
        ctx.fillStyle = '#888';
        ctx.fillText(`VAH ${fmt(vah)}`, startX+2, vahY-2);
        ctx.fillText(`VAL ${fmt(val)}`, startX+2, valY-2);
    }
    ctx.restore();
}

function drawVpPreview() {
    if (vpStartIdx === null) return;
    const x = candleX(vpStartIdx);
    ctx.save();
    ctx.strokeStyle = '#44cc44'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x + candleW/2, chartTop()); ctx.lineTo(x + candleW/2, chartBottom());
    ctx.stroke(); ctx.setLineDash([]); ctx.restore();
}

// ---------------------------------------------------------------------------
// Edit handles — drawn on top of everything when in edit mode
// ---------------------------------------------------------------------------
function drawEditHandles() {
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();

    // H-line handles — small square on the line
    hLines.forEach((price, i) => {
        const y   = priceToY(price);
        const hov = editHover && editHover.type === 'h' && editHover.index === i;
        ctx.fillStyle   = hov ? '#ff4' : '#aa0';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1;
        ctx.fillRect(PRICE_AXIS_W + 8, y - 5, 10, 10);
        ctx.strokeRect(PRICE_AXIS_W + 8, y - 5, 10, 10);
    });

    // V-line handles — small square on the line
    vLines.forEach((idx, i) => {
        const x   = candleX(idx) + candleW/2;
        const hov = editHover && editHover.type === 'v' && editHover.index === i;
        ctx.fillStyle   = hov ? '#fa0' : '#a60';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1;
        ctx.fillRect(x - 5, chartTop() + 8, 10, 10);
        ctx.strokeRect(x - 5, chartTop() + 8, 10, 10);
    });

    // VP handles — circles at start/end of each profile
    profiles.forEach((prof, i) => {
        const startX = candleX(prof.startIdx) + candleW/2;
        const endX   = candleX(prof.endIdx)   + candleW/2;
        const midY   = (priceToY(prof.vah) + priceToY(prof.val)) / 2;

        const hovStart = editHover && editHover.type === 'vp_start' && editHover.index === i;
        const hovEnd   = editHover && editHover.type === 'vp_end'   && editHover.index === i;

        // start handle
        ctx.beginPath(); ctx.arc(startX, midY, HANDLE_R, 0, Math.PI*2);
        ctx.fillStyle   = hovStart ? '#4f4' : '#2a8';
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.fill(); ctx.stroke();

        // end handle
        ctx.beginPath(); ctx.arc(endX, midY, HANDLE_R, 0, Math.PI*2);
        ctx.fillStyle   = hovEnd ? '#4f4' : '#2a8';
        ctx.fill(); ctx.stroke();

        // label
        ctx.fillStyle = '#6fa'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
        ctx.fillText('◀', startX, midY + 3);
        ctx.fillText('▶', endX,   midY + 3);
    });

    ctx.restore();
}

// ---------------------------------------------------------------------------
// H/V lines
// ---------------------------------------------------------------------------
function getHoveredLine(cx, cy) {
    for (let i = 0; i < hLines.length; i++)
        if (Math.abs(cy - priceToY(hLines[i])) <= HIT_THRESHOLD) return { type:'h', index:i };
    for (let i = 0; i < vLines.length; i++)
        if (Math.abs(cx - (candleX(vLines[i]) + candleW/2)) <= HIT_THRESHOLD) return { type:'v', index:i };
    return null;
}

function getHoveredProfile(cx, cy) {
    for (let i = 0; i < profiles.length; i++) {
        const prof   = profiles[i];
        const startX = candleX(prof.startIdx), endX = candleX(prof.endIdx) + candleW;
        if (cx < startX || cx > endX) continue;
        const pocY = priceToY(prof.poc), vahY = priceToY(prof.vah), valY = priceToY(prof.val);
        if (Math.abs(cy-pocY) <= HIT_THRESHOLD) return i;
        if (Math.abs(cy-vahY) <= HIT_THRESHOLD) return i;
        if (Math.abs(cy-valY) <= HIT_THRESHOLD) return i;
        if (cy >= Math.min(vahY,valY) && cy <= Math.max(vahY,valY)) return i;
    }
    return -1;
}

function drawHlinePreview() {
    if (drawMode !== 'hline') return;
    const rawPrice = yToPrice(_mouseY);
    const snapped  = parseFloat((Math.round(rawPrice / tickSize) * tickSize).toPrecision(10));
    const y = priceToY(snapped);
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    ctx.strokeStyle = 'rgba(255,255,0,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(PRICE_AXIS_W, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff0'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
    ctx.fillText(fmt(snapped), PRICE_AXIS_W + 4, y - 3);
    ctx.restore();
}

function drawVlinePreview() {
    if (drawMode !== 'vline') return;
    const idx = Math.floor((_mouseX + offsetX - PRICE_AXIS_W) / (candleW + 2));
    if (idx < 0 || idx >= CANDLES.length) return;
    const x = candleX(idx) + candleW / 2;
    const t = new Date(CANDLES[idx].t);
    const lbl = t.toLocaleString('en-US', {
        month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
        hour12: false, timeZone: 'America/New_York'
    });
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    ctx.strokeStyle = 'rgba(255,136,0,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(x, chartTop()); ctx.lineTo(x, chartBottom()); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f80'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText(lbl, x, chartTop() + 14);
    ctx.restore();
}

function drawLines() {
    ctx.save();
    ctx.beginPath(); ctx.rect(PRICE_AXIS_W, chartTop(), W - PRICE_AXIS_W, chartH()); ctx.clip();
    ctx.setLineDash([4,4]);
    hLines.forEach((price, i) => {
        const y   = priceToY(price);
        const hov = hoveredLine && hoveredLine.type==='h' && hoveredLine.index===i;
        ctx.strokeStyle = hov ? '#f44' : '#ff0'; ctx.lineWidth = hov ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(PRICE_AXIS_W, y); ctx.lineTo(W, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = hov ? '#f44' : '#ff0'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
        ctx.fillText(fmt(price), PRICE_AXIS_W + 4, y - 3);
        ctx.setLineDash([4,4]); ctx.lineWidth = 1;
    });
    vLines.forEach((idx, i) => {
        const x   = candleX(idx) + candleW/2;
        const hov = hoveredLine && hoveredLine.type==='v' && hoveredLine.index===i;
        ctx.strokeStyle = hov ? '#f44' : '#f80'; ctx.lineWidth = hov ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(x, chartTop()); ctx.lineTo(x, chartBottom()); ctx.stroke();
        ctx.lineWidth = 1;
        // always show time label
        if (idx >= 0 && idx < CANDLES.length) {
            const t = new Date(CANDLES[idx].t);
            const lbl = t.toLocaleString('en-US', {
                month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit',
                hour12: false, timeZone: 'America/New_York'
            });
            ctx.setLineDash([]);
            ctx.fillStyle = hov ? '#f44' : '#f80';
            ctx.font = '10px monospace'; ctx.textAlign = 'center';
            ctx.fillText(lbl, x, chartTop() + 14);
        }
    });
    ctx.setLineDash([]); ctx.restore();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function niceStep(rough) {
    const mag = Math.pow(10, Math.floor(Math.log10(rough))), norm = rough / mag;
    if (norm < 1.5) return mag; if (norm < 3) return 2*mag; if (norm < 7) return 5*mag; return 10*mag;
}
function resetView() { initView(); draw(); }

// ---------------------------------------------------------------------------
// Mouse
// ---------------------------------------------------------------------------
let isDragging=false, dragStartX=0, dragOffsetX=0, dragStartY=0, dragCenterPrice=0;

C.addEventListener('mousedown', e => {
    const cx = getCanvasX(e.clientX), cy = getCanvasY(e.clientY);

    // Edit mode: start dragging a handle
    if (drawMode === 'edit') {
        const target = getEditTarget(cx, cy);
        if (target) {
            editDrag = target;
            C.style.cursor = editCursorFor(target);
        }
        return;
    }

    if (drawMode) return;
    isDragging = true; dragStartX = e.clientX; dragOffsetX = offsetX;
    dragStartY = cy; dragCenterPrice = centerPrice;
});

C.addEventListener('click', e => {
    if (drawMode === 'edit') return; // handled by mousedown/mouseup
    if (!drawMode) return;
    const cx = getCanvasX(e.clientX), cy = getCanvasY(e.clientY);

    if (drawMode === 'vp') {
        const idx = Math.max(0, Math.min(CANDLES.length-1, xToCandle(cx)));
        if (vpStartIdx === null) {
            vpStartIdx = idx; vpHint.textContent = 'Click end candle'; draw();
        } else {
            const prof = computeProfile(vpStartIdx, idx);
            if (prof) profiles.push(prof);
            vpStartIdx = null; vpHint.textContent = 'Click start candle';
            setDrawMode(null); draw();
        }
        return;
    }

    if (drawMode === 'hline') {
        const rawPrice = yToPrice(cy);
        const snapped  = parseFloat((Math.round(rawPrice / tickSize) * tickSize).toPrecision(10));
        hLines.push(snapped);
        setDrawMode(null);
    } else if (drawMode === 'vline') {
        const idx = Math.floor((cx + offsetX - PRICE_AXIS_W) / (candleW + 2));
        if (idx >= 0 && idx < CANDLES.length) vLines.push(idx);
        setDrawMode(null);
    } else if (drawMode === 'delete') {
        const hit = getHoveredLine(cx, cy);
        if (hit) {
            if (hit.type === 'h') hLines.splice(hit.index, 1); else vLines.splice(hit.index, 1);
            hoveredLine = null;
        } else {
            const profIdx = getHoveredProfile(cx, cy);
            if (profIdx >= 0) profiles.splice(profIdx, 1);
        }
    }
    draw();
});

C.addEventListener('mousemove', e => {
    const cx = getCanvasX(e.clientX), cy = getCanvasY(e.clientY);
    _mouseX = cx; _mouseY = cy;

    // Edit mode: drag or hover
    if (drawMode === 'edit') {
        if (editDrag) {
            // Live update for h/v lines
            if (editDrag.type === 'h') {
                const rawPrice = yToPrice(cy);
                const snapped  = parseFloat((Math.round(rawPrice / tickSize) * tickSize).toPrecision(10));
                hLines[editDrag.index] = snapped;
                draw();
            } else if (editDrag.type === 'v') {
                const idx = Math.max(0, Math.min(CANDLES.length-1, xToCandle(cx)));
                vLines[editDrag.index] = idx;
                draw();
            } else if (editDrag.type === 'vp_start' || editDrag.type === 'vp_end') {
                draw();
                const newIdx = Math.max(0, Math.min(CANDLES.length-1, xToCandle(cx)));
                const x = candleX(newIdx) + candleW/2;
                ctx.save();
                ctx.strokeStyle = 'rgba(68,204,68,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
                ctx.beginPath(); ctx.moveTo(x, chartTop()); ctx.lineTo(x, chartBottom()); ctx.stroke();
                ctx.setLineDash([]); ctx.restore();
                showTooltip(e);
            }
        } else {
            // Hover detection for cursor change
            const target = getEditTarget(cx, cy);
            if (JSON.stringify(target) !== JSON.stringify(editHover)) {
                editHover = target;
                C.style.cursor = editCursorFor(target);
                draw();
            }
        }
        return;
    }

    if (isDragging) {
        offsetX = dragOffsetX - (e.clientX - dragStartX);
        const dy = cy - dragStartY;
        centerPrice = dragCenterPrice + dy * (priceMax - priceMin) / scaleY / chartH();
        draw(); return;
    }
    if (drawMode === 'delete') {
        const hit = getHoveredLine(cx, cy);
        if (hit !== hoveredLine) { hoveredLine = hit; C.style.cursor = hit ? 'pointer' : 'crosshair'; draw(); }
        return;
    }
    if (drawMode === 'hline' || drawMode === 'vline') { draw(); return; }
    showTooltip(e);
});

C.addEventListener('mouseup', e => {
    // Edit mode: commit the drag
    if (drawMode === 'edit' && editDrag) {
        if (editDrag.type === 'vp_start' || editDrag.type === 'vp_end') {
            const cx     = getCanvasX(e.clientX);
            const newIdx = Math.max(0, Math.min(CANDLES.length-1, xToCandle(cx)));
            const prof   = profiles[editDrag.index];
            const otherIdx = editDrag.type === 'vp_start' ? prof.endIdx : prof.startIdx;
            const newProf  = computeProfile(newIdx, otherIdx);
            if (newProf) profiles[editDrag.index] = newProf;
            draw();
        }
        editDrag = null;
        return;
    }
    isDragging = false;
});

C.addEventListener('mouseleave', () => {
    isDragging = false;
    editDrag   = null;
    tooltip.style.display = 'none';
    if (hoveredLine) { hoveredLine = null; draw(); }
});

C.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    if (e.ctrlKey) {
        const idx = Math.floor((getCanvasX(e.clientX) + offsetX - PRICE_AXIS_W) / (candleW + 2));
        const prevX = candleX(idx);
        candleW = Math.max(0.1, Math.min(300, candleW * factor));
        offsetX += candleX(idx) - prevX;
    } else if (e.shiftKey) {
        const mp = yToPrice(getCanvasY(e.clientY));
        scaleY = Math.max(0.1, Math.min(50, scaleY * factor));
        centerPrice += (getCanvasY(e.clientY) - priceToY(mp)) * (priceMax - priceMin) / scaleY / chartH();
    } else {
        offsetX += e.deltaY > 0 ? -60 : 60;
    }
    draw();
}, { passive: false });

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------
function showTooltip(e) {
    const isVpEdit = drawMode === 'edit' && editDrag &&
                     (editDrag.type === 'vp_start' || editDrag.type === 'vp_end');
    if (drawMode && drawMode !== 'vp' && drawMode !== 'hline' && drawMode !== 'vline' && !isVpEdit) {
        tooltip.style.display = 'none'; return;
    }
    const idx = Math.floor((getCanvasX(e.clientX) + offsetX - PRICE_AXIS_W) / (candleW + 2));
    if (idx < 0 || idx >= CANDLES.length) { tooltip.style.display = 'none'; return; }
    const c  = CANDLES[idx];
    const t  = new Date(c.t);
    const ts = t.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit', hour12:false, timeZone:'America/New_York' });
    const ds = t.toLocaleDateString('en-US', { timeZone: 'America/New_York' });
    const sign = c.delta >= 0 ? '+' : '';

    let indHtml = '';
    if (INDICATORS.length && idx < INDICATORS.length) {
        const ind = INDICATORS[idx];
        if (layers.vwap) {
            const key   = getVwapKey();
            const group = VWAP_COLS[key];
            const style = VWAP_STYLE[key];
            const vwapVal = ind[group.vwap];
            if (vwapVal != null) {
                const vwapSnapped = parseFloat((Math.round(vwapVal / tickSize) * tickSize).toPrecision(10));
                indHtml += `<div style="color:${style.line}">VWAP (${key}): <b>${fmt(vwapSnapped)}</b></div>`;
                group.bands.forEach(([n, upCol, dnCol]) => {
                    if (!stdActive[n]) return;
                    const up = ind[upCol], dn = ind[dnCol];
                    if (up != null && dn != null) {
                        const upSnapped = parseFloat((Math.round(up / tickSize) * tickSize).toPrecision(10));
                        const dnSnapped = parseFloat((Math.round(dn / tickSize) * tickSize).toPrecision(10));
                        indHtml += `<div style="color:${style.band[n-1]}">${n}σ: ${fmt(dnSnapped)} – ${fmt(upSnapped)}</div>`;
                    }
                });
            }
        }
        if (layers.cvd) {
            const cvdVal = ind['cumulative_delta'];
            if (cvdVal != null)
                indHtml += `<div style="color:#5af">CVD: <b>${cvdVal.toFixed(0)}</b></div>`;
        }
        if (layers.absorption) {
            const absScore = ind['absorption_score'];
            if (absScore != null)
                indHtml += `<div style="color:${c.delta < 0 ? '#a06aff' : '#f80'}">Absorption: <b>${absScore.toFixed(2)}</b></div>`;
        }
    }

    let btHtml = '';
    const trades = tradesByCandleIdx[idx];
    if (trades && layers.bigTrades) trades.forEach(tr => {
        const col = tr.d === 'B' ? '#6af' : '#c6f';
        btHtml += `<div style="color:${col}">${tr.d==='B'?'▲':'▼'} ${tr.s} @ ${fmt(tr.p)}</div>`;
    });

    tooltip.innerHTML = `
        <div style="color:#aaa;margin-bottom:4px">${ds} ${ts}</div>
        <div>O: <b>${fmt(c.o)}</b>  H: <b>${fmt(c.h)}</b></div>
        <div>L: <b>${fmt(c.l)}</b>  C: <b>${fmt(c.c)}</b></div>
        <div>Vol: <b>${c.vol.toLocaleString()}</b></div>
        <div style="color:#5c5">Buy: <b>${c.buy_vol.toLocaleString()}</b></div>
        <div style="color:#c55">Sell: <b>${c.sell_vol.toLocaleString()}</b></div>
        <div style="color:${c.delta>=0?'#5c5':'#c55'}">
            Delta: <b>${sign}${c.delta.toLocaleString()}</b> (${c.delta_pct.toFixed(1)}%)
        </div>
        ${indHtml}${btHtml}`;
    tooltip.style.display = 'block';
    tooltip.style.left    = Math.min(getCanvasX(e.clientX) + 12, W - 185) + 'px';
    tooltip.style.top     = Math.max(getCanvasY(e.clientY) - 10, 0) + 'px';
}

window.addEventListener('resize', resize);
initView();
resize();
</script>
"""


def render():
    if st.button("← Back"):
        go_page("home")

    st.title("Footprint Chart")
    st.write("")

    structure = get_parquet_structure()
    if not structure:
        st.error("No datasets found in data/parquet")
        return

    st.subheader("Candles")
    col1, col2, col3 = st.columns(3)
    with col1:
        candles_path, _, asset, _ = resolve_folder_path(structure, key_prefix="fp_candles")
    if candles_path is None:
        return

    available_dates = sorted([
        pd.Timestamp(f.stem) for f in candles_path.glob("*.parquet")
        if f.stem[0].isdigit()
    ])
    if not available_dates:
        st.error("No parquet files found in selected dataset")
        return

    with col2:
        end_date = st.date_input(
            "End date",
            value=available_dates[-1].date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            key=f"fp_end_{candles_path}",
        )
    with col3:
        n_days = st.slider("Days to show", min_value=1, max_value=5, value=1)

    st.write("")
    with st.expander("Big trades dataset (optional)"):
        use_big_trades = st.checkbox("Enable big trades", key="fp_bt_enable")
        bt_path = None
        if use_big_trades:
            col_bt1, _, _ = st.columns(3)
            with col_bt1:
                bt_path, _, _, _ = resolve_folder_path(structure, key_prefix="fp_bt")

    with st.expander("Indicators dataset (optional)"):
        use_indicators = st.checkbox("Enable indicators", key="fp_ind_enable")
        ind_path = None
        if use_indicators:
            col_ind1, _, _ = st.columns(3)
            with col_ind1:
                ind_path, _, _, _ = resolve_folder_path(structure, key_prefix="fp_ind")

    st.write("")

    with st.spinner("Loading data..."):
        df              = load_days(candles_path, pd.Timestamp(end_date), n_days)
        big_trades_data = load_big_trades(bt_path, pd.Timestamp(end_date), n_days) if bt_path else []
        indicators_data = load_indicators(ind_path, pd.Timestamp(end_date), n_days) if ind_path else []

    if df.empty:
        st.warning("No data found for selected range.")
        return

    candles_data = df_to_chart_data(df)

    html = CHART_HTML
    html = html.replace("__CANDLES__",             json.dumps(candles_data))
    html = html.replace("__TICK_SIZE__",            str(TICK_SIZES.get(asset.upper(), 0.25)))
    html = html.replace("__BIG_TRADES__",           json.dumps(big_trades_data))
    html = html.replace("__INDICATORS__",           json.dumps(indicators_data))
    html = html.replace("__CVD_RATIO__",            str(CVD_PANEL_RATIO))
    html = html.replace("__BIG_TRADES_DISABLED__",  "" if bt_path else "disabled")
    html = html.replace("__INDICATORS_DISABLED__",  "" if ind_path else "disabled")

    st.iframe(html, height=1200)