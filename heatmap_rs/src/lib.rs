// heatmap_rs — sequential L3 order-book replay for the 1s MBO heatmap transforms.
//
// Two entry points, both replay the full event stream maintaining an aggregated
// depth ladder (price_i -> total resting qty) and emit one end-of-second
// snapshot per active second:
//
//   replay_full    — emit the entire book per second as JSON.
//   replay_cropped — emit only levels within ±N ticks of the touch, PLUS far
//                    "big" orders (size >= mult * rolling near-book baseline).
//
// Inputs are pre-encoded numpy arrays (see the Python callers):
//   action code: A=0 C=1 M=2 F=3 R=4 T=5 other=6
//   side code:   B=0 A=1 N=2   (bid = 0, everything else routed to ask)
//   price_i = round(price * 1e9)  (integer key; 0 on NaN/R rows)
//   sec     = ts_recv_ns // 1e9   (UTC epoch second)
//
// Returns (secs, best_bid, best_ask, bid_json, ask_json) as Python lists.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

const PRICE_SCALE: f64 = 1_000_000_000.0;

type Book = FxHashMap<i64, i64>; // price_i -> total resting qty
type Loc = FxHashMap<i64, (i8, i64, i64)>; // oid -> (side_code, price_i, size)

// ── event application ──────────────────────────────────────────────────────
#[inline]
fn apply(loc: &mut Loc, bid: &mut Book, ask: &mut Book,
         acode: i8, scode: i8, px: i64, sz: i64, oid: i64) {
    match acode {
        0 => { // Add
            if scode == 0 { *bid.entry(px).or_insert(0) += sz; }
            else          { *ask.entry(px).or_insert(0) += sz; }
            loc.insert(oid, (scode, px, sz));
        }
        1 | 3 => { // Cancel / Fill — reduce by sz at the order's stored location
            if let Some(&(s, p, old)) = loc.get(&oid) {
                let m = if s == 0 { &mut *bid } else { &mut *ask };
                if let Some(e) = m.get_mut(&p) {
                    *e -= sz;
                    if *e <= 0 { m.remove(&p); }
                }
                let new = old - sz;
                if new <= 0 { loc.remove(&oid); }
                else        { loc.insert(oid, (s, p, new)); }
            }
        }
        2 => { // Modify — remove old, insert new
            if let Some(&(s, p, old)) = loc.get(&oid) {
                let m = if s == 0 { &mut *bid } else { &mut *ask };
                if let Some(e) = m.get_mut(&p) {
                    *e -= old;
                    if *e <= 0 { m.remove(&p); }
                }
            }
            if scode == 0 { *bid.entry(px).or_insert(0) += sz; }
            else          { *ask.entry(px).or_insert(0) += sz; }
            loc.insert(oid, (scode, px, sz));
        }
        4 => { bid.clear(); ask.clear(); loc.clear(); } // session reset
        _ => {} // T / no-op / side=N: book unchanged
    }
}

// ── helpers ────────────────────────────────────────────────────────────────
#[inline]
fn best_bid_i(bid: &Book) -> Option<i64> { bid.keys().copied().max() }
#[inline]
fn best_ask_i(ask: &Book) -> Option<i64> { ask.keys().copied().min() }

#[inline]
fn fmt_price(px: i64) -> String { format!("{}", px as f64 / PRICE_SCALE) }

fn book_json(levels: &[(i64, i64)]) -> String {
    let mut s = String::with_capacity(levels.len() * 16 + 2);
    s.push('{');
    for (k, (px, qty)) in levels.iter().enumerate() {
        if k > 0 { s.push(','); }
        s.push('"');
        s.push_str(&fmt_price(*px));
        s.push_str("\":");
        s.push_str(itoa(*qty).as_str());
    }
    s.push('}');
    s
}

#[inline]
fn itoa(v: i64) -> String { v.to_string() }

fn median_of_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn full_side_json(map: &Book) -> String {
    let mut levels: Vec<(i64, i64)> = map.iter().map(|(&p, &q)| (p, q)).collect();
    levels.sort_unstable_by_key(|&(p, _)| p);
    book_json(&levels)
}

// ── full-book replay ───────────────────────────────────────────────────────
#[allow(clippy::type_complexity)]
fn run_full(acode: &[i8], scode: &[i8], px: &[i64], sz: &[i64], oid: &[i64], sec: &[i64])
    -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<String>, Vec<String>) {
    let n = acode.len();
    let mut loc: Loc = FxHashMap::default();
    let mut bid: Book = FxHashMap::default();
    let mut ask: Book = FxHashMap::default();

    let (mut o_sec, mut o_bb, mut o_ba, mut o_bj, mut o_aj) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    let snap = |bid: &Book, ask: &Book, cur: i64,
                o_sec: &mut Vec<i64>, o_bb: &mut Vec<f64>, o_ba: &mut Vec<f64>,
                o_bj: &mut Vec<String>, o_aj: &mut Vec<String>| {
        o_sec.push(cur);
        o_bb.push(best_bid_i(bid).map_or(f64::NAN, |p| p as f64 / PRICE_SCALE));
        o_ba.push(best_ask_i(ask).map_or(f64::NAN, |p| p as f64 / PRICE_SCALE));
        o_bj.push(full_side_json(bid));
        o_aj.push(full_side_json(ask));
    };

    let mut cur: i64 = 0;
    let mut started = false;
    for i in 0..n {
        let s = sec[i];
        if !started || s != cur {
            if started { snap(&bid, &ask, cur, &mut o_sec, &mut o_bb, &mut o_ba, &mut o_bj, &mut o_aj); }
            cur = s;
            started = true;
        }
        apply(&mut loc, &mut bid, &mut ask, acode[i], scode[i], px[i], sz[i], oid[i]);
    }
    if started { snap(&bid, &ask, cur, &mut o_sec, &mut o_bb, &mut o_ba, &mut o_bj, &mut o_aj); }

    (o_sec, o_bb, o_ba, o_bj, o_aj)
}

// ── rolling time-window median of per-second near-book medians ──────────────
struct Roll {
    dq: VecDeque<(i64, f64)>, // (sec, per-second spatial median)
    sorted: Vec<f64>,
    window: i64,
}
impl Roll {
    fn new(window: i64) -> Self { Roll { dq: VecDeque::new(), sorted: Vec::new(), window } }
    fn evict(&mut self, cur: i64) {
        while let Some(&(s, v)) = self.dq.front() {
            if s <= cur - self.window {
                self.dq.pop_front();
                if let Ok(idx) = self.sorted.binary_search_by(|x| x.partial_cmp(&v).unwrap()) {
                    self.sorted.remove(idx);
                }
            } else { break; }
        }
    }
    fn median(&self) -> Option<f64> {
        if self.sorted.is_empty() { None } else { Some(median_of_sorted(&self.sorted)) }
    }
    fn push(&mut self, sec: i64, v: f64) {
        self.dq.push_back((sec, v));
        let idx = self.sorted.partition_point(|x| *x < v);
        self.sorted.insert(idx, v);
    }
}

// Emit one side cropped to [win_lo, win_hi] plus far levels >= thr.
// Returns (json, Option<spatial_median_of_in_window_sizes>).
fn cropped_side(map: &Book, win_lo: i64, win_hi: i64, thr: Option<f64>) -> (String, Option<f64>) {
    let mut levels: Vec<(i64, i64)> = Vec::new();
    let mut inwin: Vec<f64> = Vec::new();
    for (&px, &qty) in map.iter() {
        if px >= win_lo && px <= win_hi {
            levels.push((px, qty));
            inwin.push(qty as f64);
        } else if let Some(t) = thr {
            if qty as f64 >= t { levels.push((px, qty)); }
        }
    }
    levels.sort_unstable_by_key(|&(p, _)| p);
    let json = book_json(&levels);
    let med = if inwin.is_empty() {
        None
    } else {
        inwin.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(median_of_sorted(&inwin))
    };
    (json, med)
}

// ── cropped replay ─────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_cropped(
    acode: &[i8], scode: &[i8], px: &[i64], sz: &[i64], oid: &[i64], sec: &[i64],
    n_ticks: i64, tick_i: i64, mult: f64, window_sec: i64,
    trade_sec: &[i64], trade_lo: &[i64], trade_hi: &[i64],
) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<String>, Vec<String>) {
    let n = acode.len();
    let span = n_ticks * tick_i;
    let mut loc: Loc = FxHashMap::default();
    let mut bid: Book = FxHashMap::default();
    let mut ask: Book = FxHashMap::default();

    let (mut o_sec, mut o_bb, mut o_ba, mut o_bj, mut o_aj) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    let mut roll_b = Roll::new(window_sec);
    let mut roll_a = Roll::new(window_sec);
    let mut tp = 0usize; // pointer into the sorted trade_sec array

    let mut snap = |bid: &Book, ask: &Book, cur: i64,
                    roll_b: &mut Roll, roll_a: &mut Roll, tp: &mut usize,
                    o_sec: &mut Vec<i64>, o_bb: &mut Vec<f64>, o_ba: &mut Vec<f64>,
                    o_bj: &mut Vec<String>, o_aj: &mut Vec<String>| {
        // trade high/low for this second (merge-join on sorted trade_sec)
        while *tp < trade_sec.len() && trade_sec[*tp] < cur { *tp += 1; }
        let (lo_i, hi_i) = if *tp < trade_sec.len() && trade_sec[*tp] == cur {
            (Some(trade_lo[*tp]), Some(trade_hi[*tp]))
        } else { (None, None) };

        let bb = best_bid_i(bid);
        let ba = best_ask_i(ask);
        o_sec.push(cur);
        o_bb.push(bb.map_or(f64::NAN, |p| p as f64 / PRICE_SCALE));
        o_ba.push(ba.map_or(f64::NAN, |p| p as f64 / PRICE_SCALE));

        // rolling baseline from PRIOR seconds (evict stale, then read median)
        roll_b.evict(cur);
        roll_a.evict(cur);
        let thr_b = roll_b.median().map(|m| m * mult);
        let thr_a = roll_a.median().map(|m| m * mult);

        let lo_anchor = lo_i.or(bb).or(ba);
        let hi_anchor = hi_i.or(ba).or(bb);
        if let (Some(la), Some(ha)) = (lo_anchor, hi_anchor) {
            let win_lo = la - span;
            let win_hi = ha + span;
            let (bj, mb) = cropped_side(bid, win_lo, win_hi, thr_b);
            let (aj, ma) = cropped_side(ask, win_lo, win_hi, thr_a);
            o_bj.push(bj);
            o_aj.push(aj);
            if let Some(m) = mb { roll_b.push(cur, m); }
            if let Some(m) = ma { roll_a.push(cur, m); }
        } else {
            o_bj.push("{}".to_string());
            o_aj.push("{}".to_string());
        }
    };

    let mut cur: i64 = 0;
    let mut started = false;
    for i in 0..n {
        let s = sec[i];
        if !started || s != cur {
            if started {
                snap(&bid, &ask, cur, &mut roll_b, &mut roll_a, &mut tp,
                     &mut o_sec, &mut o_bb, &mut o_ba, &mut o_bj, &mut o_aj);
            }
            cur = s;
            started = true;
        }
        apply(&mut loc, &mut bid, &mut ask, acode[i], scode[i], px[i], sz[i], oid[i]);
    }
    if started {
        snap(&bid, &ask, cur, &mut roll_b, &mut roll_a, &mut tp,
             &mut o_sec, &mut o_bb, &mut o_ba, &mut o_bj, &mut o_aj);
    }

    (o_sec, o_bb, o_ba, o_bj, o_aj)
}

// ── PyO3 bindings ──────────────────────────────────────────────────────────
#[pyfunction]
#[allow(clippy::type_complexity)]
fn replay_full<'py>(
    py: Python<'py>,
    acode: PyReadonlyArray1<'py, i8>,
    scode: PyReadonlyArray1<'py, i8>,
    price_i: PyReadonlyArray1<'py, i64>,
    size: PyReadonlyArray1<'py, i64>,
    oid: PyReadonlyArray1<'py, i64>,
    sec: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<String>, Vec<String>)> {
    let a = acode.as_slice()?;
    let s = scode.as_slice()?;
    let p = price_i.as_slice()?;
    let z = size.as_slice()?;
    let o = oid.as_slice()?;
    let c = sec.as_slice()?;
    Ok(py.allow_threads(|| run_full(a, s, p, z, o, c)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn replay_cropped<'py>(
    py: Python<'py>,
    acode: PyReadonlyArray1<'py, i8>,
    scode: PyReadonlyArray1<'py, i8>,
    price_i: PyReadonlyArray1<'py, i64>,
    size: PyReadonlyArray1<'py, i64>,
    oid: PyReadonlyArray1<'py, i64>,
    sec: PyReadonlyArray1<'py, i64>,
    n_ticks: i64,
    tick_i: i64,
    mult: f64,
    window_sec: i64,
    trade_sec: PyReadonlyArray1<'py, i64>,
    trade_lo: PyReadonlyArray1<'py, i64>,
    trade_hi: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<String>, Vec<String>)> {
    let a = acode.as_slice()?;
    let s = scode.as_slice()?;
    let p = price_i.as_slice()?;
    let z = size.as_slice()?;
    let o = oid.as_slice()?;
    let c = sec.as_slice()?;
    let ts = trade_sec.as_slice()?;
    let tl = trade_lo.as_slice()?;
    let th = trade_hi.as_slice()?;
    Ok(py.allow_threads(|| {
        run_cropped(a, s, p, z, o, c, n_ticks, tick_i, mult, window_sec, ts, tl, th)
    }))
}

#[pymodule]
fn heatmap_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replay_full, m)?)?;
    m.add_function(wrap_pyfunction!(replay_cropped, m)?)?;
    Ok(())
}
