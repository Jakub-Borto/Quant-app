// orderbook_replay_cpp — sequential L3 order-book replay for the 1s MBO heatmap transforms.
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

constexpr double PRICE_SCALE = 1'000'000'000.0;

struct Order { int8_t side; int64_t px; int64_t sz; };

using Book = std::unordered_map<int64_t, int64_t>;  // price_i -> total resting qty
using Loc = std::unordered_map<int64_t, Order>;     // oid -> (side, price_i, size)
using Levels = std::vector<std::pair<int64_t, int64_t>>;
using Replay = std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<double>,
                          std::vector<std::string>, std::vector<std::string>>;

// ── event application ──────────────────────────────────────────────────────
static inline void apply(Loc& loc, Book& bid, Book& ask,
                         int8_t acode, int8_t scode, int64_t px, int64_t sz, int64_t oid) {
    switch (acode) {
        case 0: {  // Add
            Book& m = (scode == 0) ? bid : ask;
            m[px] += sz;
            loc[oid] = Order{scode, px, sz};
            break;
        }
        case 1:
        case 3: {  // Cancel / Fill — reduce by sz at the order's stored location
            auto it = loc.find(oid);
            if (it != loc.end()) {
                const Order o = it->second;
                Book& m = (o.side == 0) ? bid : ask;
                auto e = m.find(o.px);
                if (e != m.end()) {
                    e->second -= sz;
                    if (e->second <= 0) m.erase(e);
                }
                const int64_t left = o.sz - sz;
                if (left <= 0) loc.erase(it);
                else it->second.sz = left;
            }
            break;
        }
        case 2: {  // Modify — remove old, insert new
            auto it = loc.find(oid);
            if (it != loc.end()) {
                const Order o = it->second;
                Book& m = (o.side == 0) ? bid : ask;
                auto e = m.find(o.px);
                if (e != m.end()) {
                    e->second -= o.sz;
                    if (e->second <= 0) m.erase(e);
                }
            }
            Book& m = (scode == 0) ? bid : ask;
            m[px] += sz;
            loc[oid] = Order{scode, px, sz};
            break;
        }
        case 4:  // session reset
            bid.clear();
            ask.clear();
            loc.clear();
            break;
        default:  // T / no-op / side=N: book unchanged
            break;
    }
}

// ── helpers ────────────────────────────────────────────────────────────────
static inline std::optional<int64_t> best_bid_i(const Book& bid) {
    std::optional<int64_t> best;
    for (const auto& [px, qty] : bid)
        if (!best || px > *best) best = px;
    return best;
}

static inline std::optional<int64_t> best_ask_i(const Book& ask) {
    std::optional<int64_t> best;
    for (const auto& [px, qty] : ask)
        if (!best || px < *best) best = px;
    return best;
}

// Shortest fixed-notation round-trip formatting — the historical JSON price format;
// changing it would silently diverge from existing parquet output.
static inline void append_price(std::string& s, int64_t px) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf,
                                 static_cast<double>(px) / PRICE_SCALE,
                                 std::chars_format::fixed);
    s.append(buf, r.ptr);
}

static std::string book_json(const Levels& levels) {
    std::string s;
    s.reserve(levels.size() * 16 + 2);
    s.push_back('{');
    bool first = true;
    for (const auto& [px, qty] : levels) {
        if (!first) s.push_back(',');
        first = false;
        s.push_back('"');
        append_price(s, px);
        s += "\":";
        char buf[24];
        const auto r = std::to_chars(buf, buf + sizeof buf, qty);
        s.append(buf, r.ptr);
    }
    s.push_back('}');
    return s;
}

static double median_of_sorted(const std::vector<double>& v) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static std::string full_side_json(const Book& map) {
    Levels levels(map.begin(), map.end());
    std::sort(levels.begin(), levels.end());
    return book_json(levels);
}

static inline double px_to_f(std::optional<int64_t> px) {
    return px ? static_cast<double>(*px) / PRICE_SCALE : std::numeric_limits<double>::quiet_NaN();
}

// ── full-book replay ───────────────────────────────────────────────────────
static Replay run_full(const int8_t* acode, const int8_t* scode, const int64_t* px,
                       const int64_t* sz, const int64_t* oid, const int64_t* sec, size_t n) {
    Loc loc;
    Book bid, ask;
    std::vector<int64_t> o_sec;
    std::vector<double> o_bb, o_ba;
    std::vector<std::string> o_bj, o_aj;

    const auto snap = [&](int64_t cur) {
        o_sec.push_back(cur);
        o_bb.push_back(px_to_f(best_bid_i(bid)));
        o_ba.push_back(px_to_f(best_ask_i(ask)));
        o_bj.push_back(full_side_json(bid));
        o_aj.push_back(full_side_json(ask));
    };

    int64_t cur = 0;
    bool started = false;
    for (size_t i = 0; i < n; ++i) {
        const int64_t s = sec[i];
        if (!started || s != cur) {
            if (started) snap(cur);
            cur = s;
            started = true;
        }
        apply(loc, bid, ask, acode[i], scode[i], px[i], sz[i], oid[i]);
    }
    if (started) snap(cur);

    return {std::move(o_sec), std::move(o_bb), std::move(o_ba), std::move(o_bj), std::move(o_aj)};
}

// ── rolling time-window median of per-second near-book medians ──────────────
struct Roll {
    std::deque<std::pair<int64_t, double>> dq;  // (sec, per-second spatial median)
    std::vector<double> sorted;
    int64_t window;

    explicit Roll(int64_t w) : window(w) {}

    void evict(int64_t cur) {
        while (!dq.empty()) {
            const auto [s, v] = dq.front();
            if (s <= cur - window) {
                dq.pop_front();
                const auto it = std::lower_bound(sorted.begin(), sorted.end(), v);
                if (it != sorted.end() && *it == v) sorted.erase(it);
            } else {
                break;
            }
        }
    }

    std::optional<double> median() const {
        if (sorted.empty()) return std::nullopt;
        return median_of_sorted(sorted);
    }

    void push(int64_t sec, double v) {
        dq.emplace_back(sec, v);
        sorted.insert(std::lower_bound(sorted.begin(), sorted.end(), v), v);
    }
};

// Emit one side cropped to [win_lo, win_hi] plus far levels >= thr.
// Returns (json, optional spatial median of in-window sizes).
static std::pair<std::string, std::optional<double>>
cropped_side(const Book& map, int64_t win_lo, int64_t win_hi, std::optional<double> thr) {
    Levels levels;
    std::vector<double> inwin;
    for (const auto& [px, qty] : map) {
        if (px >= win_lo && px <= win_hi) {
            levels.emplace_back(px, qty);
            inwin.push_back(static_cast<double>(qty));
        } else if (thr && static_cast<double>(qty) >= *thr) {
            levels.emplace_back(px, qty);
        }
    }
    std::sort(levels.begin(), levels.end());
    std::string json = book_json(levels);
    std::optional<double> med;
    if (!inwin.empty()) {
        std::sort(inwin.begin(), inwin.end());
        med = median_of_sorted(inwin);
    }
    return {std::move(json), med};
}

// ── cropped replay ─────────────────────────────────────────────────────────
static Replay run_cropped(const int8_t* acode, const int8_t* scode, const int64_t* px,
                          const int64_t* sz, const int64_t* oid, const int64_t* sec, size_t n,
                          int64_t n_ticks, int64_t tick_i, double mult, int64_t window_sec,
                          const int64_t* trade_sec, const int64_t* trade_lo,
                          const int64_t* trade_hi, size_t n_trades) {
    const int64_t span = n_ticks * tick_i;
    Loc loc;
    Book bid, ask;
    std::vector<int64_t> o_sec;
    std::vector<double> o_bb, o_ba;
    std::vector<std::string> o_bj, o_aj;

    Roll roll_b(window_sec), roll_a(window_sec);
    size_t tp = 0;  // pointer into the sorted trade_sec array

    const auto snap = [&](int64_t cur) {
        // trade high/low for this second (merge-join on sorted trade_sec)
        while (tp < n_trades && trade_sec[tp] < cur) ++tp;
        std::optional<int64_t> lo_i, hi_i;
        if (tp < n_trades && trade_sec[tp] == cur) {
            lo_i = trade_lo[tp];
            hi_i = trade_hi[tp];
        }

        const auto bb = best_bid_i(bid);
        const auto ba = best_ask_i(ask);
        o_sec.push_back(cur);
        o_bb.push_back(px_to_f(bb));
        o_ba.push_back(px_to_f(ba));

        // rolling baseline from PRIOR seconds (evict stale, then read median)
        roll_b.evict(cur);
        roll_a.evict(cur);
        std::optional<double> thr_b = roll_b.median();
        if (thr_b) *thr_b *= mult;
        std::optional<double> thr_a = roll_a.median();
        if (thr_a) *thr_a *= mult;

        const auto lo_anchor = lo_i ? lo_i : (bb ? bb : ba);
        const auto hi_anchor = hi_i ? hi_i : (ba ? ba : bb);
        if (lo_anchor && hi_anchor) {
            const int64_t win_lo = *lo_anchor - span;
            const int64_t win_hi = *hi_anchor + span;
            auto [bj, mb] = cropped_side(bid, win_lo, win_hi, thr_b);
            auto [aj, ma] = cropped_side(ask, win_lo, win_hi, thr_a);
            o_bj.push_back(std::move(bj));
            o_aj.push_back(std::move(aj));
            if (mb) roll_b.push(cur, *mb);
            if (ma) roll_a.push(cur, *ma);
        } else {
            o_bj.emplace_back("{}");
            o_aj.emplace_back("{}");
        }
    };

    int64_t cur = 0;
    bool started = false;
    for (size_t i = 0; i < n; ++i) {
        const int64_t s = sec[i];
        if (!started || s != cur) {
            if (started) snap(cur);
            cur = s;
            started = true;
        }
        apply(loc, bid, ask, acode[i], scode[i], px[i], sz[i], oid[i]);
    }
    if (started) snap(cur);

    return {std::move(o_sec), std::move(o_bb), std::move(o_ba), std::move(o_bj), std::move(o_aj)};
}

// ── pybind11 bindings ──────────────────────────────────────────────────────
template <typename T>
using Arr = py::array_t<T, py::array::c_style>;

template <typename T>
static const T* checked_data(const Arr<T>& a, size_t n, const char* name) {
    if (a.ndim() != 1)
        throw py::value_error(std::string(name) + " must be 1-dimensional");
    if (static_cast<size_t>(a.size()) != n)
        throw py::value_error(std::string(name) + " length mismatch");
    return a.data();
}

static Replay replay_full(Arr<int8_t> acode, Arr<int8_t> scode, Arr<int64_t> price_i,
                          Arr<int64_t> size, Arr<int64_t> oid, Arr<int64_t> sec) {
    if (acode.ndim() != 1) throw py::value_error("acode must be 1-dimensional");
    const size_t n = static_cast<size_t>(acode.size());
    const int8_t* a = acode.data();
    const int8_t* s = checked_data(scode, n, "scode");
    const int64_t* p = checked_data(price_i, n, "price_i");
    const int64_t* z = checked_data(size, n, "size");
    const int64_t* o = checked_data(oid, n, "oid");
    const int64_t* c = checked_data(sec, n, "sec");

    py::gil_scoped_release release;
    return run_full(a, s, p, z, o, c, n);
}

static Replay replay_cropped(Arr<int8_t> acode, Arr<int8_t> scode, Arr<int64_t> price_i,
                             Arr<int64_t> size, Arr<int64_t> oid, Arr<int64_t> sec,
                             int64_t n_ticks, int64_t tick_i, double mult, int64_t window_sec,
                             Arr<int64_t> trade_sec, Arr<int64_t> trade_lo, Arr<int64_t> trade_hi) {
    if (acode.ndim() != 1) throw py::value_error("acode must be 1-dimensional");
    const size_t n = static_cast<size_t>(acode.size());
    const int8_t* a = acode.data();
    const int8_t* s = checked_data(scode, n, "scode");
    const int64_t* p = checked_data(price_i, n, "price_i");
    const int64_t* z = checked_data(size, n, "size");
    const int64_t* o = checked_data(oid, n, "oid");
    const int64_t* c = checked_data(sec, n, "sec");
    if (trade_sec.ndim() != 1) throw py::value_error("trade_sec must be 1-dimensional");
    const size_t nt = static_cast<size_t>(trade_sec.size());
    const int64_t* ts = trade_sec.data();
    const int64_t* tl = checked_data(trade_lo, nt, "trade_lo");
    const int64_t* th = checked_data(trade_hi, nt, "trade_hi");

    py::gil_scoped_release release;
    return run_cropped(a, s, p, z, o, c, n, n_ticks, tick_i, mult, window_sec, ts, tl, th, nt);
}

PYBIND11_MODULE(orderbook_replay_cpp, m) {
    m.doc() = "Sequential L3 order-book replay kernel (C++)";
    m.def("replay_full", &replay_full,
          py::arg("acode"), py::arg("scode"), py::arg("price_i"),
          py::arg("size"), py::arg("oid"), py::arg("sec"));
    m.def("replay_cropped", &replay_cropped,
          py::arg("acode"), py::arg("scode"), py::arg("price_i"),
          py::arg("size"), py::arg("oid"), py::arg("sec"),
          py::arg("n_ticks"), py::arg("tick_i"), py::arg("mult"), py::arg("window_sec"),
          py::arg("trade_sec"), py::arg("trade_lo"), py::arg("trade_hi"));
}
