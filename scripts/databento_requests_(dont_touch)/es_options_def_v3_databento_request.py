"""
Download DBN v3 ES options DEFINITION files, one .dbn.zst per day.

Output  : D:/market_data/raw_dbn/Options_on_futures/ES/ES_2025_06_28-2026-07-19_DEF_V3
Naming  : glbx-mdp3-YYYYMMDD.definition.dbn.zst

Notes
-----
* Uses the hist-preview gateway (required for DBN v3 definitions with leg data).
* All 27 parent roots are requested per day. On a resolve-422 (some weekly roots
  are dormant on that date), the offending roots are stripped from the message
  and the request is retried once with the survivors.
* Files are written to a .part file first and renamed only after the record
  count is confirmed > 0, so weekend/holiday files never land on disk and a
  killed run can be safely resumed.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import databento as db

# --------------------------------------------------------------------------
API_KEY = ""

OUT_DIR = Path(
    r"D:/market_data/raw_dbn/Options_on_futures/ES/ES_2025_06_28-2026-07-28_DEF_V3"
)

# START is INCLUSIVE  -> this day IS downloaded.
# END   is INCLUSIVE  -> this day IS downloaded.
# (Each per-day request internally uses an exclusive [day, day+1) window, which
#  is what Databento's get_range expects -- but the loop bounds you set here are
#  BOTH inclusive.)
START = date(2026, 7, 19)          # inclusive
END   = date(2026, 7, 30)          # inclusive

GATEWAY = "hist-preview.databento.com"

ES_OPTION_ROOTS = [
    "E1A", "E1B", "E1C", "E1D",
    "E2A", "E2B", "E2C", "E2D",
    "E3A", "E3B", "E3C", "E3D",
    "E4A", "E4B", "E4C", "E4D",
    "E5A", "E5B", "E5C", "E5D",
    "ES", "EW", "EW1", "EW2", "EW3", "EW4", "EYC",
]
SYMBOLS = [f"{r}.OPT" for r in ES_OPTION_ROOTS]
# --------------------------------------------------------------------------


def _request(client, symbols, day, tmp):
    """One get_range call for a single day into tmp."""
    client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=symbols,
        stype_in="parent",
        schema="definition",
        start=day.isoformat(),
        end=(day + timedelta(days=1)).isoformat(),   # exclusive window (per request)
        path=str(tmp),
    )


def fetch_with_fallback(client, symbols, day, tmp):
    """
    Try all symbols. If the API returns a resolve-422 because some roots are
    dormant that day, drop exactly those roots and retry once with the rest.
    Returns the list of symbols actually requested (for logging).
    """
    try:
        _request(client, symbols, day, tmp)
        return symbols
    except Exception as e:                                       # noqa: BLE001
        msg = str(e)
        if "symbology_invalid_request" in msg and "Could not resolve" in msg:
            # message looks like:
            #   "...Could not resolve smart symbols: E1A.OPT,E1B.OPT,..."
            try:
                tail = msg.split("smart symbols:")[1]
                bad = {
                    tok.strip()
                    for tok in tail.replace("\n", ",").split(",")
                    if tok.strip().endswith(".OPT")
                }
            except IndexError:
                raise
            survivors = [s for s in symbols if s not in bad]
            print(f"    (dropping {len(bad)} dormant root(s), retrying with "
                  f"{len(survivors)})")
            if not survivors:
                raise
            if tmp.exists():
                tmp.unlink()
            _request(client, survivors, day, tmp)
            return survivors
        raise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_days = (END - START).days + 1
    print(f"Range      : {START} -> {END}  ({total_days} calendar days, "
          f"both inclusive)")
    print(f"Symbols    : {len(SYMBOLS)} parent roots")
    print(f"Gateway    : {GATEWAY}")
    print(f"Output dir : {OUT_DIR}")
    print("-" * 80)

    client = db.Historical(key=API_KEY, gateway=GATEWAY)

    t_start = time.time()
    n_done = n_ok = n_skip = n_empty = n_fail = 0
    bytes_tot = 0
    failures: list = []

    day = START
    while day <= END:
        n_done += 1
        ymd = day.strftime("%Y%m%d")
        out = OUT_DIR / f"glbx-mdp3-{ymd}.definition.dbn.zst"
        tmp = out.with_suffix(out.suffix + ".part")

        pct = 100.0 * n_done / total_days
        elapsed = time.time() - t_start
        eta = (elapsed / n_done) * (total_days - n_done) if n_done else 0
        eta_s = f"{int(eta // 60)}m{int(eta % 60):02d}s"
        prefix = f"[{n_done:>3}/{total_days}] {pct:5.1f}% {day} ({day.strftime('%a')})"

        # already downloaded?
        if out.exists() and out.stat().st_size > 0:
            n_skip += 1
            bytes_tot += out.stat().st_size
            print(f"{prefix}  SKIP   already on disk")
            day += timedelta(days=1)
            continue

        # clean up any stale partial
        if tmp.exists():
            tmp.unlink()

        print(f"{prefix}  ...    downloading            (ETA {eta_s})",
              end="\r", flush=True)

        try:
            t0 = time.time()
            used = fetch_with_fallback(client, SYMBOLS, day, tmp)
            took = time.time() - t0

            # count records: a weekend/holiday file has only a metadata header
            n_rec = sum(1 for _ in db.DBNStore.from_file(tmp))

            if n_rec > 0:
                size = tmp.stat().st_size
                tmp.rename(out)
                n_ok += 1
                bytes_tot += size
                note = "" if len(used) == len(SYMBOLS) else f" [{len(used)}/27 roots]"
                print(f"{prefix}  OK     {n_rec:>6} recs  {size/1e6:6.2f} MB  "
                      f"{took:4.1f}s | tot {bytes_tot/1e6:7.1f} MB{note} | ETA {eta_s}")
            else:
                tmp.unlink()
                n_empty += 1
                print(f"{prefix}  EMPTY  no records (weekend/holiday)   "
                      f"          | ETA {eta_s}")

        except Exception as e:                                  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            n_fail += 1
            raw = str(e)
            failures.append((ymd, raw))
            # print the FULL raw error, not a truncated version
            print(f"{prefix}  FAIL   {type(e).__name__}")
            print("        raw error ---------------------------------------------")
            for line in (raw.splitlines() or [raw]):
                print(f"        | {line}")
            print("        -------------------------------------------------------")

        day += timedelta(days=1)
        time.sleep(0.2)

    el = time.time() - t_start
    print("-" * 80)
    print(f"Finished in {int(el // 60)}m{int(el % 60):02d}s")
    print(f"  downloaded : {n_ok}")
    print(f"  skipped    : {n_skip}")
    print(f"  empty      : {n_empty}  (weekends/holidays)")
    print(f"  failed     : {n_fail}")
    print(f"  total size : {bytes_tot/1e6:.1f} MB")

    if failures:
        print("\nFailed days (full raw errors):")
        for ymd, err in failures:
            print(f"\n  === {ymd} ===")
            for line in (err.splitlines() or [err]):
                print(f"  {line}")


if __name__ == "__main__":
    main()