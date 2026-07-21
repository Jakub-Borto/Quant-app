"""
Download DBN v3 ES options DEFINITION files, one .dbn.zst per day.

Range   : 2025-06-28 -> 2026-07-19 (inclusive)
Output  : D:/market_data/raw_dbn/Options_on_futures/ES/ES_2025_06_28-2026-07-19_DEF_V3
Naming  : glbx-mdp3-YYYYMMDD.definition.dbn.zst

Notes
-----
* Uses the hist-preview gateway (required for DBN v3 definitions with leg data).
* timeseries.get_range tolerates roots that don't resolve on a given day, so all
  27 parent roots are always requested; days where a root isn't listed simply
  return nothing for it.
* Files are written to a .part file first and renamed only after the record
  count is confirmed > 0, so weekend/holiday files never land on disk and a
  killed run can be safely resumed.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import databento as db

# --------------------------------------------------------------------------
API_KEY = "db-hkBqGX7sEWEgptWEhjV8ma7CHRHFt"

OUT_DIR = Path(
    r"D:/market_data/raw_dbn/Options_on_futures/ES/ES_2025_06_28-2026-07-19_DEF_V3"
)
START = date(2026, 6, 16)
END   = date(2026, 6, 19)          # inclusive

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_days = (END - START).days + 1
    print(f"Range      : {START} -> {END}  ({total_days} calendar days)")
    print(f"Symbols    : {len(SYMBOLS)} parent roots")
    print(f"Gateway    : {GATEWAY}")
    print(f"Output dir : {OUT_DIR}")
    print("-" * 80)

    client = db.Historical(key=API_KEY, gateway=GATEWAY)

    t_start = time.time()
    n_done = n_ok = n_skip = n_empty = n_fail = 0
    bytes_tot = 0
    failures: list[tuple[str, str]] = []

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
            client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=SYMBOLS,
                stype_in="parent",
                schema="definition",
                start=day.isoformat(),
                end=(day + timedelta(days=1)).isoformat(),   # exclusive
                path=str(tmp),
            )
            took = time.time() - t0

            # count records: a weekend/holiday file has only a metadata header
            n_rec = sum(1 for _ in db.DBNStore.from_file(tmp))

            if n_rec > 0:
                size = tmp.stat().st_size
                tmp.rename(out)
                n_ok += 1
                bytes_tot += size
                print(f"{prefix}  OK     {n_rec:>6} recs  {size/1e6:6.2f} MB  "
                      f"{took:4.1f}s | tot {bytes_tot/1e6:7.1f} MB | ETA {eta_s}")
            else:
                tmp.unlink()
                n_empty += 1
                print(f"{prefix}  EMPTY  no records (weekend/holiday)   "
                      f"          | ETA {eta_s}")

        except Exception as e:                                  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            n_fail += 1
            failures.append((ymd, str(e)))
            print(f"{prefix}  FAIL   {type(e).__name__}: {str(e)[:70]}")

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
        print("\nFailed days (just re-run this script to retry them):")
        for ymd, err in failures:
            print(f"  {ymd}: {err[:110]}")


if __name__ == "__main__":
    main()