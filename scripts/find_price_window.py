"""
Find days where the 12:46-12:57 (NY, both inclusive) price action sits inside a
given price band.

Scans <data_root>/parquet/Futures/ES/ES_1m_advanced/*.parquet, where the data
root(s) come from the app's Settings (settings.json). Prints two lists:

  CONTAINED  the whole window stayed inside the band (low >= LO, high <= HI)
  TOUCHED    the window merely traded somewhere inside the band

Tweak the constants below for another asset/dataset/window/band.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root
from modules.common.backend.settings import load_settings   # noqa: E402

# ── what to look for ──────────────────────────────────────────────────────────
DATASET   = ("Futures", "ES", "ES_1m_advanced")   # type / asset / dataset
START, END = "12:46", "12:57"                     # NY wall clock, both inclusive
LO, HI    = 6693.0, 6711.0                        # price band


def dataset_dirs() -> list[Path]:
    """The dataset folder under every configured data root that exists."""
    roots = load_settings().data_roots
    dirs = [r / "parquet" / Path(*DATASET) for r in roots]
    return [d for d in dirs if d.is_dir()]


def window_range(path: Path):
    """(low, high) of the window in one daily parquet, or None if no bars."""
    df = pd.read_parquet(path, columns=["high", "low"])
    win = df.between_time(START, END, inclusive="both")
    if win.empty:
        return None
    return float(win["low"].min()), float(win["high"].max())


def main() -> None:
    dirs = dataset_dirs()
    if not dirs:
        print("No such dataset under any configured data root.")
        return

    files = sorted(f for d in dirs for f in d.glob("*.parquet"))
    print(f"Scanning {len(files)} days in {', '.join(str(d) for d in dirs)}")
    print(f"Window {START}-{END} NY, band {LO}-{HI}\n")

    contained, touched = [], []
    for f in files:
        rng = window_range(f)
        if rng is None:
            continue
        lo, hi = rng
        if lo >= LO and hi <= HI:
            contained.append((f.stem, lo, hi))
        elif lo <= HI and hi >= LO:            # overlaps the band
            touched.append((f.stem, lo, hi))

    print(f"CONTAINED ({len(contained)}) — window stayed fully inside the band")
    for day, lo, hi in contained:
        print(f"  {day}   low {lo:8.2f}   high {hi:8.2f}")
    if not contained:
        print("  (none)")

    print(f"\nTOUCHED ({len(touched)}) — window traded into the band at some point")
    for day, lo, hi in touched:
        print(f"  {day}   low {lo:8.2f}   high {hi:8.2f}")
    if not touched:
        print("  (none)")


if __name__ == "__main__":
    main()
