"""
Forex Factory Calendar Parser — Plain Text Version
====================================================
Paste the FF calendar text directly into .txt files.

How to get the text:
    1. Go to forexfactory.com/calendar
    2. Set filter to USD only
    3. Select your date range (FF allows 2 months at once)
    4. Select all text on the page (Ctrl+A), copy (Ctrl+C)
    5. Paste into a .txt file, save into ff_data_scraper/data_usd/
    6. Repeat for each date range
    7. Run: python ff_parser.py

Output:
    ff_data_scraper/data_usd/ff_usd_events.parquet

Schema:
    date    Date    Trading date
    time    Utf8    "HH:MM" (24h) or "All Day"
    event   Utf8    Event name e.g. "Non-Farm Employment Change"
    impact  Utf8    "red" | "grey"
"""

import re
import sys
from pathlib import Path
from datetime import datetime, date

import polars as pl


# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).parent
INPUT_DIR = THIS_DIR / "data_usd"
OUTPUT    = THIS_DIR / "data_usd" / "ff_usd_events.parquet"

# ── Impact keywords that appear in the pasted text ────────────────────────────
# In the plain text copy, FF doesn't include color labels directly.
# We infer impact from a known event list + "Bank Holiday" for grey.
# Red = high impact USD events. Everything else with a time = orange/low (skipped).
# You can extend RED_EVENTS with any event you want to treat as red.

RED_EVENTS = {
    # Employment
    "non-farm employment change", "nonfarm employment change",
    "non-farm payrolls", "adp non-farm employment change",
    "unemployment rate", "unemployment claims",
    "jolt", "jolts",
    # Inflation
    "cpi", "core cpi", "cpi m/m", "cpi y/y", "core cpi m/m",
    "pce", "core pce", "pce price index", "core pce price index",
    "ppi", "core ppi", "ppi m/m",
    # Growth
    "gdp", "advance gdp", "prelim gdp", "final gdp",
    "gdp q/q", "advance gdp q/q", "prelim gdp q/q", "final gdp q/q",
    # Fed
    "federal funds rate", "fomc statement", "fomc meeting minutes",
    "fed announcement",
    # Retail / Consumer
    "retail sales", "core retail sales", "retail sales m/m", "core retail sales m/m",
    "cb consumer confidence", "prelim uom consumer sentiment",
    "uom consumer sentiment",
    # Manufacturing / Services
    "ism manufacturing pmi", "ism services pmi",
    "philly fed manufacturing index",
    # Housing
    "existing home sales", "new home sales", "pending home sales",
    "building permits", "housing starts",
    # Trade
    "trade balance",
    # Durable goods
    "durable goods orders", "core durable goods orders",
    "core durable goods orders m/m", "durable goods orders m/m",
    # Other major
    "tic long-term purchases",
}

GREY_KEYWORDS = {"bank holiday", "holiday"}

# These appear as grey on FF but have zero market impact — skip them
BLOCKLIST = {
    "daylight saving time", "daylight savings time", "dst",
    "clocks change", "clock change",
}


def classify_impact(event_name: str) -> str | None:
    """Return 'red', 'grey', or None (skip)."""
    name_lower = event_name.lower().strip()

    # Skip non-market noise
    for kw in BLOCKLIST:
        if kw in name_lower:
            return None

    # Grey: holidays
    for kw in GREY_KEYWORDS:
        if kw in name_lower:
            return "grey"

    # Red: known high-impact events
    for kw in RED_EVENTS:
        if kw in name_lower:
            return "red"

    return None  # skip orange/yellow/unknown


# ── Date parsing ───────────────────────────────────────────────────────────────

# Matches day-of-week + month + day, optionally year
# e.g. "Fri Jan 1", "Mon Feb 28", "Tue Mar 3, 2026"
DATE_PATTERN = re.compile(
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})(?:,?\s+(\d{4}))?",
    re.IGNORECASE
)

# Time pattern: "8:30" or "14:16" or "All Day"
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")


def _parse_date(day_str: str, month_str: str, year_str: str | None, ref_year: int) -> date:
    year = int(year_str) if year_str else ref_year
    return datetime.strptime(f"{month_str} {day_str} {year}", "%b %d %Y").date()


def _clean_time(raw: str) -> str:
    raw = raw.strip()
    if raw.lower() == "all day" or not raw:
        return "All Day"
    m = re.match(r"(\d{1,2}):(\d{2})(am|pm)?", raw.lower())
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        meridiem = m.group(3)
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mn:02d}"
    return raw


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_text_file(path: Path) -> list[dict]:
    """
    Parse a plain-text FF calendar paste.

    The pasted text looks like:
        Fri
        Jan 1
        All Day
        USD
        Bank Holiday

        Mon
        Jan 4
        10:00
        USD
        ISM Manufacturing PMI
        55.9  54.1  53.6
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]

    # Infer year from filename first, then from text, fallback to current year
    ref_year = _infer_year(path.name, text)

    events = []
    current_date = None
    i = 0

    MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
    DAYS   = {"mon","tue","wed","thu","fri","sat","sun"}

    while i < len(lines):
        line = lines[i]

        # ── Detect date line (day of week) ─────────────────────────────────
        if line.lower()[:3] in DAYS:
            # Next non-empty line should be "Jan 4" or "Jan 4, 2024"
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines):
                next_line = lines[j].strip()
                m = re.match(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:,?\s+(\d{4}))?", next_line, re.IGNORECASE)
                if m:
                    current_date = _parse_date(m.group(2), m.group(1), m.group(3), ref_year)
                    i = j + 1
                    continue
            i += 1
            continue

        # ── Detect time line ───────────────────────────────────────────────
        is_time = TIME_PATTERN.match(line) or line.lower() == "all day"
        if is_time and current_date is not None:
            time_str = _clean_time(line)

            # Skip "USD" currency line
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and lines[j].strip().upper() == "USD":
                j += 1

            # Skip empty lines and icon/impact placeholders
            while j < len(lines) and not lines[j]:
                j += 1

            # Next non-empty line = event name
            if j < len(lines):
                event_name = lines[j].strip()

                # Skip header/noise lines
                skip_patterns = ["date", "currency", "impact", "alerts", "detail",
                                  "actual", "forecast", "previous", "graph",
                                  "up next", "search events", "day 2", "all"]
                if event_name.lower() in skip_patterns or not event_name:
                    i += 1
                    continue

                impact = classify_impact(event_name)
                if impact is not None:
                    events.append({
                        "date":        current_date,
                        "time":        time_str,
                        "event":       event_name,
                        "impact":      impact,
                        "source_file": path.name,
                    })
                i = j + 1
                continue

        i += 1

    return events


def _infer_year(filename: str, text: str) -> int:
    """Try to extract year from filename or text content."""
    # e.g. "jan_feb_2024.txt" or "2024_q1.txt"
    m = re.search(r"(20\d{2})", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d{2})", text[:500])
    if m:
        return int(m.group(1))
    return datetime.now().year


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Forex Factory USD Calendar Parser")
    print("=" * 55)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT}")
    print()

    if not INPUT_DIR.exists():
        print(f"ERROR: {INPUT_DIR} not found.")
        sys.exit(1)

    files = (
        sorted(INPUT_DIR.glob("*.txt")) +
        sorted(INPUT_DIR.glob("*.text"))
    )

    if not files:
        print(f"ERROR: No .txt files found in {INPUT_DIR}")
        print("Paste FF calendar text into .txt files and place them in data_usd/")
        sys.exit(1)

    print(f"Found {len(files)} file(s):\n")

    all_events = []
    for f in files:
        events = parse_text_file(f)
        red   = sum(1 for e in events if e["impact"] == "red")
        grey  = sum(1 for e in events if e["impact"] == "grey")
        print(f"  {f.name}")
        print(f"    → {red} red events, {grey} grey events")
        all_events.extend(events)

    if not all_events:
        print("\nNo events found. Check that your .txt files contain pasted FF calendar text.")
        sys.exit(1)

    df = (
        pl.DataFrame(all_events)
        .with_columns(pl.col("date").cast(pl.Date))
        .unique(subset=["date", "time", "event"], keep="first")
        .sort(["date", "time"])
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT)

    print(f"\n{'─' * 55}")
    print(f"  Saved {len(df)} events → {OUTPUT.name}")
    print(f"  Date range : {df['date'].min()} → {df['date'].max()}")
    print(f"\n  Breakdown:")
    for row in df.group_by("impact").len().sort("impact").iter_rows():
        print(f"    {row[0]:<8} {row[1]} events")
    print(f"\n  Sample (first 10 rows):")
    print(df.select(["date", "time", "event", "impact"]).head(10))


if __name__ == "__main__":
    main()