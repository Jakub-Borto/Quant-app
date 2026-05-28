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
OUTPUT = THIS_DIR.parent / "data" / "news_and_holidays" / "ff_usd_events.parquet"

# ── Impact classification ──────────────────────────────────────────────────────


GREY_KEYWORDS = {"bank holiday", "holiday"}

# No market impact — skip entirely
BLOCKLIST = {
    "daylight saving time", "daylight savings time", "dst",
    "clocks change", "clock change",
}


def classify_impact(event_name: str) -> str | None:
    name_lower = event_name.lower().strip()

    # Always block these — no market impact
    for kw in BLOCKLIST:
        if kw in name_lower:
            return None

    # Grey: bank holidays
    for kw in GREY_KEYWORDS:
        if kw in name_lower:
            return "grey"

    # Everything else is red — FF page is already filtered to red/grey only
    return "red" 


# ── Helpers ────────────────────────────────────────────────────────────────────

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
DAYS   = {"mon","tue","wed","thu","fri","sat","sun"}

SKIP_LINES = {
    "date", "currency", "impact", "alerts", "detail",
    "actual", "forecast", "previous", "graph",
    "up next", "search events", "day 2", "all",
}


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


def _parse_date(day: str, month: str, year: str | None, ref_year: int) -> date:
    y = int(year) if year else ref_year
    return datetime.strptime(f"{month} {day} {y}", "%b %d %Y").date()


def _infer_year(filename: str, text: str) -> int:
    m = re.search(r"(20\d{2})", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d{2})", text[:500])
    if m:
        return int(m.group(1))
    return datetime.now().year


MONTH_TO_NUM = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}

def _validate_filename_vs_content(path: Path, text: str) -> bool:
    """
    Check that the year and month numbers in the filename match the date range
    on the first line of the file.

    Filename format: YYYY_MM_MM.txt  e.g. 2010_01_02.txt
    First line format: "Jan 1, 2010 - Feb 28, 2010"
    """
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})\.txt", path.name, re.IGNORECASE)
    if not m:
        return True  # non-standard filename — skip check

    fn_year   = int(m.group(1))
    fn_month1 = int(m.group(2))
    fn_month2 = int(m.group(3))

    # Parse first non-empty line e.g. "Jan 1, 2010 - Feb 28, 2010"
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    months_found = re.findall(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        first_line, re.IGNORECASE
    )
    year_found = re.search(r"(20\d{2})", first_line)

    if len(months_found) < 2 or not year_found:
        print(f"  WARNING {path.name}: could not parse date range from first line: {first_line!r}")
        return False

    content_year   = int(year_found.group(1))
    content_month1 = MONTH_TO_NUM[months_found[0].lower()]
    content_month2 = MONTH_TO_NUM[months_found[1].lower()]

    ok = True
    if fn_year != content_year:
        print(f"  WARNING {path.name}: filename says year {fn_year} but content says {content_year} — wrong file pasted?")
        ok = False
    if fn_month1 != content_month1 or fn_month2 != content_month2:
        print(f"  WARNING {path.name}: filename says months {fn_month1:02d}/{fn_month2:02d} "
              f"but content says {months_found[0]}/{months_found[1]} — wrong file pasted?")
        ok = False

    return ok


def _next_nonempty(lines: list[str], i: int) -> tuple[int, str]:
    """Return (index, line) of next non-empty line after i."""
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return (j, lines[j].strip()) if j < len(lines) else (j, "")


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_text_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]
    ref_year = _infer_year(path.name, text)
    _validate_filename_vs_content(path, text)

    events = []
    current_date = None
    current_time = None   # last seen time — reused by shared-time rows
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        # ── Date header (day of week) ──────────────────────────────────────
        if line.lower()[:3] in DAYS:
            j, next_line = _next_nonempty(lines, i)
            m = re.match(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:,?\s+(\d{4}))?",
                next_line, re.IGNORECASE
            )
            if m:
                current_date = _parse_date(m.group(2), m.group(1), m.group(3), ref_year)
                current_time = None   # reset time on new day
                i = j + 1
                continue
            i += 1
            continue

        # ── Time line ──────────────────────────────────────────────────────
        is_time = TIME_PATTERN.match(line) or line.lower() == "all day"
        if is_time and current_date is not None:
            current_time = _clean_time(line)

            # consume: optional empty lines, then "USD", then event name
            j, next_line = _next_nonempty(lines, i)
            if next_line.upper() == "USD":
                j, next_line = _next_nonempty(lines, j)

            event_name = next_line
            if event_name and event_name.lower() not in SKIP_LINES:
                impact = classify_impact(event_name)
                if impact is not None:
                    events.append({
                        "date":        current_date,
                        "time":        current_time,
                        "event":       event_name,
                        "impact":      impact,
                        "source_file": path.name,
                    })
                else:
                    print(f"  SKIPPED {current_date} | {event_name} | {path.name}")
            i = j + 1
            continue

        # ── Shared-time row: bare "USD" line with no preceding time ────────
        # FF omits the time when multiple events share the same slot.
        # e.g. Non-Farm Employment Change at 8:30, then Unemployment Rate
        # appears as just "USD\n\nUnemployment Rate" with no time.
        if line.upper() == "USD" and current_date is not None and current_time is not None:
            j, event_name = _next_nonempty(lines, i)
            if event_name and event_name.lower() not in SKIP_LINES:
                # Make sure it's not a time line (that would be a new event)
                if not TIME_PATTERN.match(event_name) and event_name.lower() != "all day":
                    impact = classify_impact(event_name)
                    if impact is not None:
                        events.append({
                            "date":        current_date,
                            "time":        current_time,
                            "event":       event_name,
                            "impact":      impact,
                            "source_file": path.name,
                        })
                    else:
                        print(f"  SKIPPED {current_date} | {event_name} | {path.name}")
                    i = j + 1
                    continue

        i += 1

    return events


# ── Main ───────────────────────────────────────────────────────────────────────

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

    files = sorted(INPUT_DIR.glob("*.txt")) + sorted(INPUT_DIR.glob("*.text"))

    if not files:
        print(f"ERROR: No .txt files found in {INPUT_DIR}")
        print("Paste FF calendar text into .txt files and place them in data_usd/")
        sys.exit(1)

    print(f"Found {len(files)} file(s):\n")

    all_events = []
    for f in files:
        events = parse_text_file(f)
        red  = sum(1 for e in events if e["impact"] == "red")
        grey = sum(1 for e in events if e["impact"] == "grey")
        print(f"  {f.name}")
        print(f"    → {red} red events, {grey} grey events")
        all_events.extend(events)

    if not all_events:
        print("\nNo events found.")
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