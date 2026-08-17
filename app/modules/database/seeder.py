"""SQLite port of ``data/db_Tech.sql``.

The original is a T-SQL script targeting SQL Server. It is kept in ``data/`` as
the reference for the schema and the seeding rules; this module reproduces both
in SQLite so the resulting ``data/tech.db`` can be committed and shipped to
Streamlit Community Cloud, which cannot reach a SQL Server instance.

Rules preserved verbatim from the original script:

* **Tuesday-Friday and Sunday only** — Saturday and Monday are excluded
* hourly **09:00-17:00** inclusive, so 9 slots per day
* 4 positions: ``Python Dev``, ``Sql Dev``, ``Analyst``, ``ML``
* ``available`` set pseudo-randomly at roughly 50%

The date range is the one thing that does NOT come from the original script. The
original covered 2024, matching the dataset as it was first supplied; both have
since been moved to the present so the system can be tested against live dates.
The range must always cover every conversation in
``data/sms_conversations.json`` plus a comfortable margin of future slots, or
relative dates resolve to rows that do not exist.

Schema note: SQLite has no DATE or TIME type. ``date`` is stored as
``YYYY-MM-DD`` and ``time`` as ``HH:MM:SS`` — ISO strings, which sort and compare
correctly as text.

Run via ``python scripts/seed_database.py``. Never imported for its side effects.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date as date_type
from datetime import timedelta
from pathlib import Path

#: Matches dbo.Schedule from data/db_Tech.sql.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS Schedule (
    ScheduleID INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,   -- YYYY-MM-DD
    time       TEXT    NOT NULL,   -- HH:MM:SS
    position   TEXT    NOT NULL,
    available  INTEGER NOT NULL    -- 0 / 1
);
"""

#: Wide enough to cover the whole dataset (July-August 2026) and to leave well
#: over a year of bookable future slots for live testing.
START_DATE = date_type(2026, 1, 1)
END_DATE = date_type(2027, 12, 31)

#: Weekday numbers to skip, Monday=0 .. Sunday=6. The original excluded
#: Saturday and Monday.
EXCLUDED_WEEKDAYS = frozenset({0, 5})

#: 09:00 through 17:00 inclusive.
FIRST_HOUR = 9
LAST_HOUR = 17

POSITIONS = ("Python Dev", "Sql Dev", "Analyst", "ML")

#: Probability a generated slot is free, matching the original's ~50%.
AVAILABILITY_RATE = 0.5


def seed(db_path: Path, seed_value: int = 42, overwrite: bool = False) -> int:
    """Create and populate the SQLite Schedule table.

    Args:
        db_path: Where to write the ``.db`` file.
        seed_value: RNG seed. Fixed by default so the committed database is
            reproducible — an unseeded run would produce a different file on
            every execution and pollute the diff.
        overwrite: Replace an existing database instead of leaving it alone.

    Returns:
        The number of rows inserted.
    """
    db_path = Path(db_path)
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{db_path} already exists. Pass --overwrite to rebuild it."
            )
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed_value)
    rows = []
    day = START_DATE
    while day <= END_DATE:
        if day.weekday() not in EXCLUDED_WEEKDAYS:
            iso_day = day.isoformat()
            for hour in range(FIRST_HOUR, LAST_HOUR + 1):
                slot_time = f"{hour:02d}:00:00"
                for position in POSITIONS:
                    available = int(rng.random() < AVAILABILITY_RATE)
                    rows.append((iso_day, slot_time, position, available))
        day += timedelta(days=1)

    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        # The advisor always filters on position + availability and orders by
        # date/time; without this index every lookup is a full scan.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_schedule_lookup "
            "ON Schedule (position, available, date, time);"
        )
        conn.executemany(
            "INSERT INTO Schedule (date, time, position, available) VALUES (?, ?, ?, ?);",
            rows,
        )
        conn.commit()

    return len(rows)


def summarize(db_path: Path) -> str:
    """Return a short human-readable summary of a seeded database.

    Reports the row count, the date range, the distinct weekdays present and the
    share of available slots — enough to sanity-check a seed at a glance.
    """
    with sqlite3.connect(Path(db_path)) as conn:
        total, first, last, free = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date), SUM(available) FROM Schedule;"
        ).fetchone()
        per_position = conn.execute(
            "SELECT position, COUNT(*), SUM(available) FROM Schedule GROUP BY position ORDER BY position;"
        ).fetchall()
        days = [
            date_type.fromisoformat(d)
            for (d,) in conn.execute("SELECT DISTINCT date FROM Schedule;").fetchall()
        ]

    weekdays = sorted({d.strftime("%A") for d in days}, key=lambda n: _WEEKDAY_ORDER.index(n))
    lines = [
        f"  rows          : {total:,}",
        f"  date range    : {first} -> {last}",
        f"  weekdays      : {', '.join(weekdays)}",
        f"  hours         : {FIRST_HOUR:02d}:00-{LAST_HOUR:02d}:00",
        f"  available     : {free:,} / {total:,} ({free / total:.1%})",
        "  per position  :",
    ]
    lines += [
        f"      {name:<12} {count:>6,} slots, {avail:>6,} free ({avail / count:.1%})"
        for name, count, avail in per_position
    ]
    return "\n".join(lines)


_WEEKDAY_ORDER = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
