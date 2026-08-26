"""SQLAlchemy engine and Schedule queries.

Follows the course DB pattern from ``Course18/OpenAI API - Basics.ipynb``:
``create_engine`` + ``pd.read_sql(text(...), con=engine)``. Two deliberate
differences: the URL comes from ``settings`` rather than hardcoded credentials,
and we keep the Engine instead of chaining ``.connect()`` onto it, so pandas
manages connections (CLAUDE.md 4.7).

Every query is parameterized. Never f-string candidate input into SQL.

Dates and times are stored as ISO text (``YYYY-MM-DD`` / ``HH:MM:SS``), which
compares and orders correctly as strings — so ``date >= :from_date`` is a valid
chronological filter.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from app.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine for ``settings.db_url``.

    Cached: one engine per process, with SQLAlchemy's own connection pooling
    underneath.
    """
    return create_engine(settings.db_url)


def get_available_slots(
    from_date: str,
    position: str = "Python Dev",
    limit: int = 3,
    to_date: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> pd.DataFrame:
    """Return the nearest available slots for a position, on or after a date.

    The optional bounds exist so a query can be confined to the window the
    *candidate* named. Asked "when are you free?", a candidate answers with a day
    and often a half of it ("Tuesday afternoon"); an unbounded forward search
    would answer with Tuesday morning, which is not what they said they could do.

    Args:
        from_date: ISO date (YYYY-MM-DD) to search forward from, inclusive.
        position: Role name as stored in the Schedule table.
        limit: Maximum number of slots to return.
        to_date: Last ISO date to consider, inclusive. ``None`` searches forward
            with no upper bound. Pass the same value as ``from_date`` to stay
            inside one single day.
        time_from: Earliest ``HH:MM:SS`` to consider, inclusive.
        time_to: Latest ``HH:MM:SS`` to consider, inclusive.

    Returns:
        A DataFrame with columns ``date``, ``time``, ordered chronologically.
        Empty when nothing is free in the window — which is a normal answer, not
        an error: the candidate may have named a day the schedule cannot serve.
    """
    # Clauses are assembled rather than interpolated: every value stays a bound
    # parameter (CLAUDE.md 4.7), only the fixed SQL fragments are concatenated.
    clauses = ["position = :position", "available = 1", "date >= :from_date"]
    params: dict[str, object] = {
        "position": position,
        "from_date": from_date,
        "limit": limit,
    }
    if to_date:
        clauses.append("date <= :to_date")
        params["to_date"] = to_date
    if time_from:
        clauses.append("time >= :time_from")
        params["time_from"] = time_from
    if time_to:
        clauses.append("time <= :time_to")
        params["time_to"] = time_to

    return pd.read_sql(
        text(
            "SELECT date, time FROM Schedule "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY date, time LIMIT :limit"
        ),
        con=get_engine(),
        params=params,
    )


def get_availability_calendar(
    from_date: str,
    days: int = 28,
    position: str = "Python Dev",
) -> pd.DataFrame:
    """Return every Schedule row in a window, with its availability.

    Unlike :func:`get_available_slots` this keeps the taken slots too, so a
    calendar view can tell "booked" apart from "never offered". The seeded table
    holds no Monday or Saturday rows and nothing outside 09:00-17:00
    (CLAUDE.md 6.3), so a day with zero rows is normal — it is the caller's job
    to render that as "no slots", never to invent times the DB does not have.

    Args:
        from_date: ISO date (YYYY-MM-DD) the window starts on (inclusive). A
            full ISO timestamp (``2026-08-26T17:04:22Z``) is accepted too, so a
            conversation anchor can be passed straight through.
        days: Window length in days.
        position: Role name as stored in the Schedule table.

    Returns:
        DataFrame with columns ``date`` (str YYYY-MM-DD), ``time`` (str HH:MM:SS),
        ``available`` (int 0/1), ordered by date then time. Empty when the window
        falls outside the seeded range.
    """
    start = from_date[:10]
    # The end is exclusive and computed here rather than in SQL: SQLite has no
    # DATE type, and ISO text dates compare correctly as strings.
    end = (dt.date.fromisoformat(start) + dt.timedelta(days=days)).isoformat()
    return pd.read_sql(
        text(
            "SELECT date, time, available FROM Schedule "
            "WHERE position = :position AND date >= :from_date AND date < :to_date "
            "ORDER BY date, time"
        ),
        con=get_engine(),
        params={"position": position, "from_date": start, "to_date": end},
    )


def is_slot_available(date: str, time: str, position: str = "Python Dev") -> bool:
    """Check whether one specific slot is free.

    Used to validate a slot the candidate proposed themselves, before confirming
    it. Note the seeded table holds no Monday or Saturday slots and nothing
    outside 09:00-17:00 (CLAUDE.md 6.3), so a plausible-sounding request can
    legitimately be unavailable.

    Args:
        date: ISO date (YYYY-MM-DD).
        time: 24h time (HH:MM:SS).
        position: Role name as stored in the Schedule table.
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT available FROM Schedule "
                "WHERE position = :position AND date = :date AND time = :time"
            ),
            {"position": position, "date": date, "time": time},
        ).fetchone()
    return bool(row) and bool(row[0])


def book_slot(date: str, time: str, position: str = "Python Dev") -> bool:
    """Mark a slot as taken by setting ``available = 0``.

    Args:
        date: ISO date (YYYY-MM-DD).
        time: 24h time (HH:MM:SS).
        position: Role name as stored in the Schedule table.

    Returns:
        True when a row was updated, False when the slot did not exist or was
        already taken.
    """
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                "UPDATE Schedule SET available = 0 "
                "WHERE position = :position AND date = :date AND time = :time "
                "AND available = 1"
            ),
            {"position": position, "date": date, "time": time},
        )
    return result.rowcount == 1
