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
) -> pd.DataFrame:
    """Return the nearest available slots for a position, on or after a date.

    Args:
        from_date: ISO date (YYYY-MM-DD) to search forward from.
        position: Role name as stored in the Schedule table.
        limit: Maximum number of slots to return.

    Returns:
        A DataFrame with columns ``date``, ``time``, ordered chronologically.
        Empty when nothing is free in the remaining range.
    """
    return pd.read_sql(
        text(
            "SELECT date, time FROM Schedule "
            "WHERE position = :position AND available = 1 AND date >= :from_date "
            "ORDER BY date, time LIMIT :limit"
        ),
        con=get_engine(),
        params={"position": position, "from_date": from_date, "limit": limit},
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
