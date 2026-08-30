"""Interview availability calendar for the Streamlit sidebar.

Presentation only — the SQL lives in ``app/modules/database/engine.py`` and the
decision to actually offer a slot belongs to the Scheduling Advisor. This module
draws what the schedule already says (ENGINEERING.md section 2, rule 4).

The grid is week-by-week: one row per week, one column per weekday. A day shows
its number and how many slots are still free. Three states are worth telling
apart, and the third is the one that confuses people:

* free slots left,
* the day exists in the schedule but every slot is taken,
* the day has no slots at all — Mondays and Saturdays are never seeded, and
  nothing exists outside 09:00-17:00 (ENGINEERING.md 6.3). Those columns are
  legitimately always empty, so they are labeled rather than left mysterious.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app.modules.database.engine import get_availability_calendar, get_available_slots

#: Column headings, Monday first — short enough to survive a narrow sidebar.
WEEKDAY_HEADS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]

#: Glyph per cell state. Colour *and* a count, so the state survives a
#: greyscale screenshot and a colour-blind reader.
FREE_MARK = "🟩"
BOOKED_MARK = "🟧"
EMPTY_MARK = "⬜"

#: How many upcoming slots the "next bookable" list shows.
NEXT_SLOTS = 5


@st.cache_data(show_spinner=False)
def _load_schedule(from_date: str, weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the calendar window and the next free slots, once per input.

    Cached so that every Streamlit rerun — each keystroke in the chat box is one
    — does not re-hit SQLite. The key is ``(from_date, weeks)``; ``from_date`` is
    normalised to a plain date by the caller so the cache is not missed by the
    seconds ticking in a conversation anchor.

    Args:
        from_date: ISO date (YYYY-MM-DD) the window starts on.
        weeks: Number of weeks to cover.

    Returns:
        ``(calendar, upcoming)`` — every row in the window with its
        availability, and the nearest free slots.
    """
    calendar = get_availability_calendar(from_date, days=weeks * 7)
    upcoming = get_available_slots(from_date, limit=NEXT_SLOTS)
    return calendar, upcoming


def _free_by_day(calendar: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Collapse the slot rows into per-day totals.

    Args:
        calendar: Rows as returned by ``get_availability_calendar``.

    Returns:
        ``{"YYYY-MM-DD": (slots_in_schedule, slots_still_free)}``. Days absent
        from the mapping have no rows in the schedule at all.
    """
    if calendar.empty:
        return {}
    grouped = calendar.groupby("date")["available"].agg(["count", "sum"])
    return {
        str(day): (int(row["count"]), int(row["sum"]))
        for day, row in grouped.iterrows()
    }


def _cell_text(day: dt.date, totals: dict[str, tuple[int, int]]) -> str:
    """Render one calendar cell: the day number over its availability mark.

    Args:
        day: The date the cell stands for.
        totals: Per-day totals from :func:`_free_by_day`.

    Returns:
        Markdown for the cell. The two trailing spaces are a markdown hard line
        break, which keeps the number and the mark stacked in a narrow column.
    """
    slots, free = totals.get(day.isoformat(), (0, 0))
    if free:
        mark = f"{FREE_MARK}{free}"
    elif slots:
        mark = f"{BOOKED_MARK}0"
    else:
        mark = f"{EMPTY_MARK}–"
    return f"{day.day}  \n{mark}"


def _render_grid(anchor: dt.date, weeks: int, totals: dict[str, tuple[int, int]]) -> None:
    """Draw the week-by-week grid.

    Rows start on the Monday of the anchor's week so the weekday columns line
    up; days before the anchor are left blank rather than reported as empty —
    they are outside the window, not out of slots.

    Args:
        anchor: First day the window covers.
        weeks: Number of week rows to draw.
        totals: Per-day totals from :func:`_free_by_day`.
    """
    heads = st.columns(7, gap="small")
    for column, label in zip(heads, WEEKDAY_HEADS):
        column.caption(label)

    monday = anchor - dt.timedelta(days=anchor.weekday())
    for week in range(weeks):
        cells = st.columns(7, gap="small")
        for weekday, column in enumerate(cells):
            day = monday + dt.timedelta(days=week * 7 + weekday)
            column.markdown(" " if day < anchor else _cell_text(day, totals))


def _render_next_slots(upcoming: pd.DataFrame) -> None:
    """List the nearest bookable slots in human-readable form.

    Args:
        upcoming: Rows with ``date`` and ``time`` columns, chronologically
            ordered, as returned by ``get_available_slots``.
    """
    st.markdown("**Next bookable slots**")
    if upcoming.empty:
        st.caption("Nothing free from this date on — the seeded range may have run out.")
        return

    lines = []
    for row in upcoming.itertuples():
        day = dt.date.fromisoformat(str(row.date))
        # HH:MM:SS -> HH:MM; nobody books an interview to the second.
        lines.append(f"- {day:%a} {day.day} {day:%b} — {str(row.time)[:5]}")
    st.markdown("\n".join(lines))


def render_calendar(from_date: str, weeks: int = 4) -> None:
    """Render the interview availability calendar in the current container.

    Draws a compact month-style grid of the next ``weeks`` weeks followed by the
    next bookable slots. Degrades to a one-line caption when the schedule
    database is missing or unreadable: this is the only place a missing database
    surfaces in the UI, so it must never take the app down with it.

    Args:
        from_date: The conversation anchor. Either a plain ISO date
            (``2026-08-26``) or a full timestamp (``2026-08-26T17:04:22Z``).
        weeks: How many weeks the grid covers, starting from ``from_date``.
    """
    weeks = max(1, weeks)
    try:
        anchor = dt.date.fromisoformat(from_date[:10])
    except (ValueError, TypeError):
        st.caption(f"📅 `{from_date}` is not a date I can read — expected YYYY-MM-DD.")
        return

    try:
        calendar, upcoming = _load_schedule(anchor.isoformat(), weeks)
    except Exception as exc:  # noqa: BLE001 - a missing DB must not break the UI
        st.caption(
            "📅 Schedule database unavailable — run "
            "`python scripts/seed_database.py` to seed it. "
            f"({type(exc).__name__}: {str(exc)[:80]})"
        )
        return

    last = anchor + dt.timedelta(days=weeks * 7 - 1)
    st.caption(f"{anchor:%b %d} → {last:%b %d, %Y} · day · free slots")

    _render_grid(anchor, weeks, _free_by_day(calendar))

    st.caption(
        f"{FREE_MARK} free · {BOOKED_MARK} fully booked · {EMPTY_MARK} no slots — "
        "interviews run Tue-Fri and Sun, 09:00-17:00, so Mon and Sat are always empty."
    )

    _render_next_slots(upcoming)
