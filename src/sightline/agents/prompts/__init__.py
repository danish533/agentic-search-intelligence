"""Prompt templates, versioned per agent.

Kept out of node bodies so a prompt change is a reviewable diff on its own.
"""

from __future__ import annotations

from datetime import UTC, datetime


def today_line() -> str:
    """State the current date for the model.

    A model with no date supplied falls back to its training distribution, which skews older.
    That produced a recommendation titled "Top SEO Tools ... in 2023" during a 2026 run - and
    the recommendation title is the product's actual deliverable, not an internal detail. A
    stale year in a title is one of the specific things that suppresses a page's ranking, so
    the output was actively working against the outcome it exists to improve.
    """
    return f"Today's date is {datetime.now(UTC):%d %B %Y}."
