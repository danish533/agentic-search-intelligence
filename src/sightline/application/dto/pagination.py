"""Pagination primitives shared by every list use case.

Spec S4.2 requires ``?page=1&per_page=20`` on the queries endpoint. Modelling the request and
the page as DTOs keeps offset arithmetic in one place instead of scattered across routers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_PAGE = 1
DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100


@dataclass(frozen=True, slots=True)
class PageRequest:
    """A validated, 1-indexed pagination request."""

    page: int = DEFAULT_PAGE
    per_page: int = DEFAULT_PER_PAGE

    def __post_init__(self) -> None:
        # Defensive clamping rather than raising: the API layer validates and returns 422 for
        # malformed input, so anything reaching here is already a programming-level value.
        object.__setattr__(self, "page", max(1, self.page))
        object.__setattr__(self, "per_page", min(max(1, self.per_page), MAX_PER_PAGE))

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.per_page

    @property
    def limit(self) -> int:
        return self.per_page


@dataclass(frozen=True, slots=True)
class Page[T]:
    """One page of results plus the totals a client needs to paginate."""

    items: Sequence[T]
    total_items: int
    page: int
    per_page: int

    @property
    def total_pages(self) -> int:
        if self.per_page <= 0:
            return 0
        return (self.total_items + self.per_page - 1) // self.per_page

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        return self.page > 1
