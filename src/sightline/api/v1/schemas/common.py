"""Shared response envelopes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sightline.application.dto.pagination import Page


class PaginationMeta(BaseModel):
    """Pagination state for a list response (spec S4.2 ``?page=&per_page=``)."""

    model_config = ConfigDict(frozen=True)

    page: int
    per_page: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def from_page(cls, page: Page[Any]) -> PaginationMeta:
        return cls(
            page=page.page,
            per_page=page.per_page,
            total_items=page.total_items,
            total_pages=page.total_pages,
            has_next=page.has_next,
            has_previous=page.has_previous,
        )


class ErrorResponse(BaseModel):
    """One error shape for every failure, so a client needs one branch, not seven.

    ``correlation_id`` is included deliberately: it is the identifier that ties this response
    to the server-side trace, so a user reporting a failure can hand over something that
    reconstructs the whole request.
    """

    model_config = ConfigDict(frozen=True)

    error: str = Field(description="Machine-readable error code, e.g. 'profile_not_found'.")
    message: str = Field(description="Human-readable explanation.")
    detail: list[dict[str, Any]] | None = Field(
        default=None, description="Field-level validation problems, when applicable."
    )
    correlation_id: str | None = Field(
        default=None, description="Trace identifier for this request."
    )
