"""The Recommendation entity: a content action proposed to close a visibility gap."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from sightline.domain.errors import DomainValidationError
from sightline.domain.value_objects.enums import ContentType, Priority

_MAX_TARGET_KEYWORDS = 15


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A concrete content deliverable tied to the query whose gap it addresses (spec S4.2)."""

    run_uuid: UUID
    target_query_uuid: UUID
    content_type: ContentType
    title: str
    rationale: str
    target_keywords: tuple[str, ...] = ()
    priority: Priority = Priority.MEDIUM
    uuid: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        title = self.title.strip()
        if not title:
            raise DomainValidationError("Recommendation.title must not be empty.")

        rationale = self.rationale.strip()
        if not rationale:
            raise DomainValidationError(
                "Recommendation.rationale must not be empty: a recommendation without a "
                "stated reason is not actionable."
            )

        seen: set[str] = set()
        keywords: list[str] = []
        for keyword in self.target_keywords:
            cleaned = keyword.strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                keywords.append(cleaned)

        if len(keywords) > _MAX_TARGET_KEYWORDS:
            raise DomainValidationError(
                f"A recommendation may target at most {_MAX_TARGET_KEYWORDS} keywords."
            )

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "target_keywords", tuple(keywords))
