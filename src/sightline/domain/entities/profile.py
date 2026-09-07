"""The Profile aggregate: a brand/keyword subject that DAG runs are executed against."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sightline.domain.errors import DomainValidationError

_MAX_NAME_LENGTH = 200
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_COMPETITORS = 20


def normalize_domain(raw: str) -> str:
    """Reduce a user-supplied domain to a bare, comparable host.

    ``https://WWW.SurferSEO.com/pricing/`` -> ``surferseo.com``. Normalising at the domain
    boundary means uniqueness checks and visibility matching compare like with like, rather
    than every call site re-implementing string trimming.
    """
    value = raw.strip().lower()
    for scheme in ("https://", "http://"):
        if value.startswith(scheme):
            value = value[len(scheme) :]
    value = value.removeprefix("www.")
    value = value.split("/", maxsplit=1)[0]
    return value.rstrip(".")


@dataclass(frozen=True, slots=True)
class Profile:
    """A brand under analysis, plus the competitor set it is measured against."""

    name: str
    domain: str
    industry: str
    description: str
    competitors: tuple[str, ...] = ()
    uuid: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise DomainValidationError("Profile name must not be empty.")
        if len(name) > _MAX_NAME_LENGTH:
            raise DomainValidationError(
                f"Profile name must be at most {_MAX_NAME_LENGTH} characters."
            )
        if len(self.description) > _MAX_DESCRIPTION_LENGTH:
            raise DomainValidationError(
                f"Profile description must be at most {_MAX_DESCRIPTION_LENGTH} characters."
            )

        domain = normalize_domain(self.domain)
        if not domain or "." not in domain:
            raise DomainValidationError(f"Profile domain '{self.domain}' is not a valid host.")

        if len(self.competitors) > _MAX_COMPETITORS:
            raise DomainValidationError(
                f"A profile may declare at most {_MAX_COMPETITORS} competitors."
            )

        # Normalise competitors, drop blanks and self-references, preserve declared order.
        seen: set[str] = {domain}
        competitors: list[str] = []
        for competitor in self.competitors:
            normalized = normalize_domain(competitor)
            if normalized and normalized not in seen:
                seen.add(normalized)
                competitors.append(normalized)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "industry", self.industry.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "competitors", tuple(competitors))

    def default_research_question(self) -> str:
        """Derive the pipeline's research question when the caller did not supply one.

        Spec S2 frames the input as a natural-language question while S4.2 makes the entry
        point a stored profile with no request body; this bridges the two.
        """
        industry = self.industry or "its market"
        return (
            f"How does {self.name} ({self.domain}) show up in AI-generated answers and "
            f"search results for {industry}, and where are its visibility gaps compared to "
            f"its competitors?"
        )
