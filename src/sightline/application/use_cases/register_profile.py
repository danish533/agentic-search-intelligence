"""Register a brand profile (spec S4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.entities.profile import Profile, normalize_domain
from sightline.domain.errors import DuplicateProfileError
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RegisterProfileInput:
    name: str
    domain: str
    industry: str = ""
    description: str = ""
    competitors: tuple[str, ...] = field(default_factory=tuple)


class RegisterProfileUseCase:
    """Creates a profile, rejecting a domain that is already registered."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def execute(self, data: RegisterProfileInput) -> Profile:
        """Raise :class:`DuplicateProfileError` (HTTP 409) if the domain already exists.

        The check and the insert share one transaction, and the database carries a unique
        constraint on ``domain`` as well - the check narrows the race window, the constraint
        closes it.
        """
        async with self._unit_of_work() as uow:
            existing = await uow.profiles.find_by_domain(data.domain)
            if existing is not None:
                raise DuplicateProfileError(normalize_domain(data.domain))

            profile = Profile(
                name=data.name,
                domain=data.domain,
                industry=data.industry,
                description=data.description,
                competitors=tuple(data.competitors),
            )
            await uow.profiles.add(profile)
            await uow.commit()

        _logger.info(
            "profile.registered",
            profile_uuid=str(profile.uuid),
            domain=profile.domain,
            competitors=len(profile.competitors),
        )
        return profile
