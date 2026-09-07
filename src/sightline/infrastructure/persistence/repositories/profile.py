"""SQLAlchemy implementation of ``ProfileRepository``."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from sightline.domain.entities.profile import Profile, normalize_domain
from sightline.domain.errors import DuplicateProfileError, ProfileNotFoundError
from sightline.infrastructure.persistence.mappers import profile_to_domain, profile_to_model
from sightline.infrastructure.persistence.models.profile import ProfileModel

#: The unique index on profiles.domain. Matched by name so an unrelated integrity error is
#: never mislabelled as a duplicate profile.
_DOMAIN_UNIQUE_INDEX = "ix_profiles_domain"


class SqlAlchemyProfileRepository:
    """Profile persistence bound to one session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, profile: Profile) -> None:
        """Insert a profile, translating a duplicate-domain collision into a domain error.

        The caller checks ``find_by_domain`` first, but that is check-then-act: two concurrent
        requests for the same domain both see nothing and both insert. The database settles it,
        and this turns its answer into ``DuplicateProfileError`` so the API returns 409 rather
        than falling through to a 500.

        The flush is what makes the collision surface here, at the layer allowed to know what
        an ``IntegrityError`` is, instead of at commit time inside the use case.
        """
        self._session.add(profile_to_model(profile))
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if _DOMAIN_UNIQUE_INDEX in str(exc.orig):
                raise DuplicateProfileError(profile.domain) from exc
            raise

    async def get(self, profile_uuid: UUID) -> Profile:
        model = await self._session.get(ProfileModel, profile_uuid)
        if model is None:
            raise ProfileNotFoundError(profile_uuid)
        return profile_to_domain(model)

    async def find_by_domain(self, domain: str) -> Profile | None:
        # Normalised on the way in so that "https://WWW.Example.com/" finds the row stored as
        # "example.com" - the same normalisation the entity applies on construction.
        result = await self._session.scalars(
            select(ProfileModel).where(ProfileModel.domain == normalize_domain(domain))
        )
        model = result.first()
        return profile_to_domain(model) if model is not None else None
