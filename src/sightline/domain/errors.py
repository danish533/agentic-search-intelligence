"""Domain error hierarchy.

Every error the domain can raise descends from :class:`DomainError`, which lets outer rings
map domain failures onto transport concerns (HTTP status codes, DAG routing decisions)
without catching bare ``Exception`` - forbidden by CLAUDE.md R5.
"""

from __future__ import annotations

from uuid import UUID


class DomainError(Exception):
    """Base class for every error originating in the domain ring."""


class DomainValidationError(DomainError):
    """A domain invariant was violated (e.g. a score outside its permitted range)."""


class EntityNotFoundError(DomainError):
    """A referenced aggregate does not exist.

    Outer rings translate this to HTTP 404.
    """

    entity_name: str = "Entity"

    def __init__(self, identifier: UUID | str) -> None:
        self.identifier = identifier
        super().__init__(f"{self.entity_name} '{identifier}' was not found.")


class ProfileNotFoundError(EntityNotFoundError):
    entity_name = "Profile"


class PipelineRunNotFoundError(EntityNotFoundError):
    entity_name = "Pipeline run"


class DiscoveredQueryNotFoundError(EntityNotFoundError):
    entity_name = "Discovered query"


class DuplicateProfileError(DomainError):
    """A profile already exists for the given domain.

    Outer rings translate this to HTTP 409.
    """

    def __init__(self, domain: str) -> None:
        self.domain = domain
        super().__init__(f"A profile for domain '{domain}' already exists.")


class InvalidRunTransitionError(DomainError):
    """An illegal pipeline-run state transition was attempted."""

    def __init__(self, current: str, requested: str) -> None:
        self.current = current
        self.requested = requested
        super().__init__(f"Cannot transition a pipeline run from '{current}' to '{requested}'.")
