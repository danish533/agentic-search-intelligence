"""Application entry point.

The outermost layer: it builds the container, hands it to the HTTP factory, and owns the
shutdown that releases the connection pool and HTTP client. This is the only module that knows
both how the application is wired and how it is served.

``uvicorn sightline.composition.bootstrap:create_application --factory``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sightline.api.app import create_app
from sightline.composition.container import build_container
from sightline.config.settings import Settings
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


def create_application(settings: Settings | None = None) -> FastAPI:
    """Build the fully wired, servable application."""
    container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _logger.info(
            "application.started",
            environment=container.environment,
            llm_provider=container.llm_provider_name,
            dataforseo_mode=container.dataforseo_mode,
        )
        try:
            yield
        finally:
            await container.aclose()
            _logger.info("application.stopped")

    return create_app(container=container, lifespan=lifespan)
