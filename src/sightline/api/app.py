"""FastAPI application factory.

Receives an already-wired container rather than building one. That is what keeps the HTTP ring
free of any dependency on the composition root - and therefore, transitively, on
infrastructure. The wiring entry point is
:func:`sightline.composition.bootstrap.create_application`.

A factory rather than a module-level ``app`` object, so a test can construct an isolated
application with substituted collaborators and nothing is built at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI

from sightline.api.errors import register_exception_handlers
from sightline.api.middleware import CorrelationMiddleware
from sightline.api.v1.routers import health, profiles, queries, recommendations, runs
from sightline.application.ports.container import UseCaseContainer
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)

API_V1_PREFIX = "/api/v1"

_DESCRIPTION = """\
A multi-agent DAG (LangChain/LangGraph) that answers search and AI-visibility questions
end to end: query planning, retrieval from DataForSEO, extraction, analysis, and a final
structured report.

Every response carries an `X-Correlation-ID`. Quote it to trace a request node by node.
"""


def create_app(
    *,
    container: UseCaseContainer,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    """Build the HTTP application around an already-wired container."""
    app = FastAPI(
        title="Agentic Search Intelligence System",
        description=_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.container = container
    app.add_middleware(CorrelationMiddleware)
    register_exception_handlers(app)

    for router in (profiles.router, runs.router, queries.router, recommendations.router):
        app.include_router(router, prefix=API_V1_PREFIX)
    # Health sits outside the versioned prefix: orchestrators probe a stable path, and it must
    # not move when the API version does.
    app.include_router(health.router)

    return app
