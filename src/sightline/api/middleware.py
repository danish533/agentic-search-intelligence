"""Request middleware.

Establishes the correlation scope for every request, which is what makes spec S3.6's
"follow a single request node-by-node" work: the id set here propagates through the use case,
into the DAG, through every fan-out branch, and down to each DataForSEO call - without a single
function signature having to carry it.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from sightline.observability.correlation import (
    CORRELATION_ID_HEADER,
    correlation_scope,
    new_correlation_id,
)
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Binds a correlation id for the lifetime of the request and echoes it back.

    An inbound ``X-Correlation-ID`` is honoured so a trace can span several services; absent
    one, a fresh id is minted. Either way the response carries it, so a client always has the
    identifier needed to report a problem.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        inbound = request.headers.get(CORRELATION_ID_HEADER)
        correlation_id = inbound or new_correlation_id()
        started = time.perf_counter()

        # Also stashed on request.state: Starlette's catch-all handler runs in
        # ServerErrorMiddleware, which sits *outside* this middleware, so by the time it
        # builds a 500 the context var has already unwound. Without this the one response
        # that says "quote the correlation id" would not carry one.
        request.state.correlation_id = correlation_id

        with correlation_scope(correlation_id=correlation_id):
            _logger.info(
                "http.request.started",
                method=request.method,
                path=request.url.path,
                inherited_correlation_id=bool(inbound),
            )
            response = await call_next(request)
            duration_ms = (time.perf_counter() - started) * 1000
            _logger.info(
                "http.request.completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
