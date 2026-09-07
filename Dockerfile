# syntax=docker/dockerfile:1
# Multi-stage build. The dependency layer is cached independently of application
# source, so an edit to src/ rebuilds in seconds rather than re-resolving the tree.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, without the project itself: this layer changes only when
# pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# Run unprivileged: nothing in this service needs root.
RUN groupadd --system sightline && useradd --system --gid sightline --create-home sightline

WORKDIR /app

COPY --from=builder --chown=sightline:sightline /app/.venv /app/.venv
COPY --from=builder --chown=sightline:sightline /app/src /app/src
COPY --chown=sightline:sightline migrations ./migrations
COPY --chown=sightline:sightline alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER sightline
EXPOSE 8000

CMD ["uvicorn", "sightline.composition.bootstrap:create_application", "--factory", "--host", "0.0.0.0", "--port", "8000"]
