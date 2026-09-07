# Agentic Search Intelligence System
# `make help` lists every target. `make run` is the single command from spec S7.

SHELL := /bin/bash
UV    := uv

.DEFAULT_GOAL := help
.PHONY: help install run stop logs dev db migrate migration downgrade lint format typecheck arch test test-cov check clean

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install all dependencies
	$(UV) sync --extra dev

run:  ## Start Postgres + API in Docker (single-command entry point)
	docker compose up --build

stop:  ## Stop and remove the Docker stack
	docker compose down --volumes

logs:  ## Tail API logs from the running stack
	docker compose logs -f api

db:  ## Start only Postgres, for running the API locally
	docker compose up -d db

dev:  ## Run the API locally with hot reload (requires `make db`)
	$(UV) run uvicorn sightline.composition.bootstrap:create_application --factory --reload --host 0.0.0.0 --port 8000

migrate:  ## Apply all pending database migrations
	$(UV) run alembic upgrade head

migration:  ## Autogenerate a migration from model changes (make migration m="add x")
	$(UV) run alembic revision --autogenerate -m "$(m)"

downgrade:  ## Roll back the most recent migration
	$(UV) run alembic downgrade -1

lint:  ## Lint, format-check and verify architecture contracts
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
	$(UV) run lint-imports

format:  ## Auto-fix lint findings and format the codebase
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck:  ## Run mypy in strict mode
	$(UV) run mypy

arch:  ## Verify the Clean Architecture dependency contracts only
	$(UV) run lint-imports

test:  ## Run the test suite (offline: fake LLM + mocked DataForSEO)
	$(UV) run pytest

test-cov:  ## Run tests with a coverage report
	$(UV) run pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Run every gate in the Definition of Done

clean:  ## Remove caches and build artefacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
