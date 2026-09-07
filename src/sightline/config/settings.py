"""Application configuration.

Every tunable value in the system is defined here and sourced from the environment
(CLAUDE.md rule A7). No timeout, retry count, model name, URL, or credential may appear
as a literal anywhere else in the codebase.

Settings are grouped into cohesive blocks, each with its own ``env_prefix``, so the
environment reads as a flat, documented surface (see ``.env.example``) while the code
accesses it as a typed object graph.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def credential_is_present(secret: SecretStr | None) -> bool:
    """Whether a credential is actually usable.

    ``bool(SecretStr(""))`` is ``False`` but ``bool(SecretStr("   "))`` is ``True``, so a
    truthiness check alone accepts a whitespace-only value - which then fails at request time
    as an opaque 401 rather than at startup as a missing credential. A stray space survives a
    copy-paste far more often than an empty field does.
    """
    return secret is not None and bool(secret.get_secret_value().strip())


class Environment(StrEnum):
    """Deployment environment. Controls logging verbosity and error verbosity."""

    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class LLMProviderName(StrEnum):
    """Selectable LLM backends.

    ``FAKE`` is a deterministic, offline, schema-valid provider used by the entire test
    suite so that tests never require network access or an API key (CLAUDE.md R7).
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    FAKE = "fake"


class DataForSEOMode(StrEnum):
    """How DataForSEO is reached.

    The assessment spec (S3.4) explicitly permits mocked responses behind a clearly-marked
    flag. All three modes share one client, one set of schemas, one validation path and one
    retry policy - only the transport differs.
    """

    MOCK = "mock"
    SANDBOX = "sandbox"
    LIVE = "live"


_ENV_FILE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    frozen=True,
)


class AppSettings(BaseSettings):
    """Process-level identity and behaviour."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="APP_")

    name: str = "agentic-search-intelligence"
    environment: Environment = Environment.LOCAL
    host: str = "0.0.0.0"  # noqa: S104 - containerised service must bind all interfaces
    port: int = Field(default=8000, ge=1, le=65535)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pool configuration."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="DATABASE_")

    url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://sightline:sightline@127.0.0.1:5432/sightline"),
        description="Async SQLAlchemy DSN. Must use the asyncpg driver. Uses 127.0.0.1 "
        "rather than localhost: on Windows the latter can resolve to ::1 first while "
        "Docker publishes on IPv4, which surfaces as a connection refused with everything "
        "apparently running.",
    )
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    echo_sql: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        return str(self.url)


class LLMSettings(BaseSettings):
    """LLM provider selection and per-call limits.

    Agents depend on the ``LLMProvider`` port, never on a vendor SDK; this block only
    decides which adapter the composition root builds.
    """

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="LLM_")

    provider: LLMProviderName = LLMProviderName.FAKE
    model: str = Field(
        default="gpt-4o-mini",
        description="Provider-specific model id. Ignored by the 'fake' provider.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None


class DataForSEOSettings(BaseSettings):
    """DataForSEO transport, credentials and per-call timeouts."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="DATAFORSEO_")

    mode: DataForSEOMode = DataForSEOMode.MOCK
    login: SecretStr | None = None
    password: SecretStr | None = None
    live_base_url: str = "https://api.dataforseo.com"
    sandbox_base_url: str = "https://sandbox.dataforseo.com"
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=30.0, gt=0)
    default_location_code: int = Field(
        default=2840, description="DataForSEO location code for the United States."
    )
    default_language_code: str = "en"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_url(self) -> str:
        return self.sandbox_base_url if self.mode is DataForSEOMode.SANDBOX else self.live_base_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requires_credentials(self) -> bool:
        return self.mode is not DataForSEOMode.MOCK


class ResilienceSettings(BaseSettings):
    """Retry, backoff and circuit-breaker policy for external dependencies (CLAUDE.md R5)."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="RESILIENCE_")

    max_attempts: int = Field(default=3, ge=1, description="Total attempts, including the first.")
    backoff_initial_seconds: float = Field(default=0.5, gt=0)
    backoff_multiplier: float = Field(default=2.0, gt=1.0)
    backoff_max_seconds: float = Field(default=8.0, gt=0)
    jitter_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of the computed delay applied as random jitter, to avoid "
        "synchronised retry storms across concurrent fan-out branches.",
    )
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_reset_seconds: float = Field(default=30.0, gt=0)
    circuit_breaker_half_open_successes: int = Field(default=1, ge=1)


class PipelineSettings(BaseSettings):
    """Bounds on a single DAG run."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="PIPELINE_")

    max_planned_queries: int = Field(default=8, ge=1, le=50)
    max_retrieval_concurrency: int = Field(default=4, ge=1, le=32)
    min_successful_retrievals: int = Field(
        default=1,
        ge=1,
        description="Below this, the graph routes to the degradation node and the run is "
        "reported as 'partial' rather than 'completed'.",
    )
    node_timeout_seconds: float = Field(default=120.0, gt=0)


class ObservabilitySettings(BaseSettings):
    """Structured logging, tracing and metrics (CLAUDE.md R6)."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG, env_prefix="OBSERVABILITY_")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    log_node_payloads: bool = Field(
        default=True,
        description="Include redacted node inputs/outputs in trace logs. Disable for very "
        "high-volume production traffic.",
    )
    max_logged_payload_chars: int = Field(default=2000, gt=0)

    # LangSmith is deferred by decision (see CLAUDE.md S4) but the seam is kept: when a key
    # is present the composition root attaches the tracing callback without a code change.
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "agentic-search-intelligence"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_api_key is not None


class Settings(BaseSettings):
    """Root settings object. Built once per process and injected, never imported ad hoc."""

    model_config = SettingsConfigDict(**_ENV_FILE_CONFIG)

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    dataforseo: DataForSEOSettings = Field(default_factory=DataForSEOSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that environment parsing and validation happen exactly once. Tests clear the
    cache via ``get_settings.cache_clear()`` when they need a different environment.
    """
    return Settings()
