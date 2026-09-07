"""Structured JSON logging with mandatory redaction.

Spec S3.6 requires structured logs whose inputs are "redacted where sensitive". Redaction is
applied here, as a structlog processor on the way to the sink - not by remembering to omit
fields at thousands of call sites. A call site cannot leak a credential by forgetting,
because it never gets the chance: every event passes through :func:`redact`.

Standard-library logging (uvicorn, sqlalchemy, alembic) is routed through the same processor
chain, so a run produces one homogeneous JSON stream rather than two interleaved formats that
no log aggregator can parse together.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from sightline.config.settings import ObservabilitySettings
from sightline.observability.correlation import current_context

REDACTED: Final = "***REDACTED***"
_TRUNCATION_SUFFIX: Final = "...[truncated]"
_MAX_REDACTION_DEPTH: Final = 6

#: Exact key names that always carry a credential. Kept exact (not substring) so that
#: legitimate metric fields such as ``total_tokens`` and ``prompt_tokens`` stay visible -
#: a substring match on "token" would blind the very metrics S3.6 asks us to emit.
_SENSITIVE_EXACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "id_token",
        "login",
        "cookie",
        "set-cookie",
        "x-api-key",
        "authorization",
    }
)

#: Substrings that are unambiguous credential markers wherever they appear in a key.
_SENSITIVE_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    # "login" as a fragment, not just an exact key: the field is realistically named
    # `dataforseo_login`, and a username is half a credential. No collision with "logging",
    # which does not contain the substring.
    "login",
    "username",
)

#: Value prefixes that identify a credential even when the key name is innocuous - a
#: last line of defence against a leak through an unexpected field.
_SENSITIVE_VALUE_PREFIXES: Final[tuple[str, ...]] = ("sk-", "sk_live", "Bearer ", "Basic ")


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SENSITIVE_EXACT_KEYS:
        return True
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _looks_like_credential(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in _SENSITIVE_VALUE_PREFIXES)


def redact(value: Any, *, max_chars: int, _depth: int = 0) -> Any:
    """Recursively redact credentials and truncate oversized payloads.

    Returns a structurally identical value with sensitive leaves replaced. Depth-limited so a
    cyclic or pathologically nested API payload cannot stall the logging path.
    """
    if _depth >= _MAX_REDACTION_DEPTH:
        return "...[max depth]"

    # pydantic SecretStr and anything else that hides its own value: never log it.
    if hasattr(value, "get_secret_value"):
        return REDACTED

    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if is_sensitive_key(str(key))
                else redact(item, max_chars=max_chars, _depth=_depth + 1)
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return [redact(item, max_chars=max_chars, _depth=_depth + 1) for item in value]

    if isinstance(value, str):
        if _looks_like_credential(value):
            return REDACTED
        if len(value) > max_chars:
            return value[:max_chars] + _TRUNCATION_SUFFIX
        return value

    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"

    return value


class RedactionProcessor:
    """structlog processor applying :func:`redact` to every event it passes."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    def __call__(
        self,
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        redacted: EventDict = {}
        for key, value in event_dict.items():
            if key == "event":
                # The message itself is developer-authored; keep it whole and readable.
                redacted[key] = value
            elif is_sensitive_key(str(key)):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact(value, max_chars=self._max_chars)
        return redacted


def correlation_processor(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Stamp every event with the ambient correlation ids, without touching explicit ones."""
    for key, value in current_context().as_log_fields().items():
        event_dict.setdefault(key, value)
    return event_dict


def _build_processor_chain(settings: ObservabilitySettings) -> list[Processor]:
    """Shared processors, ordered so redaction runs before rendering - never after."""
    return [
        structlog.contextvars.merge_contextvars,
        correlation_processor,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        RedactionProcessor(max_chars=settings.max_logged_payload_chars),
    ]


def configure_logging(settings: ObservabilitySettings) -> None:
    """Install the logging configuration for the process. Idempotent.

    Called once by the composition root. Tests may call it again with different settings.
    """
    shared_processors = _build_processor_chain(settings)
    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy) through the identical chain so the process
    # emits exactly one log format.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # uvicorn installs its own handlers; strip them so nothing bypasses redaction.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(noisy)
        logger.handlers.clear()
        logger.propagate = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for a module. Always prefer this over ``logging.getLogger``."""
    return structlog.stdlib.get_logger(name)
