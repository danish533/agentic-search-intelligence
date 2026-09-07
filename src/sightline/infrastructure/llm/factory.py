"""LLM provider factory.

The only vendor-aware code in the system. Everything else - every agent, every node - depends
on the ``LLMProvider`` port, so switching between OpenAI, Anthropic, Gemini and the offline
fake is an environment variable, not an edit.

Vendor SDK imports are deliberately deferred into their branches. The default mode is ``fake``,
and there is no reason for an offline test run to pay the import cost of three cloud SDKs it
will never construct.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from sightline.application.ports.llm_provider import LLMProvider
from sightline.config.settings import LLMProviderName, LLMSettings, credential_is_present
from sightline.infrastructure.llm.fake import FakeLLMProvider
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


class MissingCredentialError(RuntimeError):
    """A provider was selected without the credential it requires.

    Raised at startup rather than on the first LLM call, so a misconfiguration surfaces when
    the process boots instead of halfway through a pipeline run.
    """

    def __init__(self, provider: LLMProviderName, env_var: str) -> None:
        super().__init__(
            f"LLM_PROVIDER={provider.value} requires {env_var} to be set. "
            f"Set it in .env, or use LLM_PROVIDER=fake to run fully offline."
        )


def _require(secret: SecretStr | None, provider: LLMProviderName, env_var: str) -> str:
    # `secret is None` is checked explicitly rather than narrowed by an assertion: an
    # assertion is stripped under `python -O`, and this guard protects a startup invariant.
    if secret is None or not credential_is_present(secret):
        raise MissingCredentialError(provider, env_var)
    return secret.get_secret_value()


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    """Construct the configured provider. Called once, by the composition root."""
    provider = settings.provider

    if provider is LLMProviderName.FAKE:
        _logger.info("llm.provider.selected", provider="fake", model="deterministic-stub")
        return FakeLLMProvider()

    from sightline.infrastructure.llm.langchain import LangChainLLMProvider

    # Annotated to the abstract base: each branch builds a different concrete chat
    # model, and the adapter below depends only on the interface they share.
    model: BaseChatModel

    match provider:
        case LLMProviderName.OPENAI:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=settings.model,
                temperature=settings.temperature,
                max_completion_tokens=settings.max_tokens,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
                api_key=SecretStr(
                    _require(settings.openai_api_key, provider, "LLM_OPENAI_API_KEY")
                ),
            )

        case LLMProviderName.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model=settings.model,
                temperature=settings.temperature,
                max_tokens_to_sample=settings.max_tokens,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
                stop=None,
                api_key=SecretStr(
                    _require(settings.anthropic_api_key, provider, "LLM_ANTHROPIC_API_KEY")
                ),
            )

        case LLMProviderName.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=settings.model,
                temperature=settings.temperature,
                max_output_tokens=settings.max_tokens,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
                google_api_key=SecretStr(
                    _require(settings.google_api_key, provider, "LLM_GOOGLE_API_KEY")
                ),
            )

        case _:  # pragma: no cover - StrEnum makes this unreachable
            raise ValueError(f"Unsupported LLM provider: {provider}")

    _logger.info("llm.provider.selected", provider=provider.value, model=settings.model)
    return LangChainLLMProvider(
        model=model,
        provider_name=provider.value,
        model_name=settings.model,
    )
