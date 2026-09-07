"""LLM adapter behaviour.

The LangChain adapter is the code that runs against a real provider, so it is exercised here
against a stub chat model rather than left untested until someone supplies a key. Its most
important behaviour - forwarding *unparseable* tool calls to the validation gate instead of
dropping them - is invisible on the happy path.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, SecretStr

from sightline.application.ports.llm_provider import LLMError, ToolSpec
from sightline.config.settings import LLMProviderName, LLMSettings
from sightline.infrastructure.llm.factory import MissingCredentialError, build_llm_provider
from sightline.infrastructure.llm.fake import FakeLLMProvider, synthesize_model
from sightline.infrastructure.llm.langchain import LangChainLLMProvider, _to_openai_tool
from sightline.tools.schemas.dataforseo import FetchOrganicSerpArgs
from sightline.tools.validation import ToolCallRejected, validate_tool_calls


class Answer(BaseModel):
    verdict: str = Field(min_length=3)
    confidence: float = Field(ge=0.0, le=1.0)


class StubRunnable:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def ainvoke(self, _messages: Any) -> Any:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class StubChatModel:
    """Implements only the two surfaces the adapter uses."""

    def __init__(self, *, structured: Any = None, tools: Any = None) -> None:
        self._structured = structured
        self._tools = tools

    def with_structured_output(self, _schema: Any, **_kwargs: Any) -> StubRunnable:
        return StubRunnable(self._structured)

    def bind_tools(self, _tools: Any) -> StubRunnable:
        return StubRunnable(self._tools)


def provider(model: Any) -> LangChainLLMProvider:
    return LangChainLLMProvider(model=model, provider_name="stub", model_name="stub-1")


class TestToolSchemaRendering:
    def test_a_tool_is_rendered_from_its_own_pydantic_schema(self) -> None:
        """The model is shown the same object the gate validates against."""
        spec = ToolSpec(
            name="fetch_organic_serp", description="d", args_schema=FetchOrganicSerpArgs
        )
        rendered = _to_openai_tool(spec)

        assert rendered["type"] == "function"
        assert rendered["function"]["name"] == "fetch_organic_serp"
        assert rendered["function"]["parameters"] == FetchOrganicSerpArgs.model_json_schema()
        assert rendered["function"]["parameters"]["additionalProperties"] is False


class TestStructuredCompletion:
    async def test_a_valid_response_is_returned_with_its_token_usage(self) -> None:
        raw = AIMessage(
            content="",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
        model = StubChatModel(
            structured={
                "parsed": Answer(verdict="visible", confidence=0.8),
                "raw": raw,
                "parsing_error": None,
            }
        )

        result = await provider(model).complete_structured(
            system_prompt="s", user_prompt="u", schema=Answer
        )

        assert result.value.verdict == "visible"
        assert result.usage.prompt_tokens == 120
        assert result.usage.total_tokens == 150

    async def test_missing_usage_metadata_does_not_break_the_call(self) -> None:
        model = StubChatModel(
            structured={
                "parsed": Answer(verdict="visible", confidence=0.5),
                "raw": AIMessage(content=""),
                "parsing_error": None,
            }
        )
        result = await provider(model).complete_structured(
            system_prompt="s", user_prompt="u", schema=Answer
        )
        assert result.usage.total_tokens == 0

    @pytest.mark.parametrize(
        "structured",
        [
            {"parsed": None, "raw": AIMessage(content=""), "parsing_error": "bad json"},
            {"parsed": None, "raw": AIMessage(content=""), "parsing_error": None},
            "not-a-mapping",
        ],
        ids=["parsing-error", "no-parsed-value", "unexpected-shape"],
    )
    async def test_an_unusable_response_raises_a_classified_llm_error(
        self, structured: Any
    ) -> None:
        with pytest.raises(LLMError):
            await provider(StubChatModel(structured=structured)).complete_structured(
                system_prompt="s", user_prompt="u", schema=Answer
            )

    async def test_a_provider_exception_is_wrapped_not_leaked(self) -> None:
        model = StubChatModel(structured=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="structured completion failed"):
            await provider(model).complete_structured(
                system_prompt="s", user_prompt="u", schema=Answer
            )


class TestToolCallProposal:
    @pytest.fixture
    def specs(self, tool_registry: Any) -> list[ToolSpec]:
        return list(tool_registry.specs())

    async def test_well_formed_tool_calls_are_returned(self, specs: list[ToolSpec]) -> None:
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_organic_serp", "args": {"keyword": "seo tools"}, "id": "c1"}
            ],
        )
        result = await provider(StubChatModel(tools=message)).propose_tool_calls(
            system_prompt="s", user_prompt="u", tools=specs
        )

        assert len(result.calls) == 1
        assert result.calls[0].tool_name == "fetch_organic_serp"
        assert result.calls[0].raw_arguments is None

    async def test_unparseable_tool_calls_are_forwarded_not_dropped(
        self, specs: list[ToolSpec], tool_registry: Any
    ) -> None:
        """The 'malformed arguments' case spec S3.3 requires be handled gracefully.

        Most integrations discard ``invalid_tool_calls`` silently, which makes a malformed
        call indistinguishable from the model choosing to call nothing.
        """
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_organic_serp", "args": {"keyword": "seo tools"}, "id": "ok"}
            ],
            invalid_tool_calls=[
                {
                    "name": "fetch_ai_overview",
                    "args": '{"keyword": ',
                    "id": "bad",
                    "error": "unterminated JSON",
                }
            ],
        )

        result = await provider(StubChatModel(tools=message)).propose_tool_calls(
            system_prompt="s", user_prompt="u", tools=specs
        )
        report = validate_tool_calls(result.calls, registry=tool_registry)

        assert len(result.calls) == 2
        assert len(report.accepted) == 1
        assert len(report.rejected) == 1
        rejection = report.rejected[0]
        assert isinstance(rejection, ToolCallRejected)
        assert rejection.reason.value == "malformed_arguments"

    async def test_calling_with_no_tools_is_rejected(self) -> None:
        with pytest.raises(LLMError, match="at least one tool"):
            await provider(StubChatModel(tools=AIMessage(content=""))).propose_tool_calls(
                system_prompt="s", user_prompt="u", tools=[]
            )

    async def test_a_non_message_response_raises(self, specs: list[ToolSpec]) -> None:
        with pytest.raises(LLMError):
            await provider(StubChatModel(tools="not-a-message")).propose_tool_calls(
                system_prompt="s", user_prompt="u", tools=specs
            )


class TestFakeProvider:
    async def test_synthesis_honours_every_declared_constraint(self) -> None:
        class Constrained(BaseModel):
            label: str = Field(min_length=12, max_length=20)
            count: int = Field(ge=5, le=9)
            ratio: float = Field(ge=0.0, le=1.0)
            items: list[str] = Field(min_length=2, max_length=4)

        value = synthesize_model(Constrained, hint="best seo tools", seed="s")

        assert 12 <= len(value.label) <= 20
        assert 5 <= value.count <= 9
        assert len(value.items) >= 2

    def test_container_bounds_do_not_constrain_item_content(self) -> None:
        """Regression for B10: list max_length is a batch size, not a string length."""

        class Batch(BaseModel):
            keywords: list[str] = Field(min_length=1, max_length=3)

        value = synthesize_model(Batch, hint="a fairly long keyword phrase here", seed="s")
        assert len(value.keywords[0]) > 3

    async def test_scripted_calls_are_consumed_in_order(self, tool_registry: Any) -> None:
        from sightline.application.ports.llm_provider import ProposedToolCall

        fake = FakeLLMProvider()
        fake.script_tool_calls(
            [ProposedToolCall(tool_name="fetch_organic_serp", arguments={"keyword": "a b"})],
            [ProposedToolCall(tool_name="nope", arguments={})],
        )

        first = await fake.propose_tool_calls(
            system_prompt="s", user_prompt="u", tools=tool_registry.specs()
        )
        second = await fake.propose_tool_calls(
            system_prompt="s", user_prompt="u", tools=tool_registry.specs()
        )

        assert first.calls[0].tool_name == "fetch_organic_serp"
        assert second.calls[0].tool_name == "nope"


class TestProviderFactory:
    def test_the_offline_provider_needs_no_credential(self) -> None:
        built = build_llm_provider(LLMSettings(provider=LLMProviderName.FAKE))
        assert built.name == "fake"

    @pytest.mark.parametrize(
        ("provider_name", "env_var"),
        [
            (LLMProviderName.OPENAI, "LLM_OPENAI_API_KEY"),
            (LLMProviderName.ANTHROPIC, "LLM_ANTHROPIC_API_KEY"),
            (LLMProviderName.GEMINI, "LLM_GOOGLE_API_KEY"),
        ],
    )
    def test_a_missing_credential_fails_at_startup_naming_the_variable(
        self, provider_name: LLMProviderName, env_var: str
    ) -> None:
        """Not on the first request an hour later."""
        with pytest.raises(MissingCredentialError, match=env_var):
            build_llm_provider(
                LLMSettings(
                    provider=provider_name,
                    openai_api_key=None,
                    anthropic_api_key=None,
                    google_api_key=None,
                )
            )

    def test_a_blank_credential_counts_as_missing(self) -> None:
        with pytest.raises(MissingCredentialError):
            build_llm_provider(
                LLMSettings(provider=LLMProviderName.OPENAI, openai_api_key=SecretStr("   "))
            )

    def test_a_configured_provider_is_built_and_reports_its_model(self) -> None:
        built = build_llm_provider(
            LLMSettings(
                provider=LLMProviderName.OPENAI,
                model="gpt-4o-mini",
                openai_api_key=SecretStr("sk-test-not-a-real-key"),
            )
        )
        assert built.name == "openai"
        assert built.model == "gpt-4o-mini"
