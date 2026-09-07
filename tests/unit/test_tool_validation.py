"""Tool-call argument validation.

**Spec S5 mandated test #3: "tool-call argument validation".**

Covers every way a model can get a tool call wrong, and asserts that each is rejected with an
actionable reason rather than crashing or reaching the network.
"""

from __future__ import annotations

import pytest

from sightline.application.ports.llm_provider import ProposedToolCall
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.registry import DuplicateToolError, ToolRegistry, UnknownToolError
from sightline.tools.schemas.dataforseo import FetchOrganicSerpArgs
from sightline.tools.validation import (
    RejectionReason,
    ToolCallRejected,
    ValidatedToolCall,
    validate_tool_call,
    validate_tool_calls,
)


def call(tool: str, args: dict[str, object] | None = None, **kwargs: object) -> ProposedToolCall:
    return ProposedToolCall(tool_name=tool, arguments=args or {}, call_id="c1", **kwargs)  # type: ignore[arg-type]


class TestAcceptsWellFormedCalls:
    def test_valid_call_is_accepted_and_parsed(self, tool_registry: ToolRegistry) -> None:
        result = validate_tool_call(
            call("fetch_organic_serp", {"keyword": "best seo tools"}), registry=tool_registry
        )

        assert isinstance(result, ValidatedToolCall)
        assert isinstance(result.arguments, FetchOrganicSerpArgs)
        assert result.arguments.keyword == "best seo tools"

    def test_optional_arguments_receive_their_documented_defaults(
        self, tool_registry: ToolRegistry
    ) -> None:
        result = validate_tool_call(
            call("fetch_organic_serp", {"keyword": "seo tools"}), registry=tool_registry
        )

        assert isinstance(result, ValidatedToolCall)
        assert result.arguments.model_dump() == {
            "keyword": "seo tools",
            "location_code": 2840,
            "language_code": "en",
            "depth": 10,
        }

    def test_duplicate_keywords_are_deduplicated_case_insensitively(
        self, tool_registry: ToolRegistry
    ) -> None:
        result = validate_tool_call(
            call("fetch_keyword_metrics", {"keywords": ["seo tools", "SEO Tools", "ai seo"]}),
            registry=tool_registry,
        )

        assert isinstance(result, ValidatedToolCall)
        assert result.arguments.model_dump()["keywords"] == ["seo tools", "ai seo"]


class TestRejectsMalformedCalls:
    def test_unknown_tool_name_is_rejected_with_the_available_tools(
        self, tool_registry: ToolRegistry
    ) -> None:
        result = validate_tool_call(
            call("fetch_google_rankings", {"keyword": "x"}), registry=tool_registry
        )

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.UNKNOWN_TOOL
        assert "fetch_organic_serp" in result.repair_hint()

    def test_missing_required_field_is_rejected_and_names_the_field(
        self, tool_registry: ToolRegistry
    ) -> None:
        result = validate_tool_call(
            call("fetch_organic_serp", {"location_code": 2840}), registry=tool_registry
        )

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.MISSING_REQUIRED_FIELD
        assert any(issue.field == "keyword" for issue in result.issues)

    def test_hallucinated_extra_field_is_rejected_not_silently_dropped(
        self, tool_registry: ToolRegistry
    ) -> None:
        result = validate_tool_call(
            call("fetch_organic_serp", {"keyword": "seo", "country": "USA"}),
            registry=tool_registry,
        )

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.UNEXPECTED_FIELD
        assert any(issue.field == "country" for issue in result.issues)

    @pytest.mark.parametrize(
        ("arguments", "expected_field"),
        [
            ({"keyword": "seo", "depth": "a lot"}, "depth"),
            ({"keyword": "seo", "depth": 500}, "depth"),
            ({"keyword": "s"}, "keyword"),
            ({"keyword": "site:surferseo.com pricing"}, "keyword"),
        ],
        ids=["wrong-type", "above-maximum", "below-min-length", "search-operator"],
    )
    def test_invalid_field_values_are_rejected(
        self,
        tool_registry: ToolRegistry,
        arguments: dict[str, object],
        expected_field: str,
    ) -> None:
        result = validate_tool_call(call("fetch_organic_serp", arguments), registry=tool_registry)

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.INVALID_FIELD_VALUE
        assert any(issue.field == expected_field for issue in result.issues)

    def test_invalid_enum_value_is_rejected(self, tool_registry: ToolRegistry) -> None:
        result = validate_tool_call(
            call("fetch_llm_visibility", {"prompt": "best seo tool?", "llm_name": "claude"}),
            registry=tool_registry,
        )

        assert isinstance(result, ToolCallRejected)
        assert any(issue.field == "llm_name" for issue in result.issues)

    def test_undecodable_argument_payload_is_reported_as_malformed(
        self, tool_registry: ToolRegistry
    ) -> None:
        """A provider that could not parse the model's arguments forwards the raw text.

        Reported as *malformed payload* rather than as missing fields, so the log says what
        actually went wrong.
        """
        result = validate_tool_call(
            call("fetch_organic_serp", {}, raw_arguments='{"keyword": '),
            registry=tool_registry,
        )

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.MALFORMED_ARGUMENTS
        assert "well-formed JSON object" in result.repair_hint()

    def test_empty_arguments_are_rejected_not_defaulted(self, tool_registry: ToolRegistry) -> None:
        result = validate_tool_call(call("fetch_ai_overview", {}), registry=tool_registry)

        assert isinstance(result, ToolCallRejected)
        assert result.reason is RejectionReason.MISSING_REQUIRED_FIELD


class TestBatchValidation:
    def test_one_bad_call_does_not_discard_the_good_ones(self, tool_registry: ToolRegistry) -> None:
        """A model that proposes four calls and gets one wrong keeps the other three."""
        report = validate_tool_calls(
            [
                call("fetch_organic_serp", {"keyword": "best seo tools"}),
                call("does_not_exist", {"keyword": "x"}),
                call("fetch_ai_overview", {"keyword": "best seo tools"}),
                call("fetch_organic_serp", {"keyword": "seo", "depth": 999}),
            ],
            registry=tool_registry,
        )

        assert [accepted.tool_name for accepted in report.accepted] == [
            "fetch_organic_serp",
            "fetch_ai_overview",
        ]
        assert len(report.rejected) == 2
        assert report.has_accepted
        assert report.total == 4

    def test_every_rejection_carries_a_repair_hint(self, tool_registry: ToolRegistry) -> None:
        report = validate_tool_calls(
            [call("nope", {}), call("fetch_organic_serp", {})], registry=tool_registry
        )

        hints = report.repair_hints()
        assert len(hints) == 2
        assert all(hint.endswith("instead.") or "Re-issue" in hint for hint in hints)


class TestRegistry:
    def test_registry_exposes_one_tool_per_logical_api_call(
        self, tool_registry: ToolRegistry
    ) -> None:
        """Spec S3.4: one tool per logical call, not one giant 'call DataForSEO' tool."""
        assert len(tool_registry) == 4
        assert {tool.kind for tool in tool_registry} == set(RetrievalKind)

    def test_subset_narrows_the_tools_offered_to_the_model(
        self, tool_registry: ToolRegistry
    ) -> None:
        subset = tool_registry.subset([RetrievalKind.ORGANIC_SERP, RetrievalKind.KEYWORD_METRICS])

        assert set(subset.names) == {"fetch_organic_serp", "fetch_keyword_metrics"}
        rejected = validate_tool_call(
            call("fetch_llm_visibility", {"prompt": "anything at all"}), registry=subset
        )
        assert isinstance(rejected, ToolCallRejected)
        assert rejected.reason is RejectionReason.UNKNOWN_TOOL

    def test_duplicate_tool_names_fail_at_construction(self, tool_registry: ToolRegistry) -> None:
        """A wiring bug, not a runtime edge case: it must not resolve arbitrarily."""
        tools = list(tool_registry)
        with pytest.raises(DuplicateToolError):
            ToolRegistry([*tools, tools[0]])

    def test_require_raises_for_an_unknown_name(self, tool_registry: ToolRegistry) -> None:
        with pytest.raises(UnknownToolError):
            tool_registry.require("nope")

    def test_tool_specs_expose_the_schema_the_gate_enforces(
        self, tool_registry: ToolRegistry
    ) -> None:
        """The schema shown to the model and the one validated against must be one object."""
        for spec in tool_registry.specs():
            tool = tool_registry.require(spec.name)
            assert spec.args_schema is tool.args_schema
            assert spec.args_schema.model_json_schema()["additionalProperties"] is False
