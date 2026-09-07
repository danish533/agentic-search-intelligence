"""Redaction, correlation propagation and metrics (spec S3.6)."""

from __future__ import annotations

import asyncio
import contextlib
import io
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from sightline.config.settings import ObservabilitySettings
from sightline.observability.correlation import (
    correlation_scope,
    current_context,
    current_correlation_id,
    new_correlation_id,
)
from sightline.observability.logging import (
    REDACTED,
    RedactionProcessor,
    configure_logging,
    get_logger,
    is_sensitive_key,
    redact,
)
from sightline.observability.metrics import (
    ApiCallRecord,
    NodeExecutionRecord,
    RunMetrics,
    current_metrics,
    metrics_scope,
)


class TestRedaction:
    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "DATAFORSEO_PASSWORD",
            "api_key",
            "openai_api_key",
            "apikey",
            "secret",
            "client_secret",
            "credentials",
            "authorization",
            "cookie",
            "token",
            "access_token",
            "refresh_token",
            "login",
            "private_key",
        ],
    )
    def test_credential_keys_are_redacted(self, key: str) -> None:
        assert is_sensitive_key(key)
        assert redact({key: "sensitive"}, max_chars=100) == {key: REDACTED}

    @pytest.mark.parametrize(
        "key",
        ["total_tokens", "prompt_tokens", "completion_tokens", "max_tokens", "token_count"],
    )
    def test_token_metrics_survive_redaction(self, key: str) -> None:
        """Regression for B3: substring-matching "token" would blind the S3.6 metrics."""
        assert not is_sensitive_key(key)
        assert redact({key: 1834}, max_chars=100) == {key: 1834}

    def test_values_that_look_like_credentials_are_redacted_regardless_of_key(self) -> None:
        for value in ("sk-abc123", "Bearer eyJhb", "Basic ZGFuaXNo"):
            assert redact({"harmless_name": value}, max_chars=100) == {"harmless_name": REDACTED}

    def test_secretstr_is_never_rendered(self) -> None:
        assert redact({"anything": SecretStr("hunter2")}, max_chars=100) == {"anything": REDACTED}

    def test_redaction_recurses_through_nested_structures(self) -> None:
        payload = {"headers": {"Authorization": "Basic x", "Accept": "application/json"}}
        assert redact(payload, max_chars=100) == {
            "headers": {"Authorization": REDACTED, "Accept": "application/json"}
        }

    def test_oversized_strings_are_truncated(self) -> None:
        result = redact({"body": "x" * 500}, max_chars=50)
        assert len(result["body"]) < 100
        assert result["body"].endswith("[truncated]")

    def test_deeply_nested_payloads_terminate(self) -> None:
        payload: dict[str, object] = {"a": {}}
        cursor = payload["a"]
        for _ in range(20):
            nxt: dict[str, object] = {}
            cursor["deeper"] = nxt  # type: ignore[index]
            cursor = nxt
        assert redact(payload, max_chars=100) is not None

    def test_the_processor_preserves_the_event_message(self) -> None:
        processor = RedactionProcessor(max_chars=2000)
        event = processor(None, "info", {"event": "node.completed", "password": "x", "n": 1})
        assert event["event"] == "node.completed"
        assert event["password"] == REDACTED
        assert event["n"] == 1


class TestEndToEndRedaction:
    """The full chain: logger -> processors -> renderer -> stream.

    Unit-testing the processor alone is not enough; this asserts that nothing sensitive
    survives all the way to the bytes that reach a log sink.
    """

    @pytest.fixture
    def captured(self) -> Callable[[dict[str, Any]], str]:
        """Configure logging onto a buffer and return whatever a log call emits.

        The buffer must be in place *before* ``configure_logging``: the handler binds to
        ``sys.stdout`` at call time, so redirecting afterwards captures nothing and the
        assertions below would pass vacuously.
        """

        def emit(fields: dict[str, Any]) -> str:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                configure_logging(ObservabilitySettings(log_format="json"))
                with correlation_scope(correlation_id="trace-1", node_name="retrieval"):
                    get_logger("test").info("external.call", **fields)
            output = buffer.getvalue()
            assert output.strip(), "nothing captured - the assertions would be vacuous"
            return output

        yield emit
        configure_logging(ObservabilitySettings(log_format="console", log_level="ERROR"))

    def test_no_credential_survives_to_the_log_stream(
        self, captured: Callable[[dict[str, Any]], str]
    ) -> None:
        output = captured(
            {
                "dataforseo_login": "danish@example.com",
                "dataforseo_password": "dfs-REALPASSWORD-987",
                "openai_api_key": SecretStr("sk-live-REALKEY123456789"),
                "request_headers": {"Authorization": "Basic ZGFuaXNoOnNlY3JldA=="},
                "stray_field": "sk-live-REALKEY123456789",
                "nested": {"credentials": {"login": "u", "password": "p"}},
            }
        )

        for secret in (
            "danish@example.com",
            "dfs-REALPASSWORD-987",
            "sk-live-REALKEY123456789",
            "ZGFuaXNoOnNlY3JldA==",
        ):
            assert secret not in output, f"{secret!r} leaked into the log stream"

    def test_a_username_style_field_is_redacted_too(
        self, captured: Callable[[dict[str, Any]], str]
    ) -> None:
        """`dataforseo_login` is the realistic field name, and a username is half a credential."""
        assert "danish@example.com" not in captured(
            {"dataforseo_login": "danish@example.com", "username": "danish"}
        )

    def test_operational_fields_survive_redaction(
        self, captured: Callable[[dict[str, Any]], str]
    ) -> None:
        output = captured(
            {"total_tokens": 2700, "prompt_tokens": 2100, "retry_count": 1, "duration_ms": 412.7}
        )

        assert '"total_tokens": 2700' in output
        assert '"prompt_tokens": 2100' in output
        assert '"retry_count": 1' in output
        assert '"node": "retrieval"' in output
        assert '"correlation_id": "trace-1"' in output


class TestCorrelation:
    def test_context_is_empty_outside_a_scope(self) -> None:
        assert current_context().as_log_fields() == {}

    def test_scope_binds_and_restores(self) -> None:
        with correlation_scope(correlation_id="abc", run_uuid="run-1"):
            assert current_correlation_id() == "abc"
        assert current_correlation_id() is None

    def test_an_inner_scope_inherits_the_outer_correlation_id(self) -> None:
        with correlation_scope(correlation_id="abc"), correlation_scope(node_name="retrieval"):
            fields = current_context().as_log_fields()
            assert fields == {"correlation_id": "abc", "node": "retrieval"}

    async def test_context_propagates_across_concurrent_fan_out(self) -> None:
        """This is what makes parallel retrieval traceable branch by branch."""

        async def branch(index: int) -> dict[str, str]:
            with correlation_scope(node_name=f"retrieval[{index}]"):
                await asyncio.sleep(0)
                return current_context().as_log_fields()

        with correlation_scope(correlation_id="trace-1", run_uuid="run-1"):
            results = await asyncio.gather(*(branch(i) for i in range(4)))

        assert {r["correlation_id"] for r in results} == {"trace-1"}
        assert {r["node"] for r in results} == {f"retrieval[{i}]" for i in range(4)}

    def test_generated_ids_are_unique_and_greppable(self) -> None:
        ids = {new_correlation_id() for _ in range(100)}
        assert len(ids) == 100
        assert all("-" not in identifier for identifier in ids)


class TestMetrics:
    def test_node_latency_is_summed_per_node_across_fan_out_branches(self) -> None:
        metrics = RunMetrics()
        for duration in (10.0, 20.0, 30.0):
            metrics.record_node(
                NodeExecutionRecord(node_name="retrieval", duration_ms=duration, succeeded=True)
            )
        assert metrics.node_latency_ms() == {"retrieval": 60.0}

    def test_success_rates_and_failed_nodes_are_reported(self) -> None:
        metrics = RunMetrics()
        metrics.record_node(NodeExecutionRecord("a", 1.0, succeeded=True))
        metrics.record_node(NodeExecutionRecord("b", 1.0, succeeded=False))
        assert metrics.node_success_rate() == 0.5
        assert metrics.failed_nodes() == ["b"]

    def test_retries_are_counted_once_not_double_counted(self) -> None:
        """Regression: a node's retry_count attributes the *same* retries its calls consumed.

        Summing both reported every retry twice - a run with two real retries claimed four.
        """
        metrics = RunMetrics()
        metrics.record_node(NodeExecutionRecord("retrieval", 1.0, succeeded=True, retry_count=2))
        metrics.record_api_call(ApiCallRecord("/x", 1.0, succeeded=True, attempts=3))

        assert metrics.total_retries == 2
        assert metrics.node_retry_counts() == {"retrieval": 2}

    def test_summary_carries_everything_the_spec_asks_for(self) -> None:
        metrics = RunMetrics()
        metrics.record_node(NodeExecutionRecord("a", 5.0, succeeded=True))
        metrics.record_api_call(ApiCallRecord("/x", 2.0, succeeded=True))
        metrics.record_tokens(prompt_tokens=100, completion_tokens=50)

        summary = metrics.summary()
        for field in (
            "node_latency_ms",
            "node_success_rate",
            "api_call_count",
            "api_success_rate",
            "total_retries",
            "total_tokens",
        ):
            assert field in summary
        assert summary["total_tokens"] == 150

    def test_empty_metrics_do_not_divide_by_zero(self) -> None:
        assert RunMetrics().summary()["node_success_rate"] == 0.0

    def test_the_active_collector_is_scoped(self) -> None:
        assert current_metrics() is None
        metrics = RunMetrics()
        with metrics_scope(metrics):
            assert current_metrics() is metrics
        assert current_metrics() is None
