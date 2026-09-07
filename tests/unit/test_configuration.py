"""Configuration and credential handling.

The guards here run at process start. Their job is to turn a mis-pasted credential into a
named startup error instead of an opaque 401 twenty seconds into a pipeline run.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from sightline.composition.container import (
    MissingDataForSEOCredentialsError,
    _build_transport,
)
from sightline.config.settings import (
    DatabaseSettings,
    DataForSEOMode,
    DataForSEOSettings,
    LLMProviderName,
    LLMSettings,
    Settings,
    credential_is_present,
)
from sightline.infrastructure.dataforseo.mock.transport import MockTransport
from sightline.infrastructure.dataforseo.transport import HttpxTransport
from sightline.infrastructure.llm.factory import MissingCredentialError, build_llm_provider


class TestCredentialPresence:
    @pytest.mark.parametrize("value", [None, "", "   ", "\t", "\n  "])
    def test_absent_blank_and_whitespace_only_credentials_are_all_missing(
        self, value: str | None
    ) -> None:
        """``bool(SecretStr("   "))`` is True, so truthiness alone is not enough.

        A stray space survives a copy-paste far more often than an empty field does.
        """
        secret = None if value is None else SecretStr(value)
        assert credential_is_present(secret) is False

    @pytest.mark.parametrize("value", ["sk-abc", "user@example.com", " padded "])
    def test_a_real_credential_is_present(self, value: str) -> None:
        assert credential_is_present(SecretStr(value)) is True


class TestDataForSEOModeGuards:
    def test_mock_mode_needs_no_credentials(self) -> None:
        transport = _build_transport(
            Settings(dataforseo=DataForSEOSettings(mode=DataForSEOMode.MOCK))
        )
        assert isinstance(transport, MockTransport)

    @pytest.mark.parametrize("mode", [DataForSEOMode.LIVE, DataForSEOMode.SANDBOX])
    @pytest.mark.parametrize(
        ("login", "password"),
        [("", ""), ("   ", "   "), ("real@example.com", ""), ("", "realpass")],
        ids=["both-blank", "both-whitespace", "password-blank", "login-blank"],
    )
    def test_a_networked_mode_without_usable_credentials_fails_at_startup(
        self, mode: DataForSEOMode, login: str, password: str
    ) -> None:
        settings = Settings(
            dataforseo=DataForSEOSettings(
                mode=mode, login=SecretStr(login), password=SecretStr(password)
            )
        )
        with pytest.raises(MissingDataForSEOCredentialsError, match="DATAFORSEO_LOGIN"):
            _build_transport(settings)

    def test_real_credentials_build_the_live_transport(self) -> None:
        settings = Settings(
            dataforseo=DataForSEOSettings(
                mode=DataForSEOMode.LIVE,
                login=SecretStr("real@example.com"),
                password=SecretStr("realpass"),
            )
        )
        transport = _build_transport(settings)

        assert isinstance(transport, HttpxTransport)
        assert transport.is_mock is False
        assert transport.mode_label == "live"


class TestLLMProviderGuards:
    @pytest.mark.parametrize("key", ["", "   "])
    def test_a_blank_or_whitespace_key_fails_at_startup(self, key: str) -> None:
        with pytest.raises(MissingCredentialError, match="LLM_OPENAI_API_KEY"):
            build_llm_provider(
                LLMSettings(provider=LLMProviderName.OPENAI, openai_api_key=SecretStr(key))
            )


class TestSecretsAreNeverRendered:
    """A settings object reaches logs and tracebacks; it must not carry a readable secret."""

    @pytest.fixture
    def loaded(self) -> Settings:
        return Settings(
            llm=LLMSettings(openai_api_key=SecretStr("sk-live-REALKEY123456789")),
            dataforseo=DataForSEOSettings(
                login=SecretStr("danish@example.com"),
                password=SecretStr("dfs-REALPASSWORD-987"),
            ),
        )

    @pytest.mark.parametrize(
        "secret", ["sk-live-REALKEY123456789", "dfs-REALPASSWORD-987", "danish@example.com"]
    )
    def test_no_secret_appears_in_repr_str_or_serialisation(
        self, loaded: Settings, secret: str
    ) -> None:
        assert secret not in repr(loaded)
        assert secret not in str(loaded)
        assert secret not in str(loaded.model_dump())
        assert secret not in loaded.model_dump_json()

    def test_a_secret_is_only_readable_through_an_explicit_call(self, loaded: Settings) -> None:
        assert loaded.llm.openai_api_key is not None
        assert loaded.llm.openai_api_key.get_secret_value() == "sk-live-REALKEY123456789"


class TestOfflineDemoPlannerHonesty:
    """The offline demo plan must not claim a failure that did not happen.

    ``plan_responder`` reuses the fallback template, which is fine - but it used to inherit the
    fallback's *reason* too, so a successful planner run logged "the Query Planner was
    unavailable". The fallback node never executed on those runs. Misleading logs are a defect
    in their own right when observability is what is being graded.
    """

    @pytest.fixture
    def planner_prompt(self) -> str:
        from sightline.agents.prompts import query_planner
        from sightline.domain.entities.profile import Profile

        profile = Profile(
            name="Surfer SEO", domain="surferseo.com", industry="SEO Software", description=""
        )
        return query_planner.build_user_prompt(profile, "How visible is Surfer SEO?", 6)

    def test_the_demo_plan_does_not_claim_the_planner_failed(self, planner_prompt: str) -> None:
        from sightline.composition.demo_responders import plan_responder

        interpretation = plan_responder("", planner_prompt).interpretation

        assert "unavailable" not in interpretation.lower()
        assert "fake" in interpretation.lower() or "offline" in interpretation.lower()

    def test_the_genuine_fallback_still_states_why_it_ran(self) -> None:
        from sightline.agents.nodes.planner_fallback import build_fallback_plan
        from sightline.domain.entities.profile import Profile

        profile = Profile(
            name="Surfer SEO", domain="surferseo.com", industry="SEO Software", description=""
        )
        plan = build_fallback_plan(profile, limit=6)

        assert "unavailable" in plan.interpretation.lower()
        assert plan.is_usable


class TestDestructiveFixtureGuard:
    """The integration fixtures TRUNCATE before every test.

    Convention #27 says fixtures must not inherit the ambient environment for anything
    external. That was applied to the LLM and to DataForSEO but originally not to the database
    - the one dependency whose failure mode destroys data rather than costing money. These
    tests pin the guard that closes it.
    """

    @staticmethod
    def _settings(url: str) -> DatabaseSettings:
        return DatabaseSettings(url=url)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql+asyncpg://u:p@staging.internal:5432/app_production",
            "postgresql+asyncpg://u:p@10.0.4.22:5432/sightline",
            "postgresql+asyncpg://u:p@db.example.com:5432/sightline",
            "postgresql+asyncpg://u:p@localhost:5432/customer_data",
            "postgresql+asyncpg://u:p@localhost:5432/analytics",
            "postgresql+asyncpg://u:p@localhost:5432/sightline,prod.internal:5432/sightline",
        ],
        ids=[
            "remote-host",
            "remote-ip",
            "remote-fqdn",
            "local-but-unknown-db",
            "local-but-shared-db",
            "one-safe-host-one-production-host",
        ],
    )
    def test_a_non_disposable_target_stops_the_session(self, url: str) -> None:
        from tests.conftest import assert_database_is_disposable

        with pytest.raises(Exception) as caught:
            assert_database_is_disposable(self._settings(url))

        assert "REFUSING TO RUN" in str(caught.value)

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql+asyncpg://u:p@localhost:5432/sightline",
            "postgresql+asyncpg://u:p@localhost:5433/sightline_test",
            "postgresql+asyncpg://u:p@127.0.0.1:5432/anything_test",
            "postgresql+asyncpg://u:p@db:5432/sightline",
            "postgresql+asyncpg://u:p@db:5432/ci_test",
        ],
        ids=["default", "named-test-db", "any-_test-suffix", "compose-host", "ci-database"],
    )
    def test_a_disposable_target_is_allowed(self, url: str) -> None:
        from tests.conftest import assert_database_is_disposable

        # Passes by not raising; pytest.exit would abort the session.
        assert_database_is_disposable(self._settings(url))

    def test_the_message_says_what_is_allowed_and_how_to_fix_it(self) -> None:
        from tests.conftest import assert_database_is_disposable

        with pytest.raises(Exception) as caught:
            assert_database_is_disposable(
                self._settings("postgresql+asyncpg://u:p@prod.internal:5432/app")
            )

        message = str(caught.value)
        assert "Allowed hosts" in message
        assert "Allowed databases" in message
        assert "make db" in message
