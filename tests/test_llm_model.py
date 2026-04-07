"""
Unit tests for the multi-provider LLM factory in llm_modules/llm_model.py.

All tests run without a real API key — constructors are mocked and
os.environ is patched where needed.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

# Add the src directory to the path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from black_langcube.llm_modules.llm_model import (  # noqa: E402
    DEFAULT_PROVIDER,
    GEMINI_API_KEY,
    MISTRAL_API_KEY,
    MODEL_REGISTRY,
    OPENAI_API_KEY,
    LLMProvider,
    ModelTier,
    _load_secret,
    _resolve_provider,
    create_llm,
    default_llm,
    get_llm_high,
    get_llm_low,
    llm_analyst,
    llm_check_title,
    llm_outline,
    llm_text,
    llm_title_abstract,
)
import black_langcube.llm_modules.llm_model as llm_model_module  # noqa: E402


@pytest.mark.unit
class TestLLMProviderEnum(unittest.TestCase):
    """LLMProvider is a string enum with the expected members."""

    def test_provider_values(self):
        self.assertEqual(LLMProvider.OPENAI.value, "openai")
        self.assertEqual(LLMProvider.GEMINI.value, "gemini")
        self.assertEqual(LLMProvider.MISTRAL.value, "mistral")

    def test_provider_is_string(self):
        self.assertIsInstance(LLMProvider.OPENAI, str)
        self.assertIsInstance(LLMProvider.GEMINI, str)

    def test_provider_from_string(self):
        self.assertEqual(LLMProvider("openai"), LLMProvider.OPENAI)
        self.assertEqual(LLMProvider("gemini"), LLMProvider.GEMINI)


@pytest.mark.unit
class TestModelTierEnum(unittest.TestCase):
    """ModelTier is a string enum covering all pipeline steps."""

    def test_tier_values(self):
        self.assertEqual(ModelTier.LOW.value, "low")
        self.assertEqual(ModelTier.HIGH.value, "high")
        self.assertEqual(ModelTier.ANALYST.value, "analyst")
        self.assertEqual(ModelTier.OUTLINE.value, "outline")
        self.assertEqual(ModelTier.TEXT.value, "text")
        self.assertEqual(ModelTier.CHECK_TITLE.value, "check_title")
        self.assertEqual(ModelTier.TITLE_ABSTRACT.value, "title_abstract")

    def test_tier_is_string(self):
        for tier in ModelTier:
            self.assertIsInstance(tier, str)


@pytest.mark.unit
class TestModelRegistry(unittest.TestCase):
    """MODEL_REGISTRY covers every (provider, tier) combination."""

    def test_registry_has_all_providers_and_tiers(self):
        for provider in LLMProvider:
            self.assertIn(provider, MODEL_REGISTRY)
            for tier in ModelTier:
                self.assertIn(tier, MODEL_REGISTRY[provider])
                self.assertIsInstance(MODEL_REGISTRY[provider][tier], str)
                self.assertTrue(MODEL_REGISTRY[provider][tier])  # non-empty

    def test_openai_low_default(self):
        expected = os.getenv("OPENAI_MODEL_LOW", "gpt-4o-mini")
        self.assertEqual(MODEL_REGISTRY[LLMProvider.OPENAI][ModelTier.LOW], expected)

    def test_gemini_low_default(self):
        expected = os.getenv("GEMINI_MODEL_LOW", "gemini-2.5-flash")
        self.assertEqual(MODEL_REGISTRY[LLMProvider.GEMINI][ModelTier.LOW], expected)

    def test_gemini_high_default(self):
        expected = os.getenv("GEMINI_MODEL_HIGH", "gemini-2.5-pro")
        self.assertEqual(MODEL_REGISTRY[LLMProvider.GEMINI][ModelTier.HIGH], expected)


@pytest.mark.unit
class TestLoadSecret(unittest.TestCase):
    """_load_secret() wraps env vars in SecretStr or returns None."""

    _ABSENT_VAR = "DEFINITELY_NOT_SET_84f7a3b2_BLCTEST"

    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {self._ABSENT_VAR: ""}, clear=False):
            # ensure the var is absent (not just empty)
            os.environ.pop(self._ABSENT_VAR, None)
            result = _load_secret(self._ABSENT_VAR)
        self.assertIsNone(result)

    def test_returns_secretstr_when_set(self):
        with patch.dict(os.environ, {"BLC_TEST_KEY": "my-test-key"}):
            result = _load_secret("BLC_TEST_KEY")
        self.assertIsInstance(result, SecretStr)
        self.assertEqual(result.get_secret_value(), "my-test-key")

    def test_returns_none_for_empty_string(self):
        with patch.dict(os.environ, {"BLC_TEST_KEY": ""}):
            result = _load_secret("BLC_TEST_KEY")
        self.assertIsNone(result)


@pytest.mark.unit
class TestSecretStrLeakPrevention(unittest.TestCase):
    """SecretStr does not expose the raw value in repr / str / f-strings."""

    def _make_secret(self) -> SecretStr:
        return SecretStr("super-secret-api-key-12345")

    def test_repr_does_not_contain_value(self):
        secret = self._make_secret()
        self.assertNotIn("super-secret-api-key-12345", repr(secret))

    def test_str_does_not_contain_value(self):
        secret = self._make_secret()
        self.assertNotIn("super-secret-api-key-12345", str(secret))

    def test_fstring_does_not_contain_value(self):
        secret = self._make_secret()
        self.assertNotIn("super-secret-api-key-12345", f"{secret}")

    def test_get_secret_value_returns_raw(self):
        secret = self._make_secret()
        self.assertEqual(secret.get_secret_value(), "super-secret-api-key-12345")

    def test_module_openai_key_is_secretstr_or_none(self):
        """OPENAI_API_KEY is either a SecretStr or None — never a plain str."""
        self.assertIsInstance(OPENAI_API_KEY, (SecretStr, type(None)))

    def test_module_gemini_key_is_secretstr_or_none(self):
        self.assertIsInstance(GEMINI_API_KEY, (SecretStr, type(None)))

    def test_module_mistral_key_is_secretstr_or_none(self):
        self.assertIsInstance(MISTRAL_API_KEY, (SecretStr, type(None)))


@pytest.mark.unit
class TestCreateLLMOpenAI(unittest.TestCase):
    """create_llm() dispatches to ChatOpenAI with the correct model name."""

    def _call_and_get_kwargs(self, tier: ModelTier) -> dict:
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            create_llm(LLMProvider.OPENAI, tier)
            return mock_cls.call_args.kwargs

    def test_openai_low_uses_correct_model(self):
        kwargs = self._call_and_get_kwargs(ModelTier.LOW)
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.OPENAI][ModelTier.LOW]
        )

    def test_openai_high_uses_correct_model(self):
        kwargs = self._call_and_get_kwargs(ModelTier.HIGH)
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.OPENAI][ModelTier.HIGH]
        )

    def test_openai_analyst_uses_correct_model(self):
        kwargs = self._call_and_get_kwargs(ModelTier.ANALYST)
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.OPENAI][ModelTier.ANALYST]
        )

    def test_openai_passes_api_key(self):
        """When OPENAI_API_KEY is set, its value is forwarded to ChatOpenAI."""
        secret = SecretStr("test-openai-key")
        with patch.object(llm_model_module, "OPENAI_API_KEY", secret):
            with patch("langchain_openai.ChatOpenAI") as mock_cls:
                create_llm(LLMProvider.OPENAI, ModelTier.LOW)
                kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "test-openai-key")

    def test_openai_omits_api_key_when_none(self):
        with patch.object(llm_model_module, "OPENAI_API_KEY", None):
            with patch("langchain_openai.ChatOpenAI") as mock_cls:
                create_llm(LLMProvider.OPENAI, ModelTier.LOW)
                kwargs = mock_cls.call_args.kwargs
        self.assertNotIn("api_key", kwargs)


@pytest.mark.unit
class TestCreateLLMGemini(unittest.TestCase):
    """create_llm() dispatches to ChatGoogleGenerativeAI with correct model name."""

    def _mock_gemini_module(self) -> MagicMock:
        mock_mod = MagicMock()
        return mock_mod

    def test_gemini_low_uses_correct_model(self):
        mock_mod = self._mock_gemini_module()
        with patch.dict("sys.modules", {"langchain_google_genai": mock_mod}):
            create_llm(LLMProvider.GEMINI, ModelTier.LOW)
            kwargs = mock_mod.ChatGoogleGenerativeAI.call_args.kwargs
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.GEMINI][ModelTier.LOW]
        )

    def test_gemini_high_uses_correct_model(self):
        mock_mod = self._mock_gemini_module()
        with patch.dict("sys.modules", {"langchain_google_genai": mock_mod}):
            create_llm(LLMProvider.GEMINI, ModelTier.HIGH)
            kwargs = mock_mod.ChatGoogleGenerativeAI.call_args.kwargs
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.GEMINI][ModelTier.HIGH]
        )

    def test_gemini_passes_api_key(self):
        secret = SecretStr("test-gemini-key")
        mock_mod = self._mock_gemini_module()
        with patch.object(llm_model_module, "GEMINI_API_KEY", secret):
            with patch.dict("sys.modules", {"langchain_google_genai": mock_mod}):
                create_llm(LLMProvider.GEMINI, ModelTier.LOW)
                kwargs = mock_mod.ChatGoogleGenerativeAI.call_args.kwargs
        self.assertEqual(kwargs["google_api_key"], "test-gemini-key")

    def test_gemini_omits_google_api_key_when_none(self):
        mock_mod = self._mock_gemini_module()
        with patch.object(llm_model_module, "GEMINI_API_KEY", None):
            with patch.dict("sys.modules", {"langchain_google_genai": mock_mod}):
                create_llm(LLMProvider.GEMINI, ModelTier.LOW)
                kwargs = mock_mod.ChatGoogleGenerativeAI.call_args.kwargs
        self.assertNotIn("google_api_key", kwargs)


@pytest.mark.unit
class TestCreateLLMUnknownProvider(unittest.TestCase):
    """create_llm() raises ValueError for unsupported providers."""

    def test_invalid_string_raises(self):
        """Non-enum values cause a ValueError or KeyError before factory dispatch."""
        with self.assertRaises((ValueError, KeyError)):
            create_llm("unknown_provider", ModelTier.LOW)  # type: ignore[arg-type]


@pytest.mark.unit
class TestCreateLLMMistral(unittest.TestCase):
    """create_llm() dispatches to ChatMistralAI with the correct model name."""

    def _mock_mistral_module(self) -> MagicMock:
        return MagicMock()

    def test_mistral_low_uses_correct_model(self):
        mock_mod = self._mock_mistral_module()
        with patch.dict("sys.modules", {"langchain_mistralai": mock_mod}):
            create_llm(LLMProvider.MISTRAL, ModelTier.LOW)
            kwargs = mock_mod.ChatMistralAI.call_args.kwargs
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.MISTRAL][ModelTier.LOW]
        )

    def test_mistral_high_uses_correct_model(self):
        mock_mod = self._mock_mistral_module()
        with patch.dict("sys.modules", {"langchain_mistralai": mock_mod}):
            create_llm(LLMProvider.MISTRAL, ModelTier.HIGH)
            kwargs = mock_mod.ChatMistralAI.call_args.kwargs
        self.assertEqual(
            kwargs["model"], MODEL_REGISTRY[LLMProvider.MISTRAL][ModelTier.HIGH]
        )

    def test_mistral_passes_api_key(self):
        secret = SecretStr("test-mistral-key")
        mock_mod = self._mock_mistral_module()
        with patch.object(llm_model_module, "MISTRAL_API_KEY", secret):
            with patch.dict("sys.modules", {"langchain_mistralai": mock_mod}):
                create_llm(LLMProvider.MISTRAL, ModelTier.LOW)
                kwargs = mock_mod.ChatMistralAI.call_args.kwargs
        self.assertEqual(kwargs["mistral_api_key"], "test-mistral-key")

    def test_mistral_omits_api_key_when_none(self):
        mock_mod = self._mock_mistral_module()
        with patch.object(llm_model_module, "MISTRAL_API_KEY", None):
            with patch.dict("sys.modules", {"langchain_mistralai": mock_mod}):
                create_llm(LLMProvider.MISTRAL, ModelTier.LOW)
                kwargs = mock_mod.ChatMistralAI.call_args.kwargs
        self.assertNotIn("mistral_api_key", kwargs)


@pytest.mark.unit
class TestCreateLLMAllTiers(unittest.TestCase):
    """create_llm() dispatches without error for every (provider, tier) combination."""

    def test_all_provider_tier_combinations_dispatch(self):
        """No (provider, tier) pair raises an exception."""
        openai_mock = MagicMock()
        gemini_mock = MagicMock()
        mistral_mock = MagicMock()
        sys_modules_patch = {
            "langchain_openai": openai_mock,
            "langchain_google_genai": gemini_mock,
            "langchain_mistralai": mistral_mock,
        }
        with patch.dict("sys.modules", sys_modules_patch):
            for provider in LLMProvider:
                for tier in ModelTier:
                    with self.subTest(provider=provider, tier=tier):
                        create_llm(provider, tier)  # must not raise

    def test_all_combinations_use_registry_model_name(self):
        """create_llm() always passes the MODEL_REGISTRY name as the 'model' kwarg."""
        openai_mock = MagicMock()
        gemini_mock = MagicMock()
        mistral_mock = MagicMock()
        constructor_map = {
            LLMProvider.OPENAI: openai_mock.ChatOpenAI,
            LLMProvider.GEMINI: gemini_mock.ChatGoogleGenerativeAI,
            LLMProvider.MISTRAL: mistral_mock.ChatMistralAI,
        }
        sys_modules_patch = {
            "langchain_openai": openai_mock,
            "langchain_google_genai": gemini_mock,
            "langchain_mistralai": mistral_mock,
        }
        with patch.dict("sys.modules", sys_modules_patch):
            for provider in LLMProvider:
                for tier in ModelTier:
                    constructor_map[provider].reset_mock()
                    with self.subTest(provider=provider, tier=tier):
                        create_llm(provider, tier)
                        kwargs = constructor_map[provider].call_args.kwargs
                        self.assertEqual(
                            kwargs["model"], MODEL_REGISTRY[provider][tier]
                        )


@pytest.mark.unit
class TestResolveProvider(unittest.TestCase):
    """_resolve_provider() reads step env var and falls back to DEFAULT_PROVIDER."""

    _ABSENT_VAR = "DEFINITELY_NOT_SET_STEP_84f7a3b2_BLCTEST"

    def test_returns_llm_provider_instance(self):
        result = _resolve_provider(self._ABSENT_VAR)
        self.assertIsInstance(result, LLMProvider)

    def test_reads_step_env_var(self):
        with patch.dict(os.environ, {"BLC_TEST_STEP_PROVIDER": "gemini"}):
            result = _resolve_provider("BLC_TEST_STEP_PROVIDER")
        self.assertEqual(result, LLMProvider.GEMINI)

    def test_reads_openai_from_env_var(self):
        with patch.dict(os.environ, {"BLC_TEST_STEP_PROVIDER": "openai"}):
            result = _resolve_provider("BLC_TEST_STEP_PROVIDER")
        self.assertEqual(result, LLMProvider.OPENAI)

    def test_invalid_env_var_value_raises_value_error(self):
        with patch.dict(os.environ, {"BLC_TEST_STEP_PROVIDER": "invalid_provider_xyz"}):
            with self.assertRaises(ValueError):
                _resolve_provider("BLC_TEST_STEP_PROVIDER")

    def test_default_provider_is_llm_provider(self):
        self.assertIsInstance(DEFAULT_PROVIDER, LLMProvider)


@pytest.mark.unit
class TestGlobalProviderOverride(unittest.TestCase):
    """Setting PROVIDER env var switches the default provider for all tiers."""

    _ABSENT_STEP_VAR = "DEFINITELY_NOT_SET_STEP_84f7a3b2_BLCTEST"

    def test_global_provider_gemini_is_default_when_patched(self):
        """When DEFAULT_PROVIDER is GEMINI, unset step vars resolve to GEMINI."""
        with patch.object(llm_model_module, "DEFAULT_PROVIDER", LLMProvider.GEMINI):
            result = _resolve_provider(self._ABSENT_STEP_VAR)
        self.assertEqual(result, LLMProvider.GEMINI)

    def test_global_provider_openai_is_default_when_patched(self):
        """When DEFAULT_PROVIDER is OPENAI, unset step vars resolve to OPENAI."""
        with patch.object(llm_model_module, "DEFAULT_PROVIDER", LLMProvider.OPENAI):
            result = _resolve_provider(self._ABSENT_STEP_VAR)
        self.assertEqual(result, LLMProvider.OPENAI)

    def test_global_provider_mistral_is_default_when_patched(self):
        """When DEFAULT_PROVIDER is MISTRAL, unset step vars resolve to MISTRAL."""
        with patch.object(llm_model_module, "DEFAULT_PROVIDER", LLMProvider.MISTRAL):
            result = _resolve_provider(self._ABSENT_STEP_VAR)
        self.assertEqual(result, LLMProvider.MISTRAL)

    def test_step_override_takes_precedence_over_global(self):
        """An explicit step env var overrides DEFAULT_PROVIDER."""
        with patch.object(llm_model_module, "DEFAULT_PROVIDER", LLMProvider.GEMINI):
            with patch.dict(os.environ, {"BLC_TEST_STEP_PROVIDER": "openai"}):
                result = _resolve_provider("BLC_TEST_STEP_PROVIDER")
        self.assertEqual(result, LLMProvider.OPENAI)


@pytest.mark.unit
class TestPublicAliases(unittest.TestCase):
    """Public alias functions delegate to create_llm() with the expected tier."""

    def _assert_tier(self, fn, expected_tier: ModelTier) -> None:
        with patch.object(llm_model_module, "create_llm") as mock_create:
            fn()
            args = mock_create.call_args[0]
            self.assertEqual(args[1], expected_tier)

    def test_get_llm_low_delegates_to_low_tier(self):
        self._assert_tier(get_llm_low, ModelTier.LOW)

    def test_get_llm_high_delegates_to_high_tier(self):
        self._assert_tier(get_llm_high, ModelTier.HIGH)

    def test_llm_analyst_delegates_to_analyst_tier(self):
        self._assert_tier(llm_analyst, ModelTier.ANALYST)

    def test_llm_outline_delegates_to_outline_tier(self):
        self._assert_tier(llm_outline, ModelTier.OUTLINE)

    def test_llm_text_delegates_to_text_tier(self):
        self._assert_tier(llm_text, ModelTier.TEXT)

    def test_llm_check_title_delegates_to_check_title_tier(self):
        self._assert_tier(llm_check_title, ModelTier.CHECK_TITLE)

    def test_llm_title_abstract_delegates_to_title_abstract_tier(self):
        self._assert_tier(llm_title_abstract, ModelTier.TITLE_ABSTRACT)

    def test_default_llm_calls_get_llm_low(self):
        with patch.object(llm_model_module, "get_llm_low") as mock_low:
            default_llm()
            mock_low.assert_called_once()


@pytest.mark.unit
class TestPublicAPIExports(unittest.TestCase):
    """LLMProvider and ModelTier are exported from the top-level package."""

    def test_llm_provider_importable_from_package(self):
        from black_langcube import LLMProvider as LP

        self.assertIs(LP, LLMProvider)

    def test_model_tier_importable_from_package(self):
        from black_langcube import ModelTier as MT

        self.assertIs(MT, ModelTier)

    def test_llm_provider_in_all(self):
        import black_langcube

        self.assertIn("LLMProvider", black_langcube.__all__)

    def test_model_tier_in_all(self):
        import black_langcube

        self.assertIn("ModelTier", black_langcube.__all__)


if __name__ == "__main__":
    unittest.main()
