"""
Tests for black_langcube configuration validation.

Covers ConfigurationError, get_api_key(), and validate_config() behaviour
using environment variable mocking — no real API key required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from black_langcube.config import ConfigurationError, get_api_key, validate_config  # noqa: E402


@pytest.mark.unit
class TestConfigurationError(unittest.TestCase):
    """ConfigurationError is a distinct exception class."""

    def test_is_exception_subclass(self):
        self.assertTrue(issubclass(ConfigurationError, Exception))

    def test_can_be_raised_and_caught(self):
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("test message")

    def test_message_is_preserved(self):
        msg = "some config problem"
        exc = ConfigurationError(msg)
        self.assertEqual(str(exc), msg)


@pytest.mark.unit
class TestGetApiKey(unittest.TestCase):
    """get_api_key() reads env vars and wraps them in SecretStr."""

    def test_raises_when_env_var_absent(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                get_api_key("OPENAI_API_KEY")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_raises_when_env_var_empty_string(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                get_api_key("OPENAI_API_KEY")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_returns_secret_str_when_set(self):
        from pydantic import SecretStr

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-value"}):
            result = get_api_key("OPENAI_API_KEY")
        self.assertIsInstance(result, SecretStr)

    def test_secret_value_accessible_via_get_secret_value(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-value"}):
            result = get_api_key("OPENAI_API_KEY")
        self.assertEqual(result.get_secret_value(), "sk-test-value")

    def test_secret_not_exposed_via_str(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-value"}):
            result = get_api_key("OPENAI_API_KEY")
        self.assertNotIn("sk-test-value", str(result))

    def test_secret_not_exposed_via_repr(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-value"}):
            result = get_api_key("OPENAI_API_KEY")
        self.assertNotIn("sk-test-value", repr(result))

    def test_arbitrary_env_var_name(self):
        with patch.dict("os.environ", {"MY_CUSTOM_KEY": "abc123"}):
            result = get_api_key("MY_CUSTOM_KEY")
        self.assertEqual(result.get_secret_value(), "abc123")


@pytest.mark.unit
class TestValidateConfig(unittest.TestCase):
    """validate_config() checks all required environment variables."""

    def test_raises_when_required_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                validate_config()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_raises_when_required_key_empty(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with self.assertRaises(ConfigurationError):
                validate_config()

    def test_passes_when_all_required_vars_present(self):
        env = {"OPENAI_API_KEY": "sk-test-value"}
        with patch.dict("os.environ", env):
            try:
                validate_config()
            except ConfigurationError as exc:
                self.fail(f"validate_config() raised unexpectedly: {exc}")

    def test_is_idempotent(self):
        """Calling validate_config() multiple times with valid env is safe."""
        env = {"OPENAI_API_KEY": "sk-test-value"}
        with patch.dict("os.environ", env):
            validate_config()
            validate_config()  # second call must not raise

    def test_error_message_lists_all_missing_keys(self):
        """All missing required keys appear in the error message."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                validate_config()
        error_msg = str(ctx.exception)
        from black_langcube.config import REQUIRED_ENV_VARS

        for var in REQUIRED_ENV_VARS:
            self.assertIn(var, error_msg)


@pytest.mark.unit
class TestPublicApiExports(unittest.TestCase):
    """validate_config and ConfigurationError are exported from the top-level package."""

    def test_validate_config_exported(self):
        from black_langcube import validate_config as vc  # noqa: F401

        self.assertTrue(callable(vc))

    def test_configuration_error_exported(self):
        from black_langcube import ConfigurationError as CE  # noqa: F401

        self.assertTrue(issubclass(CE, Exception))


if __name__ == "__main__":
    unittest.main()
