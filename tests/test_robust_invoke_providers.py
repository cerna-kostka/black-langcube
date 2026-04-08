"""
Parametrized tests for provider-agnostic robust_invoke / robust_invoke_async.

All tests use mocked chains — no real API calls are made.

Coverage:
  - OpenAI path: happy path, rate-limit backoff, service-error circuit-breaker,
    circuit-open short-circuit.
  - Gemini path (simulated): ResourceExhausted triggers backoff;
    ServiceUnavailable increments circuit breaker.
  - Generic Exception path: unrecognised provider errors returned as
    {"error": ...} without propagating.
  - Token extraction: usage_metadata present; usage_metadata absent (zero
    fallback); OpenAI callback fallback.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import openai  # noqa: E402

from black_langcube.llm_modules.circuit_breaker import (  # noqa: E402
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)
from black_langcube.llm_modules.circuit_breaker_async import (  # noqa: E402
    CircuitState as AsyncCircuitState,
    get_circuit_breaker as async_get_circuit_breaker,
    reset_all_circuit_breakers as async_reset_all_circuit_breakers,
)
from black_langcube.llm_modules._token_utils import _extract_token_usage  # noqa: E402

# ---------------------------------------------------------------------------
# Fake provider error classes for non-installed packages
# ---------------------------------------------------------------------------


class _FakeResourceExhausted(Exception):
    """Simulates ``google.api_core.exceptions.ResourceExhausted`` (HTTP 429)."""


class _FakeServiceUnavailable(Exception):
    """Simulates ``google.api_core.exceptions.ServiceUnavailable`` (HTTP 503)."""


class _FakeGoogleExceptions:
    """Minimal stand-in for the ``google.api_core.exceptions`` module."""

    ResourceExhausted = _FakeResourceExhausted
    ServiceUnavailable = _FakeServiceUnavailable
    DeadlineExceeded = _FakeServiceUnavailable
    InternalServerError = _FakeServiceUnavailable


class _FakeGenericProviderError(Exception):
    """Unrecognised provider error that should trigger the generic fallback."""


class _FakeMistralAPIException(Exception):
    """Simulates ``mistralai.exceptions.MistralAPIException``."""

    def __init__(self, message="", http_status=None, status_code=None):
        super().__init__(message)
        self.http_status = http_status
        self.status_code = status_code


class _FakeMistralConnectionException(Exception):
    """Simulates ``mistralai.exceptions.MistralConnectionException``."""


class _FakeMistralExceptions:
    """Minimal stand-in for the ``mistralai.exceptions`` module."""

    MistralAPIException = _FakeMistralAPIException
    MistralConnectionException = _FakeMistralConnectionException


# ---------------------------------------------------------------------------
# Chain factories
# ---------------------------------------------------------------------------


def _sync_chain(return_value=None, side_effect=None) -> MagicMock:
    chain = MagicMock()
    if side_effect is not None:
        chain.invoke = MagicMock(side_effect=side_effect)
    else:
        chain.invoke = MagicMock(return_value=return_value)
    return chain


def _async_chain(return_value=None, side_effect=None) -> MagicMock:
    chain = MagicMock()
    if side_effect is not None:
        chain.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


def _ai_message(input_tokens: int = 5, output_tokens: int = 10) -> MagicMock:
    msg = MagicMock()
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return msg


def _openai_rate_limit() -> openai.RateLimitError:
    resp = MagicMock()
    resp.headers = {}
    return openai.RateLimitError(message="rate limit", response=resp, body=None)


def _openai_connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=MagicMock())


# ---------------------------------------------------------------------------
# Token extraction tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractTokenUsage(unittest.TestCase):
    def test_usage_metadata_returns_correct_values(self):
        result = _ai_message(input_tokens=7, output_tokens=13)
        tokens = _extract_token_usage(result)
        self.assertEqual(tokens["tokens_in"], 7)
        self.assertEqual(tokens["tokens_out"], 13)
        self.assertEqual(tokens["tokens_price"], 0)

    def test_usage_metadata_absent_returns_zeros(self):
        result = MagicMock(spec=[])  # no usage_metadata attribute
        tokens = _extract_token_usage(result)
        self.assertEqual(tokens, {"tokens_in": 0, "tokens_out": 0, "tokens_price": 0})

    def test_cb_fallback_when_no_usage_metadata(self):
        result = MagicMock(spec=[])  # no usage_metadata
        cb = MagicMock()
        cb.prompt_tokens = 3
        cb.completion_tokens = 7
        cb.total_cost = 0.001
        tokens = _extract_token_usage(result, cb)
        self.assertEqual(tokens["tokens_in"], 3)
        self.assertEqual(tokens["tokens_out"], 7)
        self.assertAlmostEqual(tokens["tokens_price"], 0.001)

    def test_usage_metadata_preferred_over_cb_for_token_counts(self):
        """usage_metadata wins for token counts; cb.total_cost still supplies price."""
        result = _ai_message(input_tokens=5, output_tokens=10)
        cb = MagicMock()
        cb.prompt_tokens = 99
        cb.completion_tokens = 99
        cb.total_cost = 99
        tokens = _extract_token_usage(result, cb)
        self.assertEqual(tokens["tokens_in"], 5)
        self.assertEqual(tokens["tokens_out"], 10)

    def test_price_from_cb_even_when_usage_metadata_present(self):
        """cb.total_cost is used for price even when usage_metadata supplies token counts."""
        result = _ai_message(input_tokens=5, output_tokens=10)
        cb = MagicMock()
        cb.prompt_tokens = 99
        cb.completion_tokens = 99
        cb.total_cost = 0.05
        tokens = _extract_token_usage(result, cb)
        self.assertEqual(tokens["tokens_in"], 5)
        self.assertEqual(tokens["tokens_out"], 10)
        self.assertAlmostEqual(tokens["tokens_price"], 0.05)

    def test_none_result_returns_zeros(self):
        tokens = _extract_token_usage(None)
        self.assertEqual(tokens, {"tokens_in": 0, "tokens_out": 0, "tokens_price": 0})


# ---------------------------------------------------------------------------
# Sync – OpenAI path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeOpenAI(unittest.TestCase):
    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke import robust_invoke

        return robust_invoke(chain, **kwargs)

    def _service_name(self):
        from black_langcube.llm_modules.robust_invoke import _SERVICE_NAME

        return _SERVICE_NAME

    def test_happy_path_with_usage_metadata(self):
        msg = _ai_message(5, 10)
        result, tokens = self._invoke(_sync_chain(return_value=msg))
        self.assertIs(result, msg)
        self.assertEqual(tokens["tokens_in"], 5)
        self.assertEqual(tokens["tokens_out"], 10)
        self.assertEqual(tokens["tokens_price"], 0)

    def test_happy_path_zero_tokens_when_no_usage_metadata(self):
        msg = MagicMock(spec=[])  # no usage_metadata
        result, tokens = self._invoke(_sync_chain(return_value=msg))
        self.assertIs(result, msg)
        self.assertEqual(tokens, {"tokens_in": 0, "tokens_out": 0, "tokens_price": 0})

    def test_rate_limit_error_triggers_backoff_and_returns_error(self):
        chain = _sync_chain(side_effect=_openai_rate_limit())
        result, _ = self._invoke(chain, max_retries=2, backoff_factor=0)
        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    def test_rate_limit_does_not_increment_circuit_breaker(self):
        chain = _sync_chain(side_effect=_openai_rate_limit())
        self._invoke(chain, max_retries=1, backoff_factor=0)
        cb = get_circuit_breaker(self._service_name())
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_api_connection_error_increments_circuit_breaker(self):
        chain = _sync_chain(side_effect=_openai_connection_error())
        result, _ = self._invoke(chain, max_retries=1)
        cb = get_circuit_breaker(self._service_name())
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    def test_api_connection_error_returns_provider_error_message(self):
        chain = _sync_chain(side_effect=_openai_connection_error())
        result, _ = self._invoke(chain, max_retries=1)
        self.assertIn("error", result)
        self.assertIn("Provider error", result["error"])
        self.assertIn("APIConnectionError", result["error"])

    def test_circuit_breaker_open_returns_error_without_calling_chain(self):
        cb = get_circuit_breaker(self._service_name())
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time()
        chain = _sync_chain(return_value=_ai_message())
        result, _ = self._invoke(chain)
        self.assertIn("Circuit breaker open", result["error"])
        chain.invoke.assert_not_called()

    def test_threshold_failures_open_circuit(self):
        chain = _sync_chain(side_effect=_openai_connection_error())
        cb = get_circuit_breaker(self._service_name())
        threshold = cb._failure_threshold
        for _ in range(threshold):
            self._invoke(chain, max_retries=1)
        self.assertEqual(cb.state, CircuitState.OPEN)


# ---------------------------------------------------------------------------
# Async – OpenAI path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeAsyncOpenAI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke_async import robust_invoke_async

        return await robust_invoke_async(chain, **kwargs)

    def _service_name(self):
        from black_langcube.llm_modules.robust_invoke_async import _SERVICE_NAME

        return _SERVICE_NAME

    async def test_happy_path_with_usage_metadata(self):
        msg = _ai_message(5, 10)
        result, tokens = await self._invoke(_async_chain(return_value=msg))
        self.assertIs(result, msg)
        self.assertEqual(tokens["tokens_in"], 5)
        self.assertEqual(tokens["tokens_out"], 10)

    async def test_rate_limit_error_triggers_backoff_and_returns_error(self):
        chain = _async_chain(side_effect=_openai_rate_limit())
        result, _ = await self._invoke(chain, max_retries=2, backoff_factor=0)
        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    async def test_rate_limit_does_not_increment_circuit_breaker(self):
        chain = _async_chain(side_effect=_openai_rate_limit())
        await self._invoke(chain, max_retries=1, backoff_factor=0)
        cb = async_get_circuit_breaker(self._service_name())
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_api_connection_error_increments_circuit_breaker(self):
        chain = _async_chain(side_effect=_openai_connection_error())
        result, _ = await self._invoke(chain, max_retries=1)
        cb = async_get_circuit_breaker(self._service_name())
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    async def test_circuit_breaker_open_returns_error_without_calling_chain(self):
        cb = async_get_circuit_breaker(self._service_name())
        cb._state = AsyncCircuitState.OPEN
        cb._opened_at = time.time()
        chain = _async_chain(return_value=_ai_message())
        result, _ = await self._invoke(chain)
        self.assertIn("Circuit breaker open", result["error"])
        chain.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Sync – Gemini path (simulated via patched imports)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeGemini(unittest.TestCase):
    """Simulate the Gemini provider by patching ``_google_exc`` and
    ``_ALL_PROVIDER_SERVICE_ERRORS`` in ``_token_utils``.

    This allows testing Gemini-specific error paths without requiring the
    ``google-api-core`` package to be installed.
    """

    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke import robust_invoke

        return robust_invoke(chain, **kwargs)

    def test_resource_exhausted_triggers_backoff(self):
        """ResourceExhausted is a rate-limit error — should backoff, not open circuit."""
        with patch(
            "black_langcube.llm_modules._token_utils._google_exc",
            _FakeGoogleExceptions,
        ):
            chain = _sync_chain(side_effect=_FakeResourceExhausted("quota"))
            result, _ = self._invoke(chain, max_retries=2, backoff_factor=0)

        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    def test_resource_exhausted_does_not_increment_circuit_breaker(self):
        with patch(
            "black_langcube.llm_modules._token_utils._google_exc",
            _FakeGoogleExceptions,
        ):
            chain = _sync_chain(side_effect=_FakeResourceExhausted("quota"))
            self._invoke(chain, max_retries=1, backoff_factor=0, provider="gemini")

        cb = get_circuit_breaker("gemini_api")
        self.assertEqual(cb._failure_count, 0)

    def test_service_unavailable_increments_circuit_breaker(self):
        """ServiceUnavailable is a service-failure error — should increment the CB."""
        with (
            patch(
                "black_langcube.llm_modules._token_utils._google_exc",
                _FakeGoogleExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"gemini": (_FakeServiceUnavailable,)},
            ),
        ):
            chain = _sync_chain(side_effect=_FakeServiceUnavailable("unavailable"))
            result, _ = self._invoke(chain, max_retries=1, provider="gemini")

        cb = get_circuit_breaker("gemini_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    def test_service_unavailable_threshold_opens_circuit(self):
        with (
            patch(
                "black_langcube.llm_modules._token_utils._google_exc",
                _FakeGoogleExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"gemini": (_FakeServiceUnavailable,)},
            ),
        ):
            chain = _sync_chain(side_effect=_FakeServiceUnavailable("unavailable"))
            cb = get_circuit_breaker("gemini_api")
            threshold = cb._failure_threshold
            for _ in range(threshold):
                self._invoke(chain, max_retries=1, provider="gemini")

        self.assertEqual(cb.state, CircuitState.OPEN)


# ---------------------------------------------------------------------------
# Async – Gemini path (simulated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeAsyncGemini(unittest.IsolatedAsyncioTestCase):
    """Simulate the Gemini provider by patching ``_google_exc`` and
    ``_ALL_PROVIDER_SERVICE_ERRORS`` in ``_token_utils``."""

    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke_async import robust_invoke_async

        return await robust_invoke_async(chain, **kwargs)

    async def test_resource_exhausted_triggers_backoff(self):
        with patch(
            "black_langcube.llm_modules._token_utils._google_exc",
            _FakeGoogleExceptions,
        ):
            chain = _async_chain(side_effect=_FakeResourceExhausted("quota"))
            result, _ = await self._invoke(chain, max_retries=2, backoff_factor=0)

        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    async def test_service_unavailable_increments_circuit_breaker(self):
        with (
            patch(
                "black_langcube.llm_modules._token_utils._google_exc",
                _FakeGoogleExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"gemini": (_FakeServiceUnavailable,)},
            ),
        ):
            chain = _async_chain(side_effect=_FakeServiceUnavailable("unavailable"))
            result, _ = await self._invoke(chain, max_retries=1, provider="gemini")

        cb = async_get_circuit_breaker("gemini_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    async def test_service_unavailable_threshold_opens_circuit(self):
        with (
            patch(
                "black_langcube.llm_modules._token_utils._google_exc",
                _FakeGoogleExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"gemini": (_FakeServiceUnavailable,)},
            ),
        ):
            chain = _async_chain(side_effect=_FakeServiceUnavailable("unavailable"))
            cb = async_get_circuit_breaker("gemini_api")
            threshold = cb._failure_threshold
            for _ in range(threshold):
                await self._invoke(chain, max_retries=1, provider="gemini")

        self.assertEqual(cb.state, AsyncCircuitState.OPEN)


# ---------------------------------------------------------------------------
# Sync – Mistral path (simulated via patched imports)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeMistral(unittest.TestCase):
    """Simulate the Mistral provider by patching ``_mistral_exc`` and
    ``_ALL_PROVIDER_SERVICE_ERRORS`` in ``_token_utils``."""

    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke import robust_invoke

        return robust_invoke(chain, **kwargs)

    def test_api_exception_with_429_triggers_backoff(self):
        """MistralAPIException with HTTP 429 is a rate-limit — should backoff."""
        with patch(
            "black_langcube.llm_modules._token_utils._mistral_exc",
            _FakeMistralExceptions,
        ):
            error = _FakeMistralAPIException("quota exceeded", http_status=429)
            chain = _sync_chain(side_effect=error)
            result, _ = self._invoke(
                chain, max_retries=2, backoff_factor=0, provider="mistral"
            )

        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    def test_api_exception_with_429_does_not_increment_circuit_breaker(self):
        with patch(
            "black_langcube.llm_modules._token_utils._mistral_exc",
            _FakeMistralExceptions,
        ):
            error = _FakeMistralAPIException("quota exceeded", http_status=429)
            chain = _sync_chain(side_effect=error)
            self._invoke(chain, max_retries=1, backoff_factor=0, provider="mistral")

        cb = get_circuit_breaker("mistral_api")
        self.assertEqual(cb._failure_count, 0)

    def test_connection_exception_increments_circuit_breaker(self):
        """MistralConnectionException is a service failure — should increment CB."""
        with (
            patch(
                "black_langcube.llm_modules._token_utils._mistral_exc",
                _FakeMistralExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"mistral": (_FakeMistralConnectionException,)},
            ),
        ):
            chain = _sync_chain(
                side_effect=_FakeMistralConnectionException("conn failed")
            )
            result, _ = self._invoke(chain, max_retries=1, provider="mistral")

        cb = get_circuit_breaker("mistral_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Async – Mistral path (simulated via patched imports)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeAsyncMistral(unittest.IsolatedAsyncioTestCase):
    """Simulate the Mistral provider by patching ``_mistral_exc`` and
    ``_ALL_PROVIDER_SERVICE_ERRORS`` in ``_token_utils``."""

    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke_async import robust_invoke_async

        return await robust_invoke_async(chain, **kwargs)

    async def test_api_exception_with_429_triggers_backoff(self):
        """MistralAPIException with HTTP 429 is a rate-limit — should backoff."""
        with patch(
            "black_langcube.llm_modules._token_utils._mistral_exc",
            _FakeMistralExceptions,
        ):
            error = _FakeMistralAPIException("quota exceeded", http_status=429)
            chain = _async_chain(side_effect=error)
            result, _ = await self._invoke(
                chain, max_retries=2, backoff_factor=0, provider="mistral"
            )

        self.assertIn("error", result)
        self.assertIn("Rate limit", result["error"])

    async def test_connection_exception_increments_circuit_breaker(self):
        """MistralConnectionException is a service failure — should increment CB."""
        with (
            patch(
                "black_langcube.llm_modules._token_utils._mistral_exc",
                _FakeMistralExceptions,
            ),
            patch(
                "black_langcube.llm_modules._token_utils._ALL_PROVIDER_SERVICE_ERRORS",
                {"mistral": (_FakeMistralConnectionException,)},
            ),
        ):
            chain = _async_chain(
                side_effect=_FakeMistralConnectionException("conn failed")
            )
            result, _ = await self._invoke(chain, max_retries=1, provider="mistral")

        cb = async_get_circuit_breaker("mistral_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Sync – Generic / unknown provider error
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeGenericError(unittest.TestCase):
    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke import robust_invoke

        return robust_invoke(chain, **kwargs)

    def test_unknown_exception_returned_as_error_dict(self):
        chain = _sync_chain(side_effect=_FakeGenericProviderError("something broke"))
        result, tokens = self._invoke(chain)
        self.assertIn("error", result)
        self.assertNotIn("Rate limit", result["error"])

    def test_unknown_exception_does_not_propagate(self):
        """No exception should escape robust_invoke."""
        chain = _sync_chain(side_effect=RuntimeError("unexpected"))
        try:
            result, _ = self._invoke(chain)
            self.assertIn("error", result)
        except Exception as exc:
            self.fail(f"robust_invoke raised an unexpected exception: {exc!r}")

    def test_unknown_exception_tokens_are_zero(self):
        chain = _sync_chain(side_effect=_FakeGenericProviderError("bad"))
        _, tokens = self._invoke(chain)
        self.assertEqual(tokens, {"tokens_in": 0, "tokens_out": 0, "tokens_price": 0})


# ---------------------------------------------------------------------------
# Async – Generic / unknown provider error
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeAsyncGenericError(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke_async import robust_invoke_async

        return await robust_invoke_async(chain, **kwargs)

    async def test_unknown_exception_returned_as_error_dict(self):
        chain = _async_chain(side_effect=_FakeGenericProviderError("broke"))
        result, tokens = await self._invoke(chain)
        self.assertIn("error", result)
        self.assertEqual(tokens, {"tokens_in": 0, "tokens_out": 0, "tokens_price": 0})

    async def test_unknown_exception_does_not_propagate(self):
        chain = _async_chain(side_effect=RuntimeError("unexpected async"))
        try:
            result, _ = await self._invoke(chain)
            self.assertIn("error", result)
        except Exception as exc:
            self.fail(f"robust_invoke_async raised an unexpected exception: {exc!r}")


if __name__ == "__main__":
    unittest.main()
