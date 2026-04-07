"""
Unit tests for the sync and async circuit-breaker implementations,
robust_invoke_async short-circuit behavior, and the module-level registry.

All tests run without real API calls — the LangChain chain and openai errors are mocked.
"""

import asyncio
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
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)
from black_langcube.llm_modules.circuit_breaker_async import (  # noqa: E402
    CircuitBreakerAsync,
    CircuitBreakerOpenError as AsyncCircuitBreakerOpenError,
    CircuitState as AsyncCircuitState,
    get_circuit_breaker as async_get_circuit_breaker,
    reset_all_circuit_breakers as async_reset_all_circuit_breakers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_breaker(threshold=3, timeout=60) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout)


def _make_async_breaker(threshold=3, timeout=60) -> CircuitBreakerAsync:
    return CircuitBreakerAsync(failure_threshold=threshold, recovery_timeout=timeout)


def _connection_error() -> openai.APIConnectionError:
    req = MagicMock()
    return openai.APIConnectionError(request=req)


def _timeout_error() -> openai.APITimeoutError:
    req = MagicMock()
    return openai.APITimeoutError(request=req)


def _internal_error() -> openai.InternalServerError:
    resp = MagicMock()
    resp.headers = {}
    return openai.InternalServerError(message="internal", response=resp, body=None)


def _rate_limit_error() -> openai.RateLimitError:
    resp = MagicMock()
    resp.headers = {}
    return openai.RateLimitError(message="rate limit", response=resp, body=None)


# ---------------------------------------------------------------------------
# Sync circuit-breaker tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncCircuitBreakerTransitions(unittest.TestCase):
    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def test_initial_state_is_closed(self):
        cb = _make_sync_breaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_closed_to_open_after_threshold_failures(self):
        cb = _make_sync_breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()  # 3rd — should open
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_open_rejects_call(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        with self.assertRaises(CircuitBreakerOpenError):
            with cb.call():
                pass  # pragma: no cover

    def test_open_transitions_to_half_open_after_timeout(self):
        cb = _make_sync_breaker(threshold=1, timeout=10)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        # Before timeout: probe calls are still rejected
        with self.assertRaises(CircuitBreakerOpenError):
            with cb.call():
                pass  # pragma: no cover
        # After timeout elapses: _check_recovery transitions to HALF_OPEN
        cb._opened_at = time.time() - 11
        cb._check_recovery()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

    def test_half_open_to_closed_on_success(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70  # force past recovery_timeout
        cb._check_recovery()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_half_open_to_open_on_failure(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_rate_limit_error_does_not_increment_counter(self):
        cb = _make_sync_breaker(threshold=3)
        # RateLimitError is NOT a SERVICE_ERROR — context manager won't record it
        for _ in range(5):
            try:
                with cb.call():
                    raise _rate_limit_error()
            except openai.RateLimitError:
                pass
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_service_errors_increment_counter(self):
        for make_error in [_connection_error, _timeout_error, _internal_error]:
            with self.subTest(error=make_error):
                cb = _make_sync_breaker(threshold=5)
                try:
                    with cb.call():
                        raise make_error()
                except Exception:
                    pass
                self.assertEqual(cb._failure_count, 1)

    def test_success_resets_failure_count(self):
        cb = _make_sync_breaker(threshold=5)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb._failure_count, 2)
        cb.record_success()
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_half_open_second_probe_is_blocked(self):
        import threading

        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        probe_entered = threading.Event()
        probe_continue = threading.Event()
        second_was_blocked = threading.Event()

        def first_probe():
            with cb.call():
                probe_entered.set()
                probe_continue.wait(timeout=2)

        def second_probe():
            probe_entered.wait(timeout=2)
            try:
                with cb.call():
                    pass  # pragma: no cover
            except CircuitBreakerOpenError:
                second_was_blocked.set()
            finally:
                probe_continue.set()  # unblock first probe

        t1 = threading.Thread(target=first_probe)
        t2 = threading.Thread(target=second_probe)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        self.assertTrue(
            second_was_blocked.is_set(),
            "Second concurrent probe should be blocked in HALF_OPEN",
        )

    def test_half_open_probe_flag_cleared_after_success(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        with cb.call():  # probe succeeds
            pass
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertFalse(cb._probe_in_flight)

    def test_half_open_probe_flag_cleared_after_service_failure(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        try:
            with cb.call():
                raise _connection_error()
        except openai.APIConnectionError:
            pass
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb._probe_in_flight)

    def test_half_open_probe_flag_cleared_after_non_service_error(self):
        cb = _make_sync_breaker(threshold=1)
        cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        try:
            with cb.call():
                raise _rate_limit_error()
        except openai.RateLimitError:
            pass
        # State stays HALF_OPEN (rate limit is not a service error)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        # But probe flag must be cleared so a future probe can attempt
        self.assertFalse(cb._probe_in_flight)


# ---------------------------------------------------------------------------
# Sync registry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncRegistry(unittest.TestCase):
    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def test_get_circuit_breaker_returns_same_instance(self):
        a = get_circuit_breaker("openai_api")
        b = get_circuit_breaker("openai_api")
        self.assertIs(a, b)

    def test_reset_all_circuit_breakers_clears_registry(self):
        a = get_circuit_breaker("openai_api")
        reset_all_circuit_breakers()
        b = get_circuit_breaker("openai_api")
        self.assertIsNot(a, b)

    def test_unknown_service_gets_default_config(self):
        cb = get_circuit_breaker("some_unknown_service")
        self.assertIsInstance(cb, CircuitBreaker)


# ---------------------------------------------------------------------------
# Async circuit-breaker tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncCircuitBreakerTransitions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def test_initial_state_is_closed(self):
        cb = _make_async_breaker()
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_closed_to_open_after_threshold_failures(self):
        cb = _make_async_breaker(threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)
        await cb.record_failure()
        self.assertEqual(cb.state, AsyncCircuitState.OPEN)

    async def test_open_rejects_call(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        self.assertEqual(cb.state, AsyncCircuitState.OPEN)
        with self.assertRaises(AsyncCircuitBreakerOpenError):
            async with cb.call():
                pass  # pragma: no cover

    async def test_open_transitions_to_half_open_after_timeout(self):
        cb = _make_async_breaker(threshold=1, timeout=10)
        await cb.record_failure()
        self.assertEqual(cb.state, AsyncCircuitState.OPEN)
        cb._opened_at = time.time() - 11
        cb._check_recovery()
        self.assertEqual(cb.state, AsyncCircuitState.HALF_OPEN)

    async def test_half_open_to_closed_on_success(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        self.assertEqual(cb.state, AsyncCircuitState.HALF_OPEN)
        await cb.record_success()
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_half_open_to_open_on_failure(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        self.assertEqual(cb.state, AsyncCircuitState.HALF_OPEN)
        await cb.record_failure()
        self.assertEqual(cb.state, AsyncCircuitState.OPEN)

    async def test_rate_limit_error_does_not_increment_counter(self):
        cb = _make_async_breaker(threshold=3)
        for _ in range(5):
            try:
                async with cb.call():
                    raise _rate_limit_error()
            except openai.RateLimitError:
                pass
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_service_errors_increment_counter(self):
        for make_error in [_connection_error, _timeout_error, _internal_error]:
            with self.subTest(error=make_error):
                cb = _make_async_breaker(threshold=5)
                try:
                    async with cb.call():
                        raise make_error()
                except Exception:
                    pass
                self.assertEqual(cb._failure_count, 1)
                # reset for next subTest
                cb._failure_count = 0

    async def test_success_resets_failure_count(self):
        cb = _make_async_breaker(threshold=5)
        await cb.record_failure()
        await cb.record_failure()
        self.assertEqual(cb._failure_count, 2)
        await cb.record_success()
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_half_open_second_probe_is_blocked(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        self.assertEqual(cb.state, AsyncCircuitState.HALF_OPEN)

        probe_entered = asyncio.Event()
        probe_continue = asyncio.Event()
        second_was_blocked = False

        async def first_probe():
            async with cb.call():
                probe_entered.set()
                await probe_continue.wait()

        async def second_probe():
            nonlocal second_was_blocked
            await probe_entered.wait()
            try:
                async with cb.call():
                    pass  # pragma: no cover
            except AsyncCircuitBreakerOpenError:
                second_was_blocked = True
            finally:
                probe_continue.set()  # unblock first probe

        await asyncio.gather(first_probe(), second_probe())
        self.assertTrue(
            second_was_blocked, "Second concurrent probe should be blocked in HALF_OPEN"
        )

    async def test_half_open_probe_flag_cleared_after_success(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        async with cb.call():  # probe succeeds
            pass
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)
        self.assertFalse(cb._probe_in_flight)

    async def test_half_open_probe_flag_cleared_after_service_failure(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        try:
            async with cb.call():
                raise _connection_error()
        except openai.APIConnectionError:
            pass
        self.assertEqual(cb.state, AsyncCircuitState.OPEN)
        self.assertFalse(cb._probe_in_flight)

    async def test_half_open_probe_flag_cleared_after_non_service_error(self):
        cb = _make_async_breaker(threshold=1)
        await cb.record_failure()
        cb._opened_at = time.time() - 70
        cb._check_recovery()
        try:
            async with cb.call():
                raise _rate_limit_error()
        except openai.RateLimitError:
            pass
        # State stays HALF_OPEN (rate limit is not a service error)
        self.assertEqual(cb.state, AsyncCircuitState.HALF_OPEN)
        # But probe flag must be cleared so a future probe can attempt
        self.assertFalse(cb._probe_in_flight)


# ---------------------------------------------------------------------------
# Async registry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncRegistry(unittest.TestCase):
    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    def test_get_circuit_breaker_returns_same_instance(self):
        a = async_get_circuit_breaker("openai_api")
        b = async_get_circuit_breaker("openai_api")
        self.assertIs(a, b)

    def test_reset_all_circuit_breakers_clears_registry(self):
        a = async_get_circuit_breaker("openai_api")
        async_reset_all_circuit_breakers()
        b = async_get_circuit_breaker("openai_api")
        self.assertIsNot(a, b)

    def test_unknown_service_gets_default_config(self):
        cb = async_get_circuit_breaker("some_unknown_service")
        self.assertIsInstance(cb, CircuitBreakerAsync)


# ---------------------------------------------------------------------------
# robust_invoke_async integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeAsyncCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """
    Verifies robust_invoke_async short-circuits without calling the chain when
    the circuit is OPEN and correctly records/ignores failures by error type.
    """

    def setUp(self):
        async_reset_all_circuit_breakers()

    def tearDown(self):
        async_reset_all_circuit_breakers()

    async def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke_async import robust_invoke_async

        return await robust_invoke_async(chain, **kwargs)

    def _open_circuit(self):
        """Force the openai_api circuit into OPEN state."""
        cb = async_get_circuit_breaker("openai_api")
        cb._state = AsyncCircuitState.OPEN
        cb._opened_at = time.time()
        return cb

    async def test_open_circuit_returns_error_dict_without_calling_chain(self):
        self._open_circuit()
        chain = AsyncMock()
        result, tokens = await self._invoke(chain)
        self.assertIn("error", result)
        self.assertIn("Circuit breaker open", result["error"])
        chain.ainvoke.assert_not_called()

    async def test_rate_limit_error_does_not_open_circuit(self):
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=_rate_limit_error())

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb.prompt_tokens = 0
            mock_cb.completion_tokens = 0
            mock_cb.total_cost = 0
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result, _ = await self._invoke(chain, max_retries=1)

        cb = async_get_circuit_breaker("openai_api")
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, AsyncCircuitState.CLOSED)

    async def test_connection_error_is_recorded_by_circuit_breaker(self):
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=_connection_error())

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result, _ = await self._invoke(chain, max_retries=1)

        cb = async_get_circuit_breaker("openai_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    async def test_threshold_failures_open_circuit(self):
        """After CB_FAILURE_THRESHOLD consecutive service errors the circuit opens."""
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=_connection_error())

        cb = async_get_circuit_breaker("openai_api")
        threshold = cb._failure_threshold

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            for _ in range(threshold):
                await self._invoke(chain, max_retries=1)

        self.assertEqual(cb.state, AsyncCircuitState.OPEN)


# ---------------------------------------------------------------------------
# robust_invoke (sync) integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRobustInvokeCircuitBreaker(unittest.TestCase):
    """
    Verifies robust_invoke short-circuits without calling the chain when the
    circuit is OPEN and correctly records/ignores failures by error type.
    """

    def setUp(self):
        reset_all_circuit_breakers()

    def tearDown(self):
        reset_all_circuit_breakers()

    def _invoke(self, chain, **kwargs):
        from black_langcube.llm_modules.robust_invoke import robust_invoke

        return robust_invoke(chain, **kwargs)

    def _open_circuit(self):
        """Force the openai_api circuit into OPEN state."""
        cb = get_circuit_breaker("openai_api")
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time()
        return cb

    def test_open_circuit_returns_error_dict_without_calling_chain(self):
        self._open_circuit()
        chain = MagicMock()
        result, tokens = self._invoke(chain)
        self.assertIn("error", result)
        self.assertIn("Circuit breaker open", result["error"])
        chain.invoke.assert_not_called()

    def test_rate_limit_error_does_not_open_circuit(self):
        chain = MagicMock()
        chain.invoke = MagicMock(side_effect=_rate_limit_error())

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb.prompt_tokens = 0
            mock_cb.completion_tokens = 0
            mock_cb.total_cost = 0
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result, _ = self._invoke(chain, max_retries=1)

        cb = get_circuit_breaker("openai_api")
        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_connection_error_is_recorded_by_circuit_breaker(self):
        chain = MagicMock()
        chain.invoke = MagicMock(side_effect=_connection_error())

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result, _ = self._invoke(chain, max_retries=1)

        cb = get_circuit_breaker("openai_api")
        self.assertEqual(cb._failure_count, 1)
        self.assertIn("error", result)

    def test_threshold_failures_open_circuit(self):
        """After CB_FAILURE_THRESHOLD consecutive service errors the circuit opens."""
        chain = MagicMock()
        chain.invoke = MagicMock(side_effect=_connection_error())

        cb = get_circuit_breaker("openai_api")
        threshold = cb._failure_threshold

        with patch("langchain_community.callbacks.get_openai_callback") as mock_cb_ctx:
            mock_cb = MagicMock()
            mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
            mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)
            for _ in range(threshold):
                self._invoke(chain, max_retries=1)

        self.assertEqual(cb.state, CircuitState.OPEN)


if __name__ == "__main__":
    unittest.main()
