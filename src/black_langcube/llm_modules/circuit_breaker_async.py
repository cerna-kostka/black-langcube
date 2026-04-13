"""
Asynchronous circuit breaker for protecting LLM API calls.

Implements a three-state machine (CLOSED → OPEN → HALF_OPEN → CLOSED) using
an asyncio.Lock for async-safe state transitions.

Configuration is read at import time from environment variables:
  CB_FAILURE_THRESHOLD  — consecutive failures before opening (default: 5)
  CB_RECOVERY_TIMEOUT   — seconds to wait before probing in HALF_OPEN (default: 60)

This module mirrors the public API of circuit_breaker.py exactly, differing only
in the locking primitive and async context-manager usage.
"""

import asyncio
import os
import time
from enum import Enum

from black_langcube.llm_modules._token_utils import PROVIDER_SERVICE_ERRORS

_DEFAULT_CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
_DEFAULT_CB_RECOVERY_TIMEOUT = int(os.getenv("CB_RECOVERY_TIMEOUT", "60"))

CIRCUIT_BREAKER_CONFIG: dict[str, dict[str, int]] = {
    provider: {
        "failure_threshold": _DEFAULT_CB_FAILURE_THRESHOLD,
        "recovery_timeout": _DEFAULT_CB_RECOVERY_TIMEOUT,
    }
    for provider in ("openai_api", "gemini_api", "mistral_api")
}

SERVICE_ERRORS: tuple[type[BaseException], ...] = PROVIDER_SERVICE_ERRORS


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""


#: Public alias — both names refer to the same exception class so that
#: code using either ``CircuitBreakerError`` or ``CircuitBreakerOpenError``
#: can catch it with a single ``except`` clause.
CircuitBreakerError = CircuitBreakerOpenError


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerAsync:
    def __init__(self, failure_threshold: int, recovery_timeout: int) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._probe_in_flight: bool = False
        # asyncio.Lock is created lazily to avoid issues with event-loop binding
        # in tests that spin up fresh loops per test case.
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Public state interrogation
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    # ------------------------------------------------------------------
    # Internal transition helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitState) -> None:
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()

    def _check_recovery(self) -> None:
        """Transition OPEN → HALF_OPEN when the timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (time.time() - self._opened_at) >= self._recovery_timeout
        ):
            self._transition(CircuitState.HALF_OPEN)

    # ------------------------------------------------------------------
    # Failure / success recording
    # ------------------------------------------------------------------

    async def record_failure(self) -> None:
        async with self._get_lock():
            if self._state == CircuitState.HALF_OPEN:
                self._failure_count = self._failure_threshold
                self._probe_in_flight = False
                self._transition(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._transition(CircuitState.OPEN)

    async def record_success(self) -> None:
        async with self._get_lock():
            self._failure_count = 0
            self._probe_in_flight = False
            self._transition(CircuitState.CLOSED)

    # ------------------------------------------------------------------
    # Call gate
    # ------------------------------------------------------------------

    def call(
        self,
        service_errors: tuple[type[BaseException], ...] | None = None,
    ):
        """
        Async context manager that gates a single call through the breaker.

        Args:
            service_errors: Exception classes that count as service failures and
                increment the failure counter.  When *None* (default) the
                module-level ``SERVICE_ERRORS`` tuple is used.

        Usage::

            async with circuit_breaker.call():
                result = await chain.ainvoke(...)
        """
        return _AsyncBreakerContext(self, service_errors)


class _AsyncBreakerContext:
    def __init__(
        self,
        breaker: CircuitBreakerAsync,
        service_errors: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self._breaker = breaker
        self._is_probe = False
        self._service_errors = (
            service_errors if service_errors is not None else SERVICE_ERRORS
        )

    async def __aenter__(self):
        async with self._breaker._get_lock():
            self._breaker._check_recovery()
            if self._breaker._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    "Circuit breaker is OPEN (service unavailable)"
                )
            if self._breaker._state == CircuitState.HALF_OPEN:
                if self._breaker._probe_in_flight:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is HALF_OPEN (probe already in flight)"
                    )
                self._breaker._probe_in_flight = True
                self._is_probe = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._breaker.record_success()
        elif self._service_errors and issubclass(exc_type, self._service_errors):
            await self._breaker.record_failure()
        elif self._is_probe:
            # Other error (e.g. RateLimitError) during a HALF_OPEN probe:
            # clear the probe flag so future probes are not blocked.
            async with self._breaker._get_lock():
                self._breaker._probe_in_flight = False
        # RateLimitError and other errors are not recorded — let them propagate
        return False  # never suppress the exception


# ------------------------------------------------------------------
# Module-level registry
# ------------------------------------------------------------------

_circuit_breaker_registry: dict[str, CircuitBreakerAsync] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreakerAsync:
    """Return the singleton CircuitBreakerAsync for the given service name, creating it if absent."""
    if service_name not in _circuit_breaker_registry:
        config = (
            CIRCUIT_BREAKER_CONFIG.get(service_name)
            or CIRCUIT_BREAKER_CONFIG["openai_api"]
        )
        _circuit_breaker_registry[service_name] = CircuitBreakerAsync(**config)
    return _circuit_breaker_registry[service_name]


async def get_circuit_breaker_async(service_name: str) -> CircuitBreakerAsync:
    """Async accessor — return the singleton ``CircuitBreakerAsync`` for *service_name*.

    .. note::
        Declared ``async`` for API symmetry with the local
        ``get_async_circuit_breaker(name)`` in Search4Science so that async
        call sites can use a uniform ``await`` style.  The function itself
        performs no I/O.

    Equivalent to the sync :func:`get_circuit_breaker` with the same
    creation-on-first-access semantics.  Useful in async call sites that prefer
    a uniform ``await`` style throughout.

    Args:
        service_name: Logical name of the protected service (e.g. ``"openai_api"``).

    Returns:
        The ``CircuitBreakerAsync`` instance registered under *service_name*.
        A new instance is created if none exists yet.
    """
    return get_circuit_breaker(service_name)


def get_all_circuit_breakers() -> dict[str, CircuitBreakerAsync]:
    """Return a shallow copy of the internal circuit-breaker registry.

    Intended for test introspection and per-test isolation.  Callers receive
    a **copy** of the registry mapping so that mutations to the returned dict
    do not affect the live registry.

    Returns:
        A ``dict`` mapping service name → ``CircuitBreakerAsync`` instance for
        every breaker that has been created so far.
    """
    return dict(_circuit_breaker_registry)


def reset_all_circuit_breakers() -> None:
    """Clear all registered circuit breakers — for use in tests only."""
    _circuit_breaker_registry.clear()


async def reset_circuit_breaker(name: str) -> None:
    """Remove a single named circuit breaker from the registry.

    .. note::
        Declared ``async`` for API symmetry with the local
        ``reset_all_async_circuit_breakers()`` in Search4Science so that async
        test teardowns can ``await`` it without a special sync wrapper.  The
        function itself performs no I/O.

    After this call the named breaker is deregistered.  The next call to
    :func:`get_circuit_breaker` or :func:`get_circuit_breaker_async` for the
    same *name* will create a fresh instance.

    If *name* is not currently registered the function does nothing (no-op).

    Args:
        name: The service name used when the breaker was registered.
    """
    _circuit_breaker_registry.pop(name, None)
