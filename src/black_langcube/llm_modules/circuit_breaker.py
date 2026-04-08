"""
Synchronous circuit breaker for protecting LLM API calls.

Implements a three-state machine (CLOSED → OPEN → HALF_OPEN → CLOSED) using
a threading.Lock for thread-safe state transitions.

Configuration is read at import time from environment variables:
  CB_FAILURE_THRESHOLD  — consecutive failures before opening (default: 5)
  CB_RECOVERY_TIMEOUT   — seconds to wait before probing in HALF_OPEN (default: 60)
"""

import os
import threading
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


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: int) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._probe_in_flight: bool = False
        self._lock = threading.Lock()

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

    def record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._failure_count = self._failure_threshold
                self._probe_in_flight = False
                self._transition(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._transition(CircuitState.OPEN)

    def record_success(self) -> None:
        with self._lock:
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
        Context manager that gates a single synchronous call through the breaker.

        Args:
            service_errors: Exception classes that count as service failures and
                increment the failure counter.  When *None* (default) the
                module-level ``SERVICE_ERRORS`` tuple is used.

        Usage::

            with circuit_breaker.call():
                result = chain.invoke(...)
        """
        return _SyncBreakerContext(self, service_errors)


class _SyncBreakerContext:
    def __init__(
        self,
        breaker: CircuitBreaker,
        service_errors: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self._breaker = breaker
        self._is_probe = False
        self._service_errors = (
            service_errors if service_errors is not None else SERVICE_ERRORS
        )

    def __enter__(self):
        with self._breaker._lock:
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._breaker.record_success()
        elif self._service_errors and issubclass(exc_type, self._service_errors):
            self._breaker.record_failure()
        elif self._is_probe:
            # Other error (e.g. RateLimitError) during a HALF_OPEN probe:
            # clear the probe flag so future probes are not blocked.
            with self._breaker._lock:
                self._breaker._probe_in_flight = False
        # RateLimitError and other errors are not recorded — let them propagate
        return False  # never suppress the exception


# ------------------------------------------------------------------
# Module-level registry
# ------------------------------------------------------------------

_circuit_breaker_registry: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Return the singleton CircuitBreaker for the given service name, creating it if absent."""
    if service_name not in _circuit_breaker_registry:
        config = (
            CIRCUIT_BREAKER_CONFIG.get(service_name)
            or CIRCUIT_BREAKER_CONFIG["openai_api"]
        )
        _circuit_breaker_registry[service_name] = CircuitBreaker(**config)
    return _circuit_breaker_registry[service_name]


def reset_all_circuit_breakers() -> None:
    """Clear all registered circuit breakers — for use in tests only."""
    _circuit_breaker_registry.clear()
