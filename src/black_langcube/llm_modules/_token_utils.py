"""Provider-agnostic utilities for token usage extraction and error classification.

Supports optional provider packages: openai, google-api-core, mistralai.
Each package is imported with a guard so this module remains importable when
only a subset of provider packages is installed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider-specific imports (guarded)
# ---------------------------------------------------------------------------

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]

try:
    import google.api_core.exceptions as _google_exc
except ImportError:
    _google_exc = None  # type: ignore[assignment]

try:
    import mistralai.exceptions as _mistral_exc
except ImportError:
    _mistral_exc = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Per-provider service error registry
#
# Maps a normalised provider name (lower-case, e.g. "openai") to the tuple of
# exception classes that represent structural service failures for that provider.
# Only entries for installed packages are populated.
# ---------------------------------------------------------------------------

_ALL_PROVIDER_SERVICE_ERRORS: dict[str, tuple[type[BaseException], ...]] = {}

if _openai is not None:
    _ALL_PROVIDER_SERVICE_ERRORS["openai"] = (
        _openai.APIConnectionError,
        _openai.APITimeoutError,
        _openai.InternalServerError,
    )

if _google_exc is not None:
    _ALL_PROVIDER_SERVICE_ERRORS["gemini"] = (
        _google_exc.ServiceUnavailable,
        _google_exc.DeadlineExceeded,
        _google_exc.InternalServerError,
    )

if _mistral_exc is not None and hasattr(_mistral_exc, "MistralConnectionException"):
    _ALL_PROVIDER_SERVICE_ERRORS["mistral"] = (
        _mistral_exc.MistralConnectionException,  # type: ignore[attr-defined]
    )


def get_service_errors(provider: str) -> tuple[type[BaseException], ...]:
    """Return service-failure exception classes for *provider*.

    Returns an empty tuple for unknown or uninstalled providers so callers
    can safely pass the result directly to ``isinstance`` / ``issubclass``.
    """
    return _ALL_PROVIDER_SERVICE_ERRORS.get(provider.lower(), ())


#: Module-level tuple for the active provider (derived from the ``PROVIDER``
#: env var at import time).  Used as the default by both circuit breakers.
PROVIDER_SERVICE_ERRORS: tuple[type[BaseException], ...] = get_service_errors(
    os.getenv("PROVIDER", "openai")
)


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is a provider rate-limit (HTTP 429) error.

    Checks all installed providers in order:

    * **OpenAI**: ``openai.RateLimitError``
    * **Gemini**: ``google.api_core.exceptions.ResourceExhausted``
    * **Mistral**: ``mistralai.exceptions.MistralAPIException`` with
      ``http_status`` / ``status_code`` == 429

    Returns ``False`` for any unrecognised exception type.
    """
    if _openai is not None and isinstance(exc, _openai.RateLimitError):
        return True
    if _google_exc is not None and isinstance(exc, _google_exc.ResourceExhausted):
        return True
    if _mistral_exc is not None and hasattr(_mistral_exc, "MistralAPIException"):
        if isinstance(exc, _mistral_exc.MistralAPIException):  # type: ignore[attr-defined]
            status = getattr(exc, "http_status", None) or getattr(
                exc, "status_code", None
            )
            return status == 429
    return False


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------

#: Shared zero-value token dict.  Callers should copy it (``dict(EMPTY_TOKENS)``)
#: when returning to untrusted code that might mutate the result.
EMPTY_TOKENS: dict[str, int | float] = {
    "tokens_in": 0,
    "tokens_out": 0,
    "tokens_price": 0,
}


def _extract_token_usage(result: object, cb: object = None) -> dict:
    """Extract token usage from a LangChain chain result.

    Token counts and price are resolved independently so that the richer
    ``usage_metadata`` token counts can coexist with a callback's cost data:

    **Token counts** (in priority order):

    1. ``AIMessage.usage_metadata`` — the cross-provider standard populated by
       LangChain ≥ 0.2 for OpenAI, Gemini, Anthropic, and Mistral.
       Keys: ``input_tokens``, ``output_tokens``.
    2. OpenAI callback object (*cb*) — ``cb.prompt_tokens`` /
       ``cb.completion_tokens``.
    3. Zero fallback with a ``DEBUG``-level log.

    **Price** (in priority order, independent of token-count source):

    1. OpenAI callback object (*cb*) — ``cb.total_cost``.  Used even when
       ``usage_metadata`` is present so that OpenAI cost data is never silently
       dropped.
    2. ``result.response_metadata`` — checks ``"cost"`` and ``"total_cost"``
       keys for providers that expose pricing in their response metadata.
    3. Zero fallback.

    Args:
        result: The return value of ``chain.invoke()`` / ``chain.ainvoke()``.
        cb: Optional callback object (e.g. from ``get_openai_callback()``).
            Its ``total_cost`` is used for pricing even when *result* already
            carries ``usage_metadata``.

    Returns:
        ``dict`` with keys ``tokens_in``, ``tokens_out``, ``tokens_price``.
    """
    # -- Token counts --
    tokens_in: int = 0
    tokens_out: int = 0
    _has_token_source = False

    usage = getattr(result, "usage_metadata", None)
    if usage and isinstance(usage, dict):
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        _has_token_source = True
    elif cb is not None:
        tokens_in = getattr(cb, "prompt_tokens", 0)
        tokens_out = getattr(cb, "completion_tokens", 0)
        _has_token_source = True

    if not _has_token_source:
        logger.debug(
            "Token usage data unavailable; returning zero-values. Result type: %s",
            type(result).__name__,
        )

    # -- Price (independent of token-count source) --
    tokens_price: int | float = 0
    if cb is not None:
        tokens_price = getattr(cb, "total_cost", 0) or 0
    else:
        response_metadata = getattr(result, "response_metadata", None)
        if isinstance(response_metadata, dict):
            _cost = response_metadata.get("cost")
            if _cost is None:
                _cost = response_metadata.get("total_cost")
            if _cost is not None:
                tokens_price = _cost

    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_price": tokens_price,
    }
