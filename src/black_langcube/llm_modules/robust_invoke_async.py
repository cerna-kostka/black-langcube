"""
Async version of robust_invoke for LangChain chain invocation with retry logic.

This module provides async-safe invocation of LangChain chains with:
- Linear backoff for provider rate-limit errors
- Provider-agnostic error handling (OpenAI, Gemini, Mistral, and generic fallback)
- Async sleep for non-blocking delays
- Async circuit breaker to stop cascading failures on a fully unavailable API

Token tracking uses ``AIMessage.usage_metadata`` (LangChain ≥ 0.2 standard)
with a zero-value fallback when no usage data is available.
"""

import asyncio
import logging
import os

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from black_langcube.llm_modules._token_utils import (
    EMPTY_TOKENS,
    _extract_token_usage,
    get_service_errors,
    is_rate_limit_error,
)
from black_langcube.llm_modules.circuit_breaker_async import (
    CircuitBreakerOpenError,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

#: Default service name derived from the ``PROVIDER`` env var at import time.
#: Individual calls may override this via the ``provider`` parameter.
_SERVICE_NAME = f"{os.getenv('PROVIDER', 'openai').lower()}_api"


async def robust_invoke_async(
    chain, extra_input=None, max_retries=3, backoff_factor=65, provider=None
):
    """
    Async version of robust_invoke for LangChain chain invocation.

    Handles:
      - ``OutputParserException`` / ``ValidationError`` — returned as
        ``{"error": ...}`` immediately (no retry).
      - Provider rate-limit errors (backoff only using ``asyncio.sleep``; does
        not increment the circuit-breaker failure counter):

        * OpenAI: ``openai.RateLimitError``
        * Gemini: ``google.api_core.exceptions.ResourceExhausted``
        * Mistral: ``mistralai.exceptions.MistralAPIException`` (HTTP 429)

      - Provider service-failure errors (recorded by the circuit breaker;
        circuit opens after ``CB_FAILURE_THRESHOLD`` consecutive failures):

        * OpenAI: ``APIConnectionError``, ``APITimeoutError``,
          ``InternalServerError``
        * Gemini: ``ServiceUnavailable``, ``DeadlineExceeded``,
          ``InternalServerError``
        * Mistral: ``MistralConnectionException`` (when available)

      - ``CircuitBreakerOpenError`` — circuit is OPEN; returns error dict
        immediately without invoking the chain.
      - Any other ``Exception`` — caught as a generic fallback and returned
        as ``{"error": ...}`` to prevent unhandled propagation; the exception
        type is logged at ``ERROR`` level to aid diagnostics.

    Token tracking uses ``AIMessage.usage_metadata`` (LangChain ≥ 0.2 standard,
    supported by OpenAI, Gemini, Anthropic, and Mistral integrations) with a
    zero-value fallback when no usage data is available.  When a callback object
    is passed to ``_extract_token_usage``, its ``total_cost`` is used for pricing
    even when ``usage_metadata`` is already present.

    Args:
        chain: A LangChain pipeline, e.g. ``prompt | llm | parser``
        extra_input: Dictionary of inputs to pass to ``chain.ainvoke()``
        max_retries: Maximum attempts to retry on rate-limit errors
        backoff_factor: Simple linear backoff (sleep time is
            ``backoff_factor * attempt``)
        provider: Override the active provider name (e.g. ``"gemini"``,
            ``"mistral"``, ``"openai"``).  When *None* (default) the ``PROVIDER``
            environment variable is used, falling back to ``"openai"``.  Setting
            this per-call allows different nodes in the same process to target
            different LLM providers with independent circuit-breaker state and
            correct service-error classification.

    Returns:
        tuple: ``(result, tokens_dict)`` on success or
            ``({"error": ...}, empty_tokens)`` on failure
    """

    resolved_provider = (provider or os.getenv("PROVIDER", "openai")).lower()
    service_name = f"{resolved_provider}_api"
    service_errors = get_service_errors(resolved_provider)
    empty_tokens = dict(EMPTY_TOKENS)
    circuit_breaker = get_circuit_breaker(service_name)

    for attempt in range(max_retries):
        try:
            async with circuit_breaker.call(service_errors=service_errors):
                if hasattr(chain, "ainvoke"):
                    result = await chain.ainvoke(extra_input)
                else:
                    # Fallback to sync invoke in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: chain.invoke(extra_input)
                    )

            tokens = _extract_token_usage(result)
            return result, tokens

        except CircuitBreakerOpenError:
            return {"error": f"Circuit breaker open: {service_name}"}, empty_tokens

        except (OutputParserException, ValidationError) as e:
            return {"error": str(e)}, empty_tokens

        except Exception as e:
            if is_rate_limit_error(e):
                # Rate-limit errors are transient — do NOT record a circuit failure.
                logger.warning("Rate limit error: %s", e)
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * attempt
                    logger.debug("Retrying in %s seconds...", sleep_time)
                    await asyncio.sleep(sleep_time)
                else:
                    return {
                        "error": f"Rate limit error after {max_retries} attempts: {str(e)}"
                    }, empty_tokens
            else:
                # Service errors: circuit breaker already recorded the failure in
                # __aexit__ when the exception propagated through circuit_breaker.call().
                # Unknown provider errors: logged for diagnostics.
                logger.error(
                    "Provider error (%s) during chain invocation: %s",
                    type(e).__name__,
                    e,
                )
                return {
                    "error": f"Provider error ({type(e).__name__}): {str(e)}"
                }, empty_tokens

    return {
        "error": "Unknown error or maximum retries reached without success."
    }, empty_tokens
