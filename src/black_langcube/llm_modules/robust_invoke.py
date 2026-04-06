import logging
import time

import openai
from langchain_core.exceptions import OutputParserException
from langchain_community.callbacks import get_openai_callback
from pydantic import ValidationError

from black_langcube.llm_modules.circuit_breaker import (
    CircuitBreakerOpenError,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

_SERVICE_NAME = "openai_api"


def robust_invoke(chain, extra_input=None, max_retries=3, backoff_factor=65):
    """
    A robust function to invoke a LangChain chain, handling:
      - OutputParserException
      - ValidationError
      - openai.RateLimitError (with linear backoff; does not increment the
        circuit-breaker failure counter)
      - openai.APIConnectionError / openai.APITimeoutError / openai.InternalServerError
        (recorded by the circuit breaker; circuit opens after CB_FAILURE_THRESHOLD
        consecutive such failures)
      - openai.OpenAIError (other OpenAI errors — not recorded by the circuit breaker)
      - CircuitBreakerOpenError (circuit is OPEN; returns error dict immediately without
        invoking the chain)
    and returning the required output or a dictionary with 'error' key.

    :param chain: A LangChain pipeline, e.g. prompt | llm | parser
    :param extra_input: Dictionary of inputs to pass to chain.invoke()
    :param max_retries: Maximum attempts to retry on rate-limit errors
    :param backoff_factor: Simple linear backoff (sleep time is backoff_factor * attempt)
    :return: result on success or {'error': ...} on failure
    """

    empty_tokens = {
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_price": 0,
    }

    circuit_breaker = get_circuit_breaker(_SERVICE_NAME)

    for attempt in range(max_retries):
        try:
            with circuit_breaker.call():
                with get_openai_callback() as cb:
                    result = chain.invoke(extra_input)

            tokens = {
                "tokens_in": cb.prompt_tokens,
                "tokens_out": cb.completion_tokens,
                "tokens_price": cb.total_cost,
            }

            return result, tokens

        except CircuitBreakerOpenError:
            return {"error": f"Circuit breaker open: {_SERVICE_NAME}"}, empty_tokens

        except (OutputParserException, ValidationError) as e:
            return {"error": str(e)}, empty_tokens

        except openai.RateLimitError as e:
            # Rate-limit errors are transient — do NOT record a circuit failure.
            logger.warning(f"Rate limit error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * attempt
                logger.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return {
                    "error": f"Rate limit error after {max_retries} attempts: {str(e)}"
                }, empty_tokens

        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as e:
            # These errors indicate a structural service failure — the circuit breaker
            # context manager already recorded the failure in __exit__; return immediately.
            return {"error": f"OpenAI error: {str(e)}"}, empty_tokens

        except openai.OpenAIError as e:
            return {"error": f"OpenAI error: {str(e)}"}, empty_tokens

    return {
        "error": "Unknown error or maximum retries reached without success."
    }, empty_tokens


def split_into_chunks(text, chunk_size=90000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


# implement functions for chunking too long input?
def chunks_robust_invoke(chunks):
    pass
