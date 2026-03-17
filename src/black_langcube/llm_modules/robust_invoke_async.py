"""
Async version of robust_invoke for LangChain chain invocation with retry logic.

This module provides async-safe invocation of LangChain chains with:
- Exponential backoff for rate limiting
- OpenAI API error handling
- Async sleep for non-blocking delays
"""

import asyncio
import logging

import openai
from langchain_core.exceptions import OutputParserException
from langchain_community.callbacks import get_openai_callback
from pydantic import ValidationError

logger = logging.getLogger(__name__)


async def robust_invoke_async(
    chain, extra_input=None, max_retries=3, backoff_factor=65
):
    """
    Async version of robust_invoke for LangChain chain invocation.

    Handles:
      - OutputParserException
      - ValidationError
      - openai.RateLimitError (with exponential backoff using asyncio.sleep)
      - openai.OpenAIError

    Args:
        chain: A LangChain pipeline, e.g. prompt | llm | parser
        extra_input: Dictionary of inputs to pass to chain.ainvoke()
        max_retries: Maximum attempts to retry on rate-limit errors
        backoff_factor: Simple linear backoff (sleep time is backoff_factor * attempt)

    Returns:
        tuple: (result, tokens_dict) on success or ({'error': ...}, empty_tokens) on failure
    """

    empty_tokens = {
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_price": 0,
    }

    for attempt in range(max_retries):
        try:
            with get_openai_callback() as cb:
                if hasattr(chain, "ainvoke"):
                    result = await chain.ainvoke(extra_input)
                else:
                    # Fallback to sync invoke in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: chain.invoke(extra_input)
                    )

            tokens = {
                "tokens_in": cb.prompt_tokens,
                "tokens_out": cb.completion_tokens,
                "tokens_price": cb.total_cost,
            }
            return result, tokens

        except (OutputParserException, ValidationError) as e:
            return {"error": str(e)}, empty_tokens

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                return {
                    "error": f"Rate limit error after {max_retries} attempts: {str(e)}"
                }, empty_tokens

        except openai.OpenAIError as e:
            return {"error": f"OpenAI error: {str(e)}"}, empty_tokens

    return {
        "error": "Unknown error or maximum retries reached without success."
    }, empty_tokens
