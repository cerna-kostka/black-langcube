import logging
logger = logging.getLogger(__name__)

import time
from pydantic.v1 import ValidationError
from langchain.schema import OutputParserException
import openai
from langchain_community.callbacks import get_openai_callback


def robust_invoke(chain, extra_input=None, max_retries=3, backoff_factor=65):
    """
    A robust function to invoke a LangChain chain, handling:
      - OutputParserException
      - ValidationError
      - openai.error.RateLimitError (with exponential backoff)
      - openai.error.OpenAIError
    and returning the required output or a dictionary with 'error' key.
    
    :param chain: A LangChain pipeline, e.g. prompt | llm | parser
    :param input_data: Dictionary of inputs to pass to chain.invoke()
    :param max_retries: Maximum attempts to retry on rate-limit errors
    :param backoff_factor: Simple linear backoff (sleep time is backoff_factor * attempt)
    :return: result on success or {'error': ...} on failure
    """

    # Initialize empty tokens dict for error cases
    empty_tokens = {
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_price": 0,
    }

    for attempt in range(max_retries):
        try:
            with get_openai_callback() as cb:
                result = chain.invoke(extra_input)

            tokens = {
                "tokens_in": cb.prompt_tokens,
                "tokens_out": cb.completion_tokens,
                "tokens_price": cb.total_cost,
            }
            
            return result, tokens

        except (OutputParserException, ValidationError) as e:
            # Return parser/validation errors directly
            return {"error": str(e)}, empty_tokens

        except openai.RateLimitError as e:
            # Handle rate-limit by retrying with exponential backoff
            logger.warning(f"Rate limit error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return {"error": f"Rate limit error after {max_retries} attempts: {str(e)}"}, empty_tokens

        except openai.OpenAIError as e:
            # Any other OpenAI-specific error
            return {"error": f"OpenAI error: {str(e)}"}, empty_tokens

    # If we exit the loop for any reason (unlikely without returning), handle gracefully
    return {"error": "Unknown error or maximum retries reached without success."}, empty_tokens

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
