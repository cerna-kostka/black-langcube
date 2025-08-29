"""
This module provides the TokenCounter class for aggregating token-related statistics from a given state dictionary.

Classes:
    TokenCounter: Aggregates token input, output, and price values for specified token keys.

TokenCounter:
    Args:
        token_keys (list[str]): A list of keys representing tokens to be counted in the state dictionary.

    Methods:
        count_tokens(state: dict) -> dict:
            Calculates the total 'tokens_in', 'tokens_out', and 'tokens_price' for the specified token keys from the provided state dictionary.

            Args:
                state (dict): A dictionary containing token data, where each key corresponds to a token and maps to a dictionary with 'tokens_in', 'tokens_out', and 'tokens_price' values.

            Returns:
                dict: A dictionary with the total 'tokens_in', 'tokens_out', and 'tokens_price' summed across all specified token keys.
"""

import logging
logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self, token_keys: list[str]):
        self.token_keys = token_keys

    def count_tokens(self, state: dict) -> dict:

        logger.info(f"Counting tokens")

        total_tokens_in = sum(
            state.get(key, {}).get("tokens_in", 0) for key in self.token_keys
        )
        total_tokens_out = sum(
            state.get(key, {}).get("tokens_out", 0) for key in self.token_keys
        )
        total_tokens_price = sum(
            state.get(key, {}).get("tokens_price", 0) for key in self.token_keys
        )
        return {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "tokens_price": total_tokens_price,
        }