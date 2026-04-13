"""
This script provides a function to calculate the number of tokens in a given text string using the tiktoken library.

Functions:
    num_tokens_from_string(string: str) -> int:
        Returns the number of tokens in a text string.

Uncomment the print statements or the function calls at the bottom of the script for debugging purposes.
"""

import tiktoken

# Fixed to "gpt-4o": no automatic mapping exists for model 4.1; OpenAI states
# that gpt-4o and gpt-4.1 share the same tokeniser.
_ENCODING = tiktoken.encoding_for_model("gpt-4o")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(_ENCODING.encode(string))
    # print(f"Number of tokens in string: {num_tokens}") # Uncomment for debugging
    return num_tokens
