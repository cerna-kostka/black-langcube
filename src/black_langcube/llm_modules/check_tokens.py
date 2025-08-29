"""
This script provides a function to calculate the number of tokens in a given text string using the tiktoken library.

Functions:
    num_tokens_from_string(string: str) -> int:
        Returns the number of tokens in a text string.

Uncomment the print statements or the function calls at the bottom of the script for debugging purposes.
"""

import tiktoken
from black_langcube.llm_modules.llm_model import model_name_low


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o") #fixed to "gpt-4o" due to not available automatic mapping for model 4.1, besides, opanAI claims 4o and 4.1 have the same tokeniser
    num_tokens = len(encoding.encode(string))
    #print(f"Number of tokens in string: {num_tokens}") # Uncomment for debugging
    return num_tokens
