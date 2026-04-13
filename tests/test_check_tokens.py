"""
Unit tests for llm_modules/check_tokens.py.

Verifies:
- The module-level _ENCODING constant is a tiktoken.Encoding instance.
- num_tokens_from_string returns the correct token count for a known input.
"""

import sys
from pathlib import Path

import tiktoken

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from black_langcube.llm_modules.check_tokens import (  # noqa: E402
    _ENCODING,
    num_tokens_from_string,
)


class TestEncodingConstant:
    def test_encoding_is_tiktoken_encoding_instance(self):
        assert isinstance(_ENCODING, tiktoken.Encoding)


class TestNumTokensFromString:
    def test_known_token_count(self):
        # "hello world" encodes to 2 tokens with gpt-4o (cl100k_base / o200k_base)
        assert num_tokens_from_string("hello world") == 2

    def test_empty_string_returns_zero(self):
        assert num_tokens_from_string("") == 0

    def test_return_type_is_int(self):
        assert isinstance(num_tokens_from_string("test"), int)
