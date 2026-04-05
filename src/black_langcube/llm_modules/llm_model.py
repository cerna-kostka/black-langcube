"""
This module initializes and configures language model instances using the ChatOpenAI class from the langchain_openai package.
It loads environment variables from a local .env file to securely manage credentials and settings.
Two model names are specified: one for a lightweight model and one for a high-performance model, with recommendations to periodically check for updates in OpenAI's available models.
The module provides lazy factory functions for model access so that ChatOpenAI instances are only created on first use, not at import time.

    - langchain_openai.ChatOpenAI: Interface for OpenAI's chat-based language models.
    - dotenv.load_dotenv, dotenv.find_dotenv: Utilities for loading environment variables from a .env file.

    - model_name_low (str): Name of the lightweight language model (default: "gpt-4o-mini").
    - model_name_high (str): Name of the high-performance language model (default: "gpt-4.1").
    - get_llm_low() -> ChatOpenAI: Lazy singleton returning the lightweight model instance.
    - get_llm_high() -> ChatOpenAI: Lazy singleton returning the high-performance model instance.
    - default_llm, llm_analyst, llm_outline, llm_text, llm_check_title, llm_title_abstract: Lazy factory aliases for the models, used for specialized tasks.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())  # read local .env file

model_name_low = "gpt-4o-mini"  # this by default should be "gpt-4o-mini-2024-07-18" - changes in openAI available model policies should be checked periodically
model_name_high = "gpt-4.1"  # this by default should be "gpt-4o-2024-08-06" - changes in openAI available model policies should be checked periodically

# Lazy singletons — instantiated on first call, not at import time.
_llm_low: ChatOpenAI | None = None
_llm_high: ChatOpenAI | None = None


def get_llm_low() -> ChatOpenAI:
    global _llm_low
    if _llm_low is None:
        _llm_low = ChatOpenAI(model=model_name_low)
    return _llm_low


def get_llm_high() -> ChatOpenAI:
    global _llm_high
    if _llm_high is None:
        _llm_high = ChatOpenAI(model=model_name_high)
    return _llm_high


# Lazy factory aliases for the models:


def default_llm() -> ChatOpenAI:
    # the default is used at:
    # llm_IsLanguageNode
    # llm_TranslateQuestionNode
    # ...
    return get_llm_low()


def llm_analyst() -> ChatOpenAI:
    return get_llm_high()


def llm_outline() -> ChatOpenAI:
    return get_llm_high()


def llm_text() -> ChatOpenAI:
    return get_llm_high()


def llm_check_title() -> ChatOpenAI:
    return get_llm_high()


def llm_title_abstract() -> ChatOpenAI:
    return get_llm_high()
