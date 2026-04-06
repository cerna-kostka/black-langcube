"""
Provider-agnostic LLM factory for black_langcube.

Supports multiple LLM providers (OpenAI, Gemini, Mistral) selected at runtime via
environment variables.  All API keys are wrapped in ``pydantic.SecretStr``
to prevent accidental leakage in logs and tracebacks.

Environment variables
---------------------
PROVIDER
    Default provider for all tiers (e.g. ``openai``, ``gemini``).
    Defaults to ``openai``.

<STEP>_PROVIDER
    Per-step provider override.  ``<STEP>`` is one of ``LOW``, ``HIGH``,
    ``ANALYST``, ``OUTLINE``, ``TEXT``, ``CHECK_TITLE``, ``TITLE_ABSTRACT``.

OPENAI_API_KEY / GEMINI_API_KEY / MISTRAL_API_KEY
    API keys for each provider.  Only the key for the active provider must
    be set.

<PROVIDER>_MODEL_<TIER>
    Override the model name for a specific provider/tier combination, e.g.
    ``OPENAI_MODEL_LOW=gpt-4o-mini``.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    # Extend when further providers are added


class ModelTier(str, Enum):
    LOW = "low"
    HIGH = "high"
    ANALYST = "analyst"
    OUTLINE = "outline"
    TEXT = "text"
    CHECK_TITLE = "check_title"
    TITLE_ABSTRACT = "title_abstract"


# ---------------------------------------------------------------------------
# Model registry — all names sourced from environment variables with defaults
# ---------------------------------------------------------------------------
# NOTE: os.getenv() calls below are evaluated once at module import time.
# Environment variable overrides (e.g. OPENAI_MODEL_LOW) must be set before
# this module is first imported to take effect.
MODEL_REGISTRY: dict[LLMProvider, dict[ModelTier, str]] = {
    LLMProvider.OPENAI: {
        ModelTier.LOW: os.getenv("OPENAI_MODEL_LOW", "gpt-4o-mini"),
        ModelTier.HIGH: os.getenv("OPENAI_MODEL_HIGH", "gpt-4.1"),
        ModelTier.ANALYST: os.getenv("OPENAI_MODEL_ANALYST", "gpt-4.1"),
        ModelTier.OUTLINE: os.getenv("OPENAI_MODEL_OUTLINE", "gpt-4.1"),
        ModelTier.TEXT: os.getenv("OPENAI_MODEL_TEXT", "gpt-4.1"),
        ModelTier.CHECK_TITLE: os.getenv("OPENAI_MODEL_CHECK_TITLE", "gpt-4.1"),
        ModelTier.TITLE_ABSTRACT: os.getenv("OPENAI_MODEL_TITLE_ABSTRACT", "gpt-4.1"),
    },
    LLMProvider.GEMINI: {
        ModelTier.LOW: os.getenv("GEMINI_MODEL_LOW", "gemini-2.5-flash"),
        ModelTier.HIGH: os.getenv("GEMINI_MODEL_HIGH", "gemini-2.5-pro"),
        ModelTier.ANALYST: os.getenv("GEMINI_MODEL_ANALYST", "gemini-2.5-pro"),
        ModelTier.OUTLINE: os.getenv("GEMINI_MODEL_OUTLINE", "gemini-2.5-pro"),
        ModelTier.TEXT: os.getenv("GEMINI_MODEL_TEXT", "gemini-2.5-pro"),
        ModelTier.CHECK_TITLE: os.getenv(
            "GEMINI_MODEL_CHECK_TITLE", "gemini-2.5-flash"
        ),
        ModelTier.TITLE_ABSTRACT: os.getenv(
            "GEMINI_MODEL_TITLE_ABSTRACT", "gemini-2.5-flash"
        ),
    },
    LLMProvider.MISTRAL: {
        ModelTier.LOW: os.getenv("MISTRAL_MODEL_LOW", "mistral-small-latest"),
        ModelTier.HIGH: os.getenv("MISTRAL_MODEL_HIGH", "mistral-large-latest"),
        ModelTier.ANALYST: os.getenv("MISTRAL_MODEL_ANALYST", "mistral-large-latest"),
        ModelTier.OUTLINE: os.getenv("MISTRAL_MODEL_OUTLINE", "mistral-large-latest"),
        ModelTier.TEXT: os.getenv("MISTRAL_MODEL_TEXT", "mistral-large-latest"),
        ModelTier.CHECK_TITLE: os.getenv(
            "MISTRAL_MODEL_CHECK_TITLE", "mistral-small-latest"
        ),
        ModelTier.TITLE_ABSTRACT: os.getenv(
            "MISTRAL_MODEL_TITLE_ABSTRACT", "mistral-small-latest"
        ),
    },
}

# ---------------------------------------------------------------------------
# API key helpers — SecretStr prevents leakage in logs / repr / tracebacks
# ---------------------------------------------------------------------------


def _load_secret(env_var: str) -> SecretStr | None:
    value = os.getenv(env_var)
    return SecretStr(value) if value else None


OPENAI_API_KEY: SecretStr | None = _load_secret("OPENAI_API_KEY")
GEMINI_API_KEY: SecretStr | None = _load_secret("GEMINI_API_KEY")
MISTRAL_API_KEY: SecretStr | None = _load_secret("MISTRAL_API_KEY")

# ---------------------------------------------------------------------------
# Active provider resolution
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER: LLMProvider = LLMProvider(
    os.getenv("PROVIDER", LLMProvider.OPENAI.value).lower()
)


def _resolve_provider(step_env_var: str) -> LLMProvider:
    raw = os.getenv(step_env_var, DEFAULT_PROVIDER.value).lower()
    return LLMProvider(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm(provider: LLMProvider, tier: ModelTier) -> BaseChatModel:
    """Return a ``BaseChatModel`` for *provider* at *tier*.

    Provider-specific packages are imported lazily so that only the package
    for the active provider needs to be installed.

    Raises:
        ValueError: For an unrecognised *provider*.
    """
    model_name = MODEL_REGISTRY[provider][tier]
    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": model_name}
        if OPENAI_API_KEY:
            kwargs["api_key"] = OPENAI_API_KEY.get_secret_value()
        return ChatOpenAI(**kwargs)
    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": model_name}
        if GEMINI_API_KEY:
            kwargs["google_api_key"] = GEMINI_API_KEY.get_secret_value()
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == LLMProvider.MISTRAL:
        from langchain_mistralai import ChatMistralAI

        kwargs = {"model": model_name}
        if MISTRAL_API_KEY:
            kwargs["mistral_api_key"] = MISTRAL_API_KEY.get_secret_value()
        return ChatMistralAI(**kwargs)
    raise ValueError(f"Unsupported provider: {provider!r}")


# ---------------------------------------------------------------------------
# Backward-compatible public aliases
# ---------------------------------------------------------------------------


def get_llm_low() -> BaseChatModel:
    return create_llm(_resolve_provider("LOW_PROVIDER"), ModelTier.LOW)


def get_llm_high() -> BaseChatModel:
    return create_llm(_resolve_provider("HIGH_PROVIDER"), ModelTier.HIGH)


def default_llm() -> BaseChatModel:
    return get_llm_low()


def llm_analyst() -> BaseChatModel:
    return create_llm(_resolve_provider("ANALYST_PROVIDER"), ModelTier.ANALYST)


def llm_outline() -> BaseChatModel:
    return create_llm(_resolve_provider("OUTLINE_PROVIDER"), ModelTier.OUTLINE)


def llm_text() -> BaseChatModel:
    return create_llm(_resolve_provider("TEXT_PROVIDER"), ModelTier.TEXT)


def llm_check_title() -> BaseChatModel:
    return create_llm(_resolve_provider("CHECK_TITLE_PROVIDER"), ModelTier.CHECK_TITLE)


def llm_title_abstract() -> BaseChatModel:
    return create_llm(
        _resolve_provider("TITLE_ABSTRACT_PROVIDER"), ModelTier.TITLE_ABSTRACT
    )
