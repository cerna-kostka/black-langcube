"""
Configuration validation for black_langcube.

Provides fail-fast validation of required environment variables at application
startup, before any pipeline execution begins.
"""

import os

from pydantic import SecretStr

REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]


class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid."""


def get_api_key(env_var: str) -> SecretStr:
    """Read an environment variable and return it as a SecretStr.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        The value wrapped in SecretStr to prevent accidental leakage.

    Raises:
        ConfigurationError: If the variable is absent or empty.
    """
    value = os.getenv(env_var)
    if not value:
        raise ConfigurationError(
            f"Required environment variable '{env_var}' is not set or is empty."
        )
    return SecretStr(value)


def validate_config() -> None:
    """Validate all required environment variables.

    Call this at application startup before any pipeline execution.
    All missing variables are collected and reported in a single
    ConfigurationError so that the user sees every problem at once.

    Raises:
        ConfigurationError: With a descriptive message listing every missing
            or empty required environment variable.
    """
    errors: list[str] = []
    for env_var in REQUIRED_ENV_VARS:
        try:
            get_api_key(env_var)
        except ConfigurationError as exc:
            errors.append(str(exc))

    if errors:
        raise ConfigurationError("\n".join(errors))
