"""LLM Provider domain module for Vatuta.

Provides centralized, provider-independent language model abstractions, lifecycle
management, and error handling via LiteLLM and DSPy.
"""

from src.llm.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    VatutaError,
    map_litellm_exception,
)
from src.llm.provider import LLMProviderManager

__all__ = [
    "LLMProviderManager",
    "VatutaError",
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMBadRequestError",
    "map_litellm_exception",
]
