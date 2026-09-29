"""Domain exception hierarchy for LLM provider operations in Vatuta."""

from typing import Any, Optional, Type

litellm_exceptions: Optional[Any]
try:
    import litellm.exceptions as _litellm_exceptions

    litellm_exceptions = _litellm_exceptions
except ImportError:  # pragma: no cover
    litellm_exceptions = None

_openai_error: Optional[Type[Exception]]
try:
    import openai

    _openai_error = openai.OpenAIError
except ImportError:  # pragma: no cover
    _openai_error = None


class VatutaError(Exception):
    """Base exception class for all Vatuta domain errors."""

    def __init__(self, message: str) -> None:
        """Initialize Vatuta error.

        Args:
            message: Descriptive error message.
        """
        super().__init__(message)
        self.message = message


class LLMError(VatutaError):
    """Base exception for all language model provider failures."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None) -> None:
        """Initialize LLM error.

        Args:
            message: Descriptive error message.
            original_exception: Underlying exception causing this failure, if any.
        """
        super().__init__(message)
        self.original_exception = original_exception


class LLMConfigurationError(LLMError):
    """Raised when an LLM backend configuration is malformed or invalid."""


class LLMAuthenticationError(LLMError):
    """Raised when provider credentials or authentication fails."""


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits or quotas are exceeded (HTTP 429)."""


class LLMConnectionError(LLMError):
    """Raised when connection or network communication with provider fails."""


class LLMBadRequestError(LLMError):
    """Raised when request payload or parameters are invalid for the provider."""


def map_litellm_exception(exc: Exception) -> LLMError:
    """Map a LiteLLM exception to an appropriate domain LLMError subclass.

    Args:
        exc: Exception caught from a LiteLLM invocation.

    Returns:
        Mapped domain LLMError instance with clear diagnostic context.
    """
    if litellm_exceptions is not None:
        if isinstance(exc, litellm_exceptions.AuthenticationError):
            return LLMAuthenticationError(
                f"Authentication failed: {exc}",
                original_exception=exc,
            )
        if isinstance(exc, litellm_exceptions.RateLimitError):
            return LLMRateLimitError(
                f"Rate limit exceeded: {exc}",
                original_exception=exc,
            )
        if isinstance(exc, litellm_exceptions.APIConnectionError):
            return LLMConnectionError(
                f"Connection error: {exc}",
                original_exception=exc,
            )
        if isinstance(exc, litellm_exceptions.BadRequestError):
            return LLMBadRequestError(
                f"Bad request: {exc}",
                original_exception=exc,
            )
        if getattr(type(exc), "__module__", "").startswith("litellm") or (
            _openai_error is not None and isinstance(exc, _openai_error)
        ):
            return LLMError(
                f"LLM execution error: {exc}",
                original_exception=exc,
            )

    return LLMError(f"Unexpected LLM failure: {exc}", original_exception=exc)
