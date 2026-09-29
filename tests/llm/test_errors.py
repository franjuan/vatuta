"""Unit tests for domain LLM exception hierarchy and LiteLLM error mapping."""

import unittest

import litellm.exceptions as litellm_exceptions

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


class TestLLMErrors(unittest.TestCase):
    """Test suite for domain exceptions and mapping logic."""

    def test_inheritance_hierarchy(self) -> None:
        """Verify that all LLM errors inherit from LLMError and VatutaError."""
        subclasses = [
            LLMConfigurationError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMConnectionError,
            LLMBadRequestError,
        ]
        for cls in subclasses:
            err = cls("Test message")
            self.assertIsInstance(err, LLMError)
            self.assertIsInstance(err, VatutaError)
            self.assertEqual(err.message, "Test message")
            self.assertEqual(str(err), "Test message")

    def test_original_exception_retention(self) -> None:
        """Verify that original underlying exception is retained."""
        cause = ValueError("Root cause")
        err = LLMError("Wrapped error", original_exception=cause)
        self.assertEqual(err.original_exception, cause)

    def test_map_authentication_error(self) -> None:
        """Verify mapping of LiteLLM AuthenticationError."""
        mock_auth_exc = litellm_exceptions.AuthenticationError(
            message="Invalid API Key",
            llm_provider="gemini",
            model="gemini/gemini-2.5-flash",
        )
        mapped = map_litellm_exception(mock_auth_exc)
        self.assertIsInstance(mapped, LLMAuthenticationError)
        self.assertIn("Authentication failed", mapped.message)
        self.assertEqual(mapped.original_exception, mock_auth_exc)

    def test_map_rate_limit_error(self) -> None:
        """Verify mapping of LiteLLM RateLimitError."""
        mock_rate_exc = litellm_exceptions.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="gemini",
            model="gemini/gemini-2.5-flash",
        )
        mapped = map_litellm_exception(mock_rate_exc)
        self.assertIsInstance(mapped, LLMRateLimitError)
        self.assertIn("Rate limit exceeded", mapped.message)
        self.assertEqual(mapped.original_exception, mock_rate_exc)

    def test_map_connection_error(self) -> None:
        """Verify mapping of LiteLLM APIConnectionError."""
        mock_conn_exc = litellm_exceptions.APIConnectionError(
            message="Failed to connect to host",
            llm_provider="gemini",
            model="gemini/gemini-2.5-flash",
        )
        mapped = map_litellm_exception(mock_conn_exc)
        self.assertIsInstance(mapped, LLMConnectionError)
        self.assertIn("Connection error", mapped.message)
        self.assertEqual(mapped.original_exception, mock_conn_exc)

    def test_map_bad_request_error(self) -> None:
        """Verify mapping of LiteLLM BadRequestError."""
        mock_bad_req = litellm_exceptions.BadRequestError(
            message="Invalid parameters",
            llm_provider="gemini",
            model="gemini/gemini-2.5-flash",
        )
        mapped = map_litellm_exception(mock_bad_req)
        self.assertIsInstance(mapped, LLMBadRequestError)
        self.assertIn("Bad request", mapped.message)
        self.assertEqual(mapped.original_exception, mock_bad_req)

    def test_map_generic_litellm_error(self) -> None:
        """Verify mapping of generic LiteLLMError."""
        mock_litellm_err = litellm_exceptions.APIError(
            status_code=500,
            message="Upstream internal server error",
            llm_provider="gemini",
            model="gemini/gemini-2.5-flash",
        )
        mapped = map_litellm_exception(mock_litellm_err)
        self.assertIsInstance(mapped, LLMError)
        self.assertIn("LLM execution error", mapped.message)

    def test_map_unknown_exception(self) -> None:
        """Verify mapping of non-LiteLLM arbitrary exception."""
        unknown_exc = RuntimeError("System memory low")
        mapped = map_litellm_exception(unknown_exc)
        self.assertIsInstance(mapped, LLMError)
        self.assertIn("Unexpected LLM failure", mapped.message)
        self.assertEqual(mapped.original_exception, unknown_exc)
