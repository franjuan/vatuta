"""Centralized provider-independent LLM manager for Vatuta."""

import logging
import os
from typing import Any, Dict, Tuple

import dspy
import litellm

from src.llm.errors import LLMConfigurationError
from src.metrics.llm_metrics import VatutaMetricsLogger
from src.models.config import RagConfig

logger = logging.getLogger(__name__)


class LLMProviderManager:
    """Manages lifecycle, validation, and instantiation of LLM backends."""

    def __init__(self, config: RagConfig) -> None:
        """Initialize provider manager with validated RAG configuration.

        Args:
            config: Validated RAG configuration containing llm_backends.
        """
        self.config = config
        self._cached_lms: Dict[Tuple[str, str], dspy.LM] = {}
        self._setup_metrics_callback()

    def _setup_metrics_callback(self) -> None:
        """Register VatutaMetricsLogger callback with LiteLLM if not present."""
        try:
            callbacks = getattr(litellm, "callbacks", None)
            if callbacks is not None:
                if not any(isinstance(cb, VatutaMetricsLogger) for cb in callbacks):
                    callbacks.append(VatutaMetricsLogger())
        except Exception as e:
            logger.debug("Could not attach VatutaMetricsLogger to litellm: %s", e)

    def validate_backends(self) -> None:
        """Execute broad, non-intrusive pre-flight validation on configured backends.

        Validates:
        1. Model format conforms to 'provider/model_name' without making network calls.
        2. Required provider environment variables are present via dynamic LiteLLM inspection
           (`litellm.validate_environment(model)`) without hardcoding provider or variable names.

        Does NOT make active network probes, ping calls, or test completion prompts.

        Raises:
            LLMConfigurationError: If any backend has invalid configuration or missing credentials.
        """
        for backend_name, backend_conf in self.config.llm_backends.items():
            model = backend_conf.model
            if "/" not in model:
                raise LLMConfigurationError(
                    f"Model identifier '{model}' in backend '{backend_name}' must follow 'provider/model' format."
                )

            empty_env_keys = [k for k, v in os.environ.items() if not v.strip()]
            for k in empty_env_keys:
                del os.environ[k]
            try:
                validation = litellm.validate_environment(model=model)
            except Exception as e:
                raise LLMConfigurationError(
                    f"LiteLLM environment validation failed for model '{model}': {e}",
                    original_exception=e,
                ) from e
            finally:
                for k in empty_env_keys:
                    os.environ[k] = ""

            if not validation.get("keys_in_environment", False):
                missing = ", ".join(validation.get("missing_keys", []))
                raise LLMConfigurationError(
                    f"Missing required environment credentials for model '{model}' "
                    f"in backend '{backend_name}'. Missing keys: {missing}"
                )

            logger.info("Validated LLM backend '%s' with model '%s'", backend_name, model)

    def get_dspy_lm(self, backend_name: str, role: str = "general") -> dspy.LM:
        """Instantiate or return cached DSPy language model instance.

        Args:
            backend_name: Key of backend in config.llm_backends.
            role: Pipeline role label ('router', 'generator', etc.) for telemetry.

        Returns:
            Configured dspy.LM instance.

        Raises:
            LLMConfigurationError: If backend_name is not configured.
        """
        if backend_name not in self.config.llm_backends:
            available = ", ".join(self.config.llm_backends.keys())
            raise LLMConfigurationError(
                f"Backend '{backend_name}' not found in llm_backends. Configured backends: {available}"
            )

        cache_key = (backend_name, role)
        if cache_key in self._cached_lms:
            return self._cached_lms[cache_key]

        backend_conf = self.config.llm_backends[backend_name]
        kwargs: Dict[str, Any] = dict(backend_conf.extra_kwargs)

        metadata = dict(kwargs.get("metadata", {}))
        metadata["role"] = role
        kwargs["metadata"] = metadata

        if backend_conf.api_base:
            kwargs["api_base"] = backend_conf.api_base
        if backend_conf.top_p is not None:
            kwargs["top_p"] = backend_conf.top_p

        logger.info(
            "Instantiating dspy.LM for backend '%s' (role='%s', model='%s')",
            backend_name,
            role,
            backend_conf.model,
        )

        lm = dspy.LM(
            model=backend_conf.model,
            temperature=backend_conf.temperature,
            max_tokens=backend_conf.max_tokens,
            **kwargs,
        )
        self._cached_lms[cache_key] = lm
        return lm
