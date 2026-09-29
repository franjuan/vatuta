"""Prometheus metrics and LiteLLM callback logger for LLM invocations."""

import datetime
import logging
from typing import Any, Dict, Tuple

import prometheus_client as _prom

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover

    class CustomLogger:  # type: ignore[no-redef]
        """Fallback base class when litellm is not installed."""

        pass


logger = logging.getLogger(__name__)

Counter = _prom.Counter
Histogram = _prom.Histogram

LLM_CALL_LATENCY = Histogram(
    "vatuta_llm_call_latency_seconds",
    "Language model call latency in seconds",
    ["provider", "role", "model", "status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, float("inf")),
)

LLM_TOKENS = Counter(
    "vatuta_llm_tokens_total",
    "Total tokens consumed by language model calls by token type (prompt, completion)",
    ["provider", "role", "model", "token_type"],
)

LLM_CALLS = Counter(
    "vatuta_llm_calls_total",
    "Total language model calls by provider, role, model, and status",
    ["provider", "role", "model", "status"],
)


def _extract_call_metadata(kwargs: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract model, provider, and role metadata from LiteLLM callback kwargs.

    Args:
        kwargs: Keyword arguments dictionary passed to LiteLLM callback.

    Returns:
        Tuple of (provider, role, model).
    """
    model: str = str(kwargs.get("model") or "unknown")
    provider: str = model.split("/")[0] if "/" in model else "unknown"

    role: str = "general"
    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict) and "role" in metadata:
        role = str(metadata["role"])
    else:
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            nested_meta = litellm_params.get("metadata")
            if isinstance(nested_meta, dict) and "role" in nested_meta:
                role = str(nested_meta["role"])

    return provider, role, model


def _calculate_latency(start_time: Any, end_time: Any) -> float:
    """Calculate elapsed seconds between start and end timestamps.

    Args:
        start_time: Start timestamp (datetime or float/int).
        end_time: End timestamp (datetime or float/int).

    Returns:
        Elapsed time in seconds as float.
    """
    if isinstance(start_time, (datetime.datetime, datetime.date)) and isinstance(
        end_time, (datetime.datetime, datetime.date)
    ):
        return max(0.0, (end_time - start_time).total_seconds())
    try:
        return max(0.0, float(end_time) - float(start_time))
    except (TypeError, ValueError):
        return 0.0


def _extract_token_usage(response_obj: Any) -> Tuple[int, int]:
    """Extract prompt and completion token counts from response object.

    Args:
        response_obj: Response object or dict returned from LiteLLM.

    Returns:
        Tuple of (prompt_tokens, completion_tokens).
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0

    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    return prompt_tokens, completion_tokens


class VatutaMetricsLogger(CustomLogger):
    """Custom logger integrating LiteLLM callbacks with Prometheus metrics."""

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Record Prometheus metrics for successful LLM invocations.

        Args:
            kwargs: LiteLLM execution parameters and metadata.
            response_obj: Response output object.
            start_time: Start timestamp of the invocation.
            end_time: End timestamp of the invocation.
        """
        provider, role, model = _extract_call_metadata(kwargs)
        latency = _calculate_latency(start_time, end_time)

        LLM_CALL_LATENCY.labels(provider=provider, role=role, model=model, status="success").observe(latency)
        LLM_CALLS.labels(provider=provider, role=role, model=model, status="success").inc()

        prompt_tokens, completion_tokens = _extract_token_usage(response_obj)
        if prompt_tokens > 0:
            LLM_TOKENS.labels(provider=provider, role=role, model=model, token_type="prompt").inc(prompt_tokens)
        if completion_tokens > 0:
            LLM_TOKENS.labels(provider=provider, role=role, model=model, token_type="completion").inc(completion_tokens)

        logger.debug(
            "Recorded LLM success metrics: provider=%s role=%s model=%s latency=%.3fs prompt=%d completion=%d",
            provider,
            role,
            model,
            latency,
            prompt_tokens,
            completion_tokens,
        )

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Record Prometheus metrics for failed LLM invocations.

        Args:
            kwargs: LiteLLM execution parameters and metadata.
            response_obj: Response output or exception info.
            start_time: Start timestamp of the invocation.
            end_time: End timestamp of the invocation.
        """
        provider, role, model = _extract_call_metadata(kwargs)
        latency = _calculate_latency(start_time, end_time)

        LLM_CALL_LATENCY.labels(provider=provider, role=role, model=model, status="failure").observe(latency)
        LLM_CALLS.labels(provider=provider, role=role, model=model, status="failure").inc()

        logger.debug(
            "Recorded LLM failure metrics: provider=%s role=%s model=%s latency=%.3fs",
            provider,
            role,
            model,
            latency,
        )
