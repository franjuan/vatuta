"""Unit tests for LLM Prometheus metrics and VatutaMetricsLogger callback."""

import datetime
import unittest
from typing import Dict

import prometheus_client

from src.metrics.llm_metrics import (
    VatutaMetricsLogger,
    _calculate_latency,
    _extract_call_metadata,
    _extract_token_usage,
)


def _get_metric_sample(metric_name: str, labels: Dict[str, str]) -> float:
    """Read a sample value from the Prometheus default registry.

    Args:
        metric_name: Name of the Prometheus metric.
        labels: Label dictionary to match.

    Returns:
        Sample value as float, or 0.0 if not found.
    """
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return float(sample.value)
    return 0.0


class TestLLMMetrics(unittest.TestCase):
    """Test suite for VatutaMetricsLogger and LLM Prometheus telemetry."""

    def setUp(self) -> None:
        """Initialize logger instance."""
        self.logger = VatutaMetricsLogger()

    def test_extract_call_metadata_standard(self) -> None:
        """Verify metadata extraction with explicit provider and role."""
        kwargs = {
            "model": "gemini/gemini-2.5-flash",
            "metadata": {"role": "router"},
        }
        provider, role, model = _extract_call_metadata(kwargs)
        self.assertEqual(provider, "gemini")
        self.assertEqual(role, "router")
        self.assertEqual(model, "gemini/gemini-2.5-flash")

    def test_extract_call_metadata_nested_litellm_params(self) -> None:
        """Verify role extraction from nested litellm_params."""
        kwargs = {
            "model": "anthropic/claude-3-7-sonnet",
            "litellm_params": {"metadata": {"role": "generator"}},
        }
        provider, role, model = _extract_call_metadata(kwargs)
        self.assertEqual(provider, "anthropic")
        self.assertEqual(role, "generator")
        self.assertEqual(model, "anthropic/claude-3-7-sonnet")

    def test_extract_call_metadata_defaults(self) -> None:
        """Verify default fallback values when metadata is omitted."""
        kwargs = {"model": "simple-model"}
        provider, role, model = _extract_call_metadata(kwargs)
        self.assertEqual(provider, "unknown")
        self.assertEqual(role, "general")
        self.assertEqual(model, "simple-model")

    def test_calculate_latency_datetime(self) -> None:
        """Verify latency calculation with datetime objects."""
        start = datetime.datetime(2026, 9, 29, 12, 0, 0)
        end = datetime.datetime(2026, 9, 29, 12, 0, 1, 500000)
        self.assertAlmostEqual(_calculate_latency(start, end), 1.5)

    def test_calculate_latency_floats(self) -> None:
        """Verify latency calculation with numeric float timestamps."""
        self.assertAlmostEqual(_calculate_latency(10.0, 12.3), 2.3)

    def test_extract_token_usage_dict(self) -> None:
        """Verify token extraction from dictionary response."""
        response = {"usage": {"prompt_tokens": 120, "completion_tokens": 45}}
        prompt, completion = _extract_token_usage(response)
        self.assertEqual(prompt, 120)
        self.assertEqual(completion, 45)

    def test_log_success_event(self) -> None:
        """Verify metrics emission on success event."""
        kwargs = {
            "model": "gemini/gemini-test-model",
            "metadata": {"role": "router"},
        }
        response = {"usage": {"prompt_tokens": 50, "completion_tokens": 25}}
        start_time = 100.0
        end_time = 100.8

        labels_calls = {
            "provider": "gemini",
            "role": "router",
            "model": "gemini/gemini-test-model",
            "status": "success",
        }
        initial_calls = _get_metric_sample("vatuta_llm_calls_total", labels_calls)

        self.logger.log_success_event(kwargs, response, start_time, end_time)

        updated_calls = _get_metric_sample("vatuta_llm_calls_total", labels_calls)
        self.assertEqual(updated_calls, initial_calls + 1.0)

        prompt_labels = {
            "provider": "gemini",
            "role": "router",
            "model": "gemini/gemini-test-model",
            "token_type": "prompt",
        }
        completion_labels = {
            "provider": "gemini",
            "role": "router",
            "model": "gemini/gemini-test-model",
            "token_type": "completion",
        }
        self.assertGreaterEqual(_get_metric_sample("vatuta_llm_tokens_total", prompt_labels), 50.0)
        self.assertGreaterEqual(_get_metric_sample("vatuta_llm_tokens_total", completion_labels), 25.0)

    def test_log_failure_event(self) -> None:
        """Verify metrics emission on failure event."""
        kwargs = {
            "model": "gemini/gemini-fail-model",
            "metadata": {"role": "generator"},
        }
        start_time = 200.0
        end_time = 201.2

        labels_calls = {
            "provider": "gemini",
            "role": "generator",
            "model": "gemini/gemini-fail-model",
            "status": "failure",
        }
        initial_calls = _get_metric_sample("vatuta_llm_calls_total", labels_calls)

        self.logger.log_failure_event(kwargs, None, start_time, end_time)

        updated_calls = _get_metric_sample("vatuta_llm_calls_total", labels_calls)
        self.assertEqual(updated_calls, initial_calls + 1.0)
