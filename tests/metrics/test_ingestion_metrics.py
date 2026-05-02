"""Tests for ingestion quality metrics across sources."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import prometheus_client

from src.sources.slack import DocumentUnit, SlackConfig, SlackSource


def _get_metric_value(metric_name: str, labels: dict[str, str]) -> float:
    """Read a sample value from the Prometheus registry by name and label set.

    Supports metrics with suffixes like '_count' for Histograms.
    """
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return float(sample.value)
    return 0.0


class TestSlackIngestionMetrics(unittest.TestCase):
    """Verify that ingestion quality metrics are emitted during Slack chunking."""

    def setUp(self) -> None:
        """Set up a minimal SlackSource with mocked dependencies."""
        self.config = SlackConfig(
            id="test-slack",
            workspace_domain="test.slack.com",
            chunk_time_interval_minutes=60,
            chunk_max_size_chars=500,
            chunk_max_messages=5,
            chunk_similarity_threshold=0.5,
            chunk_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enabled=True,
        )
        self.source = SlackSource(
            config=self.config,
            secrets={"bot_token": "xoxb-test"},
            storage_path=tempfile.gettempdir(),
        )
        self.source.rate_limiter = MagicMock()
        self.doc = DocumentUnit(
            document_id="doc-slack-1",
            source="slack",
            source_instance_id="test-slack",
            source_doc_id="s1",
            title="Test Doc",
        )

    def _run_chunking(self, msgs: list, embeddings: list, similarities: list) -> list:
        """Helper: mock model + cos_sim, run chunking, return chunks."""
        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        self.source._embedding_model = mock_model

        sim_mocks = [MagicMock(item=MagicMock(return_value=s)) for s in similarities]
        with patch("sentence_transformers.util.cos_sim", side_effect=sim_mocks):
            return self.source._build_chunks_for_messages(self.doc, msgs)

    def test_chunk_size_chars_observed(self) -> None:
        """ingest_chunk_size_chars histogram should be observed for each chunk."""
        msgs = [{"ts": "1000", "text": "Hello"}]
        before = _get_metric_value(
            "ingest_chunk_size_chars_count",
            {"source": "slack", "source_id": "test-slack", "chunk_type": "slack_message"},
        )
        self._run_chunking(msgs, [[1.0, 0.0]], [])
        after = _get_metric_value(
            "ingest_chunk_size_chars_count",
            {"source": "slack", "source_id": "test-slack", "chunk_type": "slack_message"},
        )
        self.assertGreater(after, before, "chunk_size_chars should have been observed")

    def test_token_budget_ratio_observed(self) -> None:
        """ingest_chunk_token_budget_ratio histogram should be observed for each chunk."""
        msgs = [{"ts": "1000", "text": "Hello world"}]
        before = _get_metric_value(
            "ingest_chunk_token_budget_ratio_count",
            {"source": "slack", "source_id": "test-slack", "chunk_type": "slack_message"},
        )
        self._run_chunking(msgs, [[1.0, 0.0]], [])
        after = _get_metric_value(
            "ingest_chunk_token_budget_ratio_count",
            {"source": "slack", "source_id": "test-slack", "chunk_type": "slack_message"},
        )
        self.assertGreater(after, before, "token_budget_ratio should have been observed")

    def test_chunks_per_document_observed(self) -> None:
        """ingest_chunks_per_document histogram should be observed after building chunks."""
        msgs = [{"ts": "1000", "text": "Msg"}]
        before = _get_metric_value(
            "ingest_chunks_per_document_count",
            {"source": "slack", "source_id": "test-slack"},
        )
        self._run_chunking(msgs, [[1.0]], [])
        after = _get_metric_value(
            "ingest_chunks_per_document_count",
            {"source": "slack", "source_id": "test-slack"},
        )
        self.assertGreater(after, before, "chunks_per_document should have been observed")

    def test_split_reason_time_incremented(self) -> None:
        """ingest_chunk_split_reason_total with reason=time should increment on time gaps."""
        ts1 = 1000
        ts2 = 1000 + (self.config.chunk_time_interval_minutes + 5) * 60  # beyond interval

        msgs = [{"ts": str(ts1), "text": "Early msg"}, {"ts": str(ts2), "text": "Late msg"}]

        before = _get_metric_value(
            "ingest_chunk_split_reason_total",
            {"source": "slack", "source_id": "test-slack", "reason": "time"},
        )
        # High similarity — ensures only time causes the split
        self._run_chunking(msgs, [[1.0, 0.0], [1.0, 0.0]], [0.99])
        after = _get_metric_value(
            "ingest_chunk_split_reason_total",
            {"source": "slack", "source_id": "test-slack", "reason": "time"},
        )
        self.assertGreater(after, before, "split_reason 'time' should have been incremented")

    def test_split_reason_size_chars_incremented(self) -> None:
        """ingest_chunk_split_reason_total with reason=size_chars should increment."""
        long_text = "x" * 400  # > chunk_max_size_chars (500) when combined
        msgs = [{"ts": "1000", "text": long_text}, {"ts": "1001", "text": long_text}]

        before = _get_metric_value(
            "ingest_chunk_split_reason_total",
            {"source": "slack", "source_id": "test-slack", "reason": "size_chars"},
        )
        self._run_chunking(msgs, [[1.0], [1.0]], [1.0])
        after = _get_metric_value(
            "ingest_chunk_split_reason_total",
            {"source": "slack", "source_id": "test-slack", "reason": "size_chars"},
        )
        self.assertGreater(after, before, "split_reason 'size_chars' should have been incremented")

    def test_embedding_latency_observed(self) -> None:
        """ingest_embedding_latency_seconds histogram should be observed after model.encode()."""
        msgs = [{"ts": "1000", "text": "Hello"}]
        before = _get_metric_value(
            "ingest_embedding_latency_seconds_count",
            {"source": "slack", "source_id": "test-slack"},
        )
        self._run_chunking(msgs, [[1.0]], [])
        after = _get_metric_value(
            "ingest_embedding_latency_seconds_count",
            {"source": "slack", "source_id": "test-slack"},
        )
        self.assertGreater(after, before, "embedding_latency should have been observed")


if __name__ == "__main__":
    unittest.main()
