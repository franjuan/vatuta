"""Tests for Slack chunking strategies."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.sources.slack import DocumentUnit, SlackConfig, SlackSource


class TestSlackChunking(unittest.TestCase):
    """Test cases for Slack chunking logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = SlackConfig(
            id="test",
            workspace_domain="test.slack.com",
            chunk_time_interval_minutes=60,  # 1 hour
            chunk_max_size_chars=100,
            chunk_max_count=3,
            chunk_similarity_threshold=0.5,
            enabled=True,
        )
        self.source = SlackSource(
            config=self.config, secrets={"bot_token": "xoxb-test"}, storage_path=tempfile.gettempdir()
        )  #
        # Mock rate limiter to avoid issues during init/usage if accidentally called
        self.source.rate_limiter = MagicMock()

        # Test document
        self.doc = DocumentUnit(
            document_id="doc1", source="slack", source_instance_id="test", source_doc_id="s1", title="Test Doc"
        )

    def test_single_message(self) -> None:
        """Test simple single message case."""
        msgs = [{"ts": "1000", "text": "Hello world"}]

        # Mock embeddings: 1 message -> 1 embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0, 0.0, 0.0]]
        self.source._embedding_model = mock_model

        chunks = self.source._build_chunks_for_messages(self.doc, msgs)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Hello world", chunks[0].text)

    def test_time_split(self) -> None:
        """Test splitting by time interval."""
        # 1 hour gap config
        msg1_ts = 1000
        msg2_ts = 1000 + (61 * 60)  # 61 minutes later

        msgs = [{"ts": str(msg1_ts), "text": "Message 1"}, {"ts": str(msg2_ts), "text": "Message 2"}]

        # Mock embeddings
        mock_model = MagicMock()
        # High similarity to ensure time causes split, not content
        mock_model.encode.return_value = [[1.0, 0.0], [0.99, 0.0]]
        self.source._embedding_model = mock_model

        with patch("sentence_transformers.util.cos_sim") as mock_sim:
            # Should return high similarity
            mock_sim.return_value.item.return_value = 0.99

            chunks = self.source._build_chunks_for_messages(self.doc, msgs)

            self.assertEqual(len(chunks), 2)
            self.assertIn("Message 1", chunks[0].text)
            self.assertIn("Message 2", chunks[1].text)

    def test_time_no_split(self) -> None:
        """Test no split when within time interval."""
        msg1_ts = 1000
        msg2_ts = 1000 + (30 * 60)  # 30 minutes later

        msgs = [{"ts": str(msg1_ts), "text": "Message 1"}, {"ts": str(msg2_ts), "text": "Message 2"}]

        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0, 0.0], [0.99, 0.0]]
        self.source._embedding_model = mock_model

        with patch("sentence_transformers.util.cos_sim") as mock_sim:
            mock_sim.return_value.item.return_value = 0.99
            chunks = self.source._build_chunks_for_messages(self.doc, msgs)

            self.assertEqual(len(chunks), 1)
            self.assertIn("Message 1", chunks[0].text)
            self.assertIn("Message 2", chunks[0].text)

    def test_size_split_chars(self) -> None:
        """Test splitting by character limit."""
        # Limit is 100 chars
        long_text = "a" * 60
        msgs = [{"ts": "1000", "text": long_text}, {"ts": "1001", "text": long_text}]  # 60+60 = 120 > 100

        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0], [1.0]]
        self.source._embedding_model = mock_model

        with patch("sentence_transformers.util.cos_sim") as mock_sim:
            mock_sim.return_value.item.return_value = 1.0

            chunks = self.source._build_chunks_for_messages(self.doc, msgs)
            self.assertEqual(len(chunks), 2)

    def test_size_split_count(self) -> None:
        """Test splitting by message count."""
        # Limit is 3
        msgs = [
            {"ts": "1000", "text": "1"},
            {"ts": "1001", "text": "2"},
            {"ts": "1002", "text": "3"},
            {"ts": "1003", "text": "4"},  # 4th message should start new chunk/split
        ]

        mock_model = MagicMock()
        # 4 embeddings
        mock_model.encode.return_value = [[1.0]] * 4
        self.source._embedding_model = mock_model

        with patch("sentence_transformers.util.cos_sim") as mock_sim:
            mock_sim.return_value.item.return_value = 1.0

            chunks = self.source._build_chunks_for_messages(self.doc, msgs)
            # Expect chunk 1: 1,2,3; Chunk 2: 4
            self.assertEqual(len(chunks), 2)
            self.assertIn("3", chunks[0].text)
            self.assertIn("4", chunks[1].text)

    def test_semantic_split(self) -> None:
        """Test splitting by semantic similarity."""
        # Threshold 0.5. Sim < 0.5 -> Split.
        msgs = [
            {"ts": "1000", "text": "Apple"},
            {"ts": "1001", "text": "Banana"},  # Sim High (Fruits)
            {"ts": "1002", "text": "Car"},  # Sim Low (Car vs Banana) -> Split
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0], [0.9], [0.1]]
        self.source._embedding_model = mock_model

        with patch("sentence_transformers.util.cos_sim") as mock_sim:
            # Sim sequence: (Apple, Banana) -> 0.9; (Banana, Car) -> 0.2
            # The code calls cos_sim(curr, prev).
            # Call 1: Banana vs Apple -> 0.9
            # Call 2: Car vs Banana -> 0.2
            side_effect = [MagicMock(item=MagicMock(return_value=0.9)), MagicMock(item=MagicMock(return_value=0.2))]
            mock_sim.side_effect = side_effect

            chunks = self.source._build_chunks_for_messages(self.doc, msgs)

            self.assertEqual(len(chunks), 2)
            # Chunk 1: Apple, Banana
            # Chunk 2: Car
            self.assertIn("Banana", chunks[0].text)
            self.assertIn("Car", chunks[1].text)


if __name__ == "__main__":
    unittest.main()
