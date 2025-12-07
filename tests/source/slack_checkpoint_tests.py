"""Tests for Slack checkpoint functionality.

This module tests the SlackCheckpoint class behavior including default values,
update logic, and state structure.
"""

import unittest
from datetime import datetime, timezone

from src.sources.slack import SlackCheckpoint, SlackConfig


class TestSlackCheckpoint(unittest.TestCase):
    """Test cases for SlackCheckpoint class."""

    def test_default_behavior(self) -> None:
        """Test default checkpoint behavior with initial lookback days."""
        config = SlackConfig(initial_lookback_days=7, id="test", enabled=True, workspace_domain="test.slack.com")
        checkpoint = SlackCheckpoint(config=config)

        # Check default latest_ts
        now = datetime.now(timezone.utc).timestamp()
        expected_default = now - (7 * 24 * 60 * 60)

        latest_ts = checkpoint.get_latest_ts("channel_new")
        # Allow small delta for execution time
        self.assertAlmostEqual(latest_ts, expected_default, delta=5.0)

        # Check default earliest_ts (should be None or handle gracefully)
        earliest_ts = checkpoint.get_earliest_ts("channel_new")
        self.assertIsNone(earliest_ts)

    def test_update_logic(self) -> None:
        """Test checkpoint update logic for expanding time ranges."""
        checkpoint = SlackCheckpoint(config=SlackConfig(id="test", enabled=True, workspace_domain="test.slack.com"))
        channel_id = "C1"

        # Initial update
        ts1 = 1000.0
        ts2 = 2000.0
        checkpoint.update_channel_ts(channel_id, newest_ts=ts2, oldest_ts=ts1)

        self.assertEqual(checkpoint.get_latest_ts(channel_id), ts2)
        self.assertEqual(checkpoint.get_earliest_ts(channel_id), ts1)

        # Update with newer max and older min
        ts3 = 3000.0  # Newer
        ts0 = 500.0  # Older
        checkpoint.update_channel_ts(channel_id, newest_ts=ts3, oldest_ts=ts0)

        self.assertEqual(checkpoint.get_latest_ts(channel_id), ts3)
        self.assertEqual(checkpoint.get_earliest_ts(channel_id), ts0)

        # Update with inside range (should not change)
        ts_mid = 1500.0
        checkpoint.update_channel_ts(channel_id, newest_ts=ts_mid, oldest_ts=ts_mid)

        self.assertEqual(checkpoint.get_latest_ts(channel_id), ts3)
        self.assertEqual(checkpoint.get_earliest_ts(channel_id), ts0)

    def test_state_structure(self) -> None:
        """Test internal state structure of checkpoint."""
        checkpoint = SlackCheckpoint(config=SlackConfig(id="test", enabled=True, workspace_domain="test.slack.com"))
        checkpoint.update_channel_ts("C1", newest_ts=200.0, oldest_ts=100.0)

        channels = checkpoint.state.get("channels", {})
        self.assertIn("C1", channels)
        self.assertEqual(channels["C1"]["latest_ts"], 200.0)
        self.assertEqual(channels["C1"]["earliest_ts"], 100.0)


if __name__ == "__main__":
    unittest.main()
