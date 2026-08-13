"""Unit tests for logging_config module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

from src.utils.logging_config import LoggerWriter, setup_logging


def test_logger_writer_single_line() -> None:
    """Test LoggerWriter buffers and logs single line."""
    mock_logger = MagicMock(spec=logging.Logger)
    writer = LoggerWriter(mock_logger, level=logging.INFO)

    writer.write("Hello world\n")

    mock_logger.log.assert_called_once_with(logging.INFO, "%s", "Hello world")


def test_logger_writer_multi_line_and_flush() -> None:
    """Test LoggerWriter handles multiple lines and flush."""
    mock_logger = MagicMock(spec=logging.Logger)
    writer = LoggerWriter(mock_logger, level=logging.WARNING)

    writer.write("Line 1\nLine 2\nLine 3 incomplete")

    assert mock_logger.log.call_count == 2
    mock_logger.log.assert_any_call(logging.WARNING, "%s", "Line 1")
    mock_logger.log.assert_any_call(logging.WARNING, "%s", "Line 2")

    writer.flush()
    assert mock_logger.log.call_count == 3
    mock_logger.log.assert_any_call(logging.WARNING, "%s", "Line 3 incomplete")


def test_setup_logging_fallback(tmp_path: Path) -> None:
    """Test setup_logging with non-existent file falls back gracefully."""
    non_existent = tmp_path / "non_existent.yaml"
    setup_logging(config_path=str(non_existent), verbose=False)
    assert logging.getLogger().level in (logging.INFO, logging.NOTSET)


def test_setup_logging_verbose(tmp_path: Path) -> None:
    """Test setup_logging with verbose=True sets DEBUG level."""
    non_existent = tmp_path / "non_existent.yaml"
    setup_logging(config_path=str(non_existent), verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("src").level == logging.DEBUG


def test_setup_logging_valid_yaml(tmp_path: Path) -> None:
    """Test setup_logging with a valid YAML config file."""
    config_file = tmp_path / "logging.yaml"
    config_file.write_text(
        """
version: 1
disable_existing_loggers: false
handlers:
  console:
    class: logging.StreamHandler
    stream: ext://sys.stdout
root:
  level: WARNING
  handlers: [console]
loggers:
  src:
    level: INFO
""",
        encoding="utf-8",
    )

    setup_logging(config_path=str(config_file), verbose=False)
    assert logging.getLogger("src").level == logging.INFO
