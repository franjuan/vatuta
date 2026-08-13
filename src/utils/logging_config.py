"""Logging configuration and utilities for Vatuta.

Provides YAML dictConfig setup and stream adapters for capturing external process logs.
"""

import io
import logging
import logging.config
from pathlib import Path
from typing import Any, Optional

import yaml
from rich.logging import RichHandler


class LoggerWriter(io.TextIOBase):
    """TextIO stream adapter that redirects writes to a Python Logger.

    Useful for capturing process streams (e.g., Docker container stderr) and redirecting
    them to a standard Python logger.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """Initialize the LoggerWriter adapter.

        Args:
            logger (logging.Logger): Target logger instance.
            level (int): Logging level to use for emitted log records. Defaults to logging.INFO.
        """
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer: str = ""
        self._write_fd: Optional[int] = None
        self._read_thread: Optional[Any] = None

    def _read_pipe(self, read_fd: int) -> None:
        """Background thread that reads from the OS pipe and logs it."""
        import os

        with os.fdopen(read_fd, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.logger.log(self.level, "%s", line)

    def fileno(self) -> int:
        """Return a real OS file descriptor, lazily creating a pipe if necessary."""
        if self._write_fd is None:
            import os
            import threading

            read_fd, self._write_fd = os.pipe()
            self._read_thread = threading.Thread(target=self._read_pipe, args=(read_fd,), daemon=True)
            self._read_thread.start()
        return self._write_fd

    def write(self, s: str) -> int:
        """Write string data to the stream buffer and emit complete lines as logs.

        Args:
            s (str): String snippet to write.

        Returns:
            int: Number of characters written.
        """
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, "%s", line)
        return len(s)

    def flush(self) -> None:
        """Flush any remaining buffered text to the logger."""
        if self._buffer.strip():
            self.logger.log(self.level, "%s", self._buffer.strip())
            self._buffer = ""

    def close(self) -> None:
        """Close the underlying pipe write file descriptor if it was created."""
        if self._write_fd is not None:
            import os

            try:
                os.close(self._write_fd)
            except OSError:
                pass
            self._write_fd = None
        super().close()

    def isatty(self) -> bool:
        """Return whether stream is interactive.

        Returns:
            bool: Always False for logger adapter.
        """
        return False


def setup_logging(config_path: Optional[str] = None, verbose: bool = False) -> None:
    """Configure Python logging using YAML dictConfig or basic fallback with Rich tracebacks.

    Args:
        config_path (Optional[str]): Path to the YAML logging configuration file.
            Defaults to "config/logging.yaml".
        verbose (bool): If True, forces root and 'src' loggers to DEBUG level.
    """
    target_path = Path(config_path) if config_path else Path("config/logging.yaml")

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                logging.config.dictConfig(config)
            else:
                logging.basicConfig(
                    level=logging.DEBUG if verbose else logging.INFO,
                    datefmt="[%X]",
                    handlers=[RichHandler(rich_tracebacks=True, log_time_format="[%X]")],
                )
        except Exception as e:
            logging.basicConfig(
                level=logging.DEBUG if verbose else logging.INFO,
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, log_time_format="[%X]")],
            )
            logging.warning("Failed to load logging config from %s: %s", target_path, e)
    else:
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, log_time_format="[%X]")],
        )

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)
        # Also ensure console handlers allow DEBUG
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
