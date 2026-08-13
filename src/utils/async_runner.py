"""Utility for bridging synchronous code to asynchronous event loops."""

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class AsyncLoopThread:
    """A thread running a dedicated asyncio event loop.

    Useful for managing long-lived asynchronous resources (like MCP servers)
    from a synchronous context.
    """

    def __init__(self) -> None:
        """Initialize the thread and event loop."""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self._ready = threading.Event()

    def _run_loop(self) -> None:
        """Run the event loop forever."""
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def start(self) -> None:
        """Start the background thread and event loop."""
        self.thread.start()
        self._ready.wait()

    def stop(self) -> None:
        """Stop the background thread and event loop."""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()

    def run_coroutine(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine in this thread's event loop and block until it returns.

        Args:
            coro: The coroutine to run.

        Returns:
            The result of the coroutine.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
