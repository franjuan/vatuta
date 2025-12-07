"""Adaptive, per-method rate limiter for Slack Web API usage.

This module provides an `AdaptiveRateLimiter` designed for proactive, tier-aware
rate limiting. It implements:

- Token bucket pacing per method
- Small per-call jitter to avoid thundering herd
- Automatic backoff on HTTP 429 honoring Retry-After
- Gradual recovery when operating healthily

Intended usage:
    limiter = AdaptiveRateLimiter(
        defaults={
            "conversations.list": {"rpm": 18, "cap": 24, "burst": 5},
            "conversations.history": {"rpm": 45, "cap": 60, "burst": 10},
        }
    )
    limiter.acquire("conversations.list")
    # perform API call
    ...
    # on 429:
    limiter.on_rate_limited("conversations.list", retry_after_seconds)

Notes:
- RPM stands for requests-per-minute.
- `cap` is the upper bound the limiter may ramp to during recovery.
- `burst` is the token bucket capacity (how many immediate calls can proceed).

Algorithm overview:
    - Token bucket:
        Each method M has a bucket with capacity = `burst` and a refill rate
        r = target_rpm / 60 tokens per second. To perform a call, we must
        acquire 1 token; if no token is available, we sleep until one is.
        On each acquire attempt we also add a small random jitter J ~ U(0.05, 0.15)
        seconds to reduce synchronization across workers.
    - Backoff on 429:
        When Slack returns HTTP 429 with a `Retry-After` header (seconds),
        we immediately:
            1) set `next_allowed_after = now + Retry-After`,
            2) collapse burst to 1 (stop new bursts post-429),
            3) reduce target_rpm by 50% (multiplicative decrease) but not below
               a safety floor (`min_rpm`, default 6 rpm).
        Subsequent acquires will sleep until `next_allowed_after` if needed.
    - Recovery:
        If a method has no 429s for 120 seconds, we:
            1) increase target_rpm by 10% (additive/multiplicative hybrid)
               up to the configured `cap`,
            2) widen burst_capacity by +1 step up to the method's original
               configured `burst` (`recovery_max_burst`).
        This yields conservative ramp-up minimizing oscillations and re-limits.

Rationale and Slack tiers:
    Slack publishes per-method rate tiers (per minute) and emphasizes handling
    `Retry-After`: https://docs.slack.dev/apis/web-api/rate-limits/
    - `conversations.list` is Tier 2 (20+ rpm tolerated with bursts)
    - `conversations.history` is Tier 3 (50+ rpm tolerated with bursts)
    We seed target RPMs slightly under nominal tier thresholds (e.g., 18, 45)
    and allow recovery up to caps (e.g., 24, 60) with controlled bursts.

Concurrency and scope:
    - This limiter is process-local and method-local; it does not coordinate
      across processes or hosts.
    - Buckets are stored in a shared map guarded by a lock, but per-bucket
      state updates are not individually locked; for heavy multithreading
      or distributed workers, consider using an external coordinator or
      wrap per-bucket ops with stronger synchronization if needed.
"""

import logging
import random
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class _Bucket:
    """Internal token bucket with adaptive backoff/recovery.

    The bucket controls pacing for a single API method. It refills tokens at the
    configured target RPM, applies backoff when rate limited, and gradually
    increases throughput during healthy periods until reaching a configured cap.

    Args:
        rpm: Initial target requests per minute.
        cap: Maximum requests per minute the limiter may reach via recovery.
        burst: Maximum burst size (token bucket capacity).

    Attributes:
        target_rpm: Current target RPM (adapts on backoff/recovery).
        cap_rpm: Upper bound for target RPM.
        burst_capacity: Token bucket capacity (burst size).
        recovery_max_burst: Upper bound for burst capacity during recovery.
        tokens: Current available tokens.
        last_refill_ts: Last time tokens were refilled.
        next_allowed_after: Wall-clock timestamp before which no calls are allowed.
        last_429_ts: Last time a 429 was observed.
        healthy_since_ts: Timestamp from which we count healthy operation.
        min_rpm: Lower bound of target RPM after backoff.
    """

    def __init__(self, rpm: float, cap: float, burst: int):
        # Configuration
        self.target_rpm = max(1.0, float(rpm))
        self.cap_rpm = max(self.target_rpm, float(cap))
        self.burst_capacity = max(1, int(burst))
        self.recovery_max_burst = self.burst_capacity
        # Runtime state
        self.tokens = float(self.burst_capacity)
        self.last_refill_ts = time.time()
        self.next_allowed_after: float = 0.0
        self.last_429_ts: float = 0.0
        self.healthy_since_ts: float = time.time()
        # Min rpm to avoid stalling forever
        self.min_rpm = 6.0

    def refill(self, now: float) -> None:
        """Refill tokens according to current target RPM.

        Tokens accumulate continuously at (target_rpm / 60) per second up to the
        bucket's burst capacity.

        Args:
            now: Current wall-clock timestamp in seconds.
        """
        per_sec = self.target_rpm / 60.0
        elapsed = max(0.0, now - self.last_refill_ts)
        added = elapsed * per_sec
        if added > 0:
            self.tokens = min(self.burst_capacity, self.tokens + added)
            self.last_refill_ts = now

    def acquire_one(self, now: float) -> float:
        """Try to acquire a single token.

        Args:
            now: Current wall-clock timestamp in seconds.

        Returns:
            Seconds to sleep before a token is available (0.0 if available now).
        """
        self.refill(now)
        # Enforce pause after rate-limit events
        if now < self.next_allowed_after:
            return self.next_allowed_after - now
        # If we have at least 1 token, consume it and return 0 (no sleep needed)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        # Need to wait until a token is available
        per_sec = self.target_rpm / 60.0  # tokens per second generation rate
        if per_sec <= 0:
            return 1.0  # safety: if per_sec is 0, we need to wait for 1 second
        deficit = 1.0 - self.tokens  # how many tokens we need to wait for to get 1 token
        return deficit / per_sec  # how many seconds we need to wait for to get 1 token

    def on_rate_limited(self, wait_seconds: int) -> None:
        """Apply backoff and honor server-provided Retry-After.

        - Cuts `target_rpm` by 50% down to a floor (`min_rpm`).
        - Shrinks burst to 1 to immediately stop bursts after limiting.
        - Sets `next_allowed_after` to enforce server cooldown window.

        Args:
            wait_seconds: Value from Retry-After header (seconds).
        """
        now = time.time()
        self.last_429_ts = now
        self.healthy_since_ts = now  # reset healthy window
        # Reduce target rpm by 50%, but not below min_rpm
        new_rpm = max(self.min_rpm, self.target_rpm * 0.5)
        if new_rpm < self.target_rpm:
            logger.info(f"Limiter backoff: rpm {self.target_rpm:.2f} -> {new_rpm:.2f}")
        self.target_rpm = new_rpm
        # Collapse burst to 1 immediately after a limit, to stop bursts
        if self.burst_capacity > 1:
            logger.info(f"Limiter backoff: burst {self.burst_capacity} -> 1")
        self.burst_capacity = 1
        self.tokens = min(self.tokens, float(self.burst_capacity))
        # Respect Retry-After window
        self.next_allowed_after = max(self.next_allowed_after, now + float(wait_seconds))

    def maybe_recover(self) -> None:
        """Gradually increase throughput when operating healthily.

        Every 120 seconds without a rate limit, increase `target_rpm` by 10%
        up to `cap_rpm`. Also widens `burst_capacity` gradually up to the
        method's configured burst (`recovery_max_burst`).
        """
        now = time.time()
        healthy_for = now - self.healthy_since_ts
        if healthy_for >= 120.0:
            increased = min(self.cap_rpm, self.target_rpm * 1.10)
            if increased > self.target_rpm:
                logger.debug(f"Limiter recovery: rpm {self.target_rpm:.2f} -> {increased:.2f}")
                self.target_rpm = increased
            if self.burst_capacity < self.recovery_max_burst:
                new_burst = min(self.recovery_max_burst, self.burst_capacity + 1)
                logger.debug(f"Limiter recovery: burst {self.burst_capacity} -> {new_burst}")
                self.burst_capacity = new_burst
            self.healthy_since_ts = now


class AdaptiveRateLimiter:
    """Adaptive, per-method rate limiter with backoff and recovery.

    The limiter maintains a token bucket per API method and paces calls by
    sleeping before issuing the request. It injects a small random jitter to
    reduce alignment across workers, backs off aggressively upon rate limits
    (HTTP 429), and gradually recovers when operating healthily.

    Args:
        defaults: Mapping of method -> configuration dict with keys:
            - rpm (float): initial target requests per minute
            - cap (float): maximum requests per minute during recovery
            - burst (int): token bucket capacity (burst size)
    """

    def __init__(self, defaults: Dict[str, Dict[str, float]]):
        """Initialize the rate limiter with per-method configurations."""
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        for method, cfg in defaults.items():
            self._buckets[method] = _Bucket(
                rpm=float(cfg.get("rpm", 20.0)),
                cap=float(cfg.get("cap", 20.0)),
                burst=int(cfg.get("burst", 5)),
            )

    def _get_bucket(self, method: str) -> _Bucket:
        r"""Return the bucket for the given method, creating a default if needed.

        Args:
            method: API method name (e.g., \"conversations.list\").

        Returns:
            The bucket instance managing the method's pacing.
        """
        with self._lock:
            if method not in self._buckets:
                # Sensible default for unknown methods
                self._buckets[method] = _Bucket(rpm=20.0, cap=20.0, burst=5)
            return self._buckets[method]

    def acquire(self, method: str) -> None:
        r"""Block until a request for `method` is allowed to proceed.

        This function may sleep multiple times, adding 50–150 ms of jitter
        to each sleep to reduce synchronized wakes across workers.

        Args:
            method: API method name (e.g., \"conversations.list\").
        """
        bucket = self._get_bucket(method)
        while True:
            bucket.maybe_recover()
            now = time.time()
            sleep_needed = bucket.acquire_one(now)
            # Add small jitter (50–150ms) to reduce thundering herd
            jitter = random.uniform(0.05, 0.15) if sleep_needed > 0 or bucket.next_allowed_after > now else 0.0
            total_sleep = max(0.0, sleep_needed) + jitter
            if total_sleep > 0:
                # Provide context: cooldown vs normal pacing, plus current state
                cooldown = max(0.0, bucket.next_allowed_after - now)
                if cooldown > 0:
                    logger.debug(
                        f"[{method}] cooldown: sleeping {total_sleep:.3f}s "
                        f"(cooldown {cooldown:.3f}s, jitter {jitter:.3f}s, "
                        f"rpm {bucket.target_rpm:.2f}, burst {bucket.burst_capacity}, "
                        f"tokens {bucket.tokens:.2f})"
                    )
                else:
                    logger.debug(
                        f"[{method}] pacing: sleeping {total_sleep:.3f}s "
                        f"(jitter {jitter:.3f}s, rpm {bucket.target_rpm:.2f}, "
                        f"burst {bucket.burst_capacity}, tokens {bucket.tokens:.2f})"
                    )
                time.sleep(total_sleep)
                continue
            break

    def on_rate_limited(self, method: str, wait_seconds: Optional[int]) -> None:
        r"""Notify the limiter that `method` was rate limited (HTTP 429).

        The limiter honors the provided Retry-After (seconds) and applies
        immediate throughput reduction. Passing None or non-positive values
        will default to a minimal 1-second cooldown.

        Args:
            method: API method name (e.g., \"conversations.list\").
            wait_seconds: Value from Retry-After header (seconds).
        """
        if wait_seconds is None:
            wait_seconds = 1
        bucket = self._get_bucket(method)
        bucket.on_rate_limited(int(wait_seconds))
