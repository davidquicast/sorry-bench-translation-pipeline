"""
Per-backend async rate limiting using aiolimiter's leaky-bucket algorithm.
"""

from __future__ import annotations

from aiolimiter import AsyncLimiter


class BackendRateLimiter:
    """
    Manages per-backend rate limiters.

    Each backend gets its own AsyncLimiter configured with a max
    requests-per-second rate from config.
    """

    def __init__(self, rate_limits: dict[str, float]) -> None:
        """
        Args:
            rate_limits: Mapping of backend name → max requests per second.
        """
        self._limiters: dict[str, AsyncLimiter] = {
            name: AsyncLimiter(max_rate=rate, time_period=1.0)
            for name, rate in rate_limits.items()
        }

    async def acquire(self, backend_name: str) -> None:
        """Wait until the rate limit allows a request for this backend."""
        if backend_name in self._limiters:
            await self._limiters[backend_name].acquire()

    def has_limiter(self, backend_name: str) -> bool:
        return backend_name in self._limiters
