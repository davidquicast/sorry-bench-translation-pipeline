"""
Retry policies using tenacity: exponential backoff with jitter for API calls.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from aya_safety.config import RetryConfig

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when an API returns a rate-limit response (429)."""


class ServerError(Exception):
    """Raised when an API returns a server error (5xx)."""


class TranslationError(Exception):
    """General translation failure."""


# Default retryable exception types
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    RateLimitError,
    ServerError,
    OSError,
)


def make_retry_decorator(config: RetryConfig) -> Callable:
    """Create a tenacity retry decorator from config."""
    return retry(
        wait=wait_random_exponential(
            multiplier=config.backoff_multiplier,
            max=config.backoff_max,
        ) if config.jitter else wait_random_exponential(
            multiplier=config.backoff_multiplier,
            max=config.backoff_max,
        ),
        stop=stop_after_attempt(config.max_attempts),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Convenience: default retry decorator (can be overridden per-backend)
api_retry = retry(
    wait=wait_random_exponential(multiplier=1.0, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
