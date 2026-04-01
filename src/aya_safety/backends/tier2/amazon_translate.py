"""
Amazon Translate backend via boto3.

$15/million characters. Supports 75+ languages.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from aya_safety.backends import register_backend
from aya_safety.backends.base import MTBackend
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

_AWS_LANG_MAP = {
    "zh": "zh",
    "zh-TW": "zh-TW",
}


@register_backend("amazon_translate")
class AmazonTranslateBackend(MTBackend):
    """Amazon Translate API via boto3."""

    def __init__(
        self,
        language_registry: LanguageRegistry | None = None,
        **kwargs,
    ) -> None:
        self._lang_registry = language_registry or LanguageRegistry()
        self._client = None

    @property
    def name(self) -> str:
        return "amazon_translate"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True  # 75+ languages

    async def startup(self) -> None:
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("translate", region_name=region)
        logger.info("Amazon Translate client ready (region=%s)", region)

    async def shutdown(self) -> None:
        self._client = None

    @api_retry
    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        tgt = _AWS_LANG_MAP.get(target_lang, target_lang)
        src = _AWS_LANG_MAP.get(source_lang, source_lang)

        start = time.perf_counter()

        try:
            result = await asyncio.to_thread(
                self._client.translate_text,
                Text=record.source_text,
                SourceLanguageCode=src,
                TargetLanguageCode=tgt,
            )

            translation = result["TranslatedText"]
            elapsed_ms = (time.perf_counter() - start) * 1000

            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER2_COMPLETE,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            err_name = type(e).__name__
            if "Throttling" in err_name or "TooManyRequests" in err_name:
                raise RateLimitError(f"Amazon Translate throttled: {e}")
            if "InternalServer" in err_name:
                raise ServerError(f"Amazon Translate error: {e}")
            raise
