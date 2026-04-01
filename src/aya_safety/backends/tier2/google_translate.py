"""
Google Cloud Translation API v3 backend.

$20/million characters, 500K characters/month free tier.
Supports 130+ languages, strong quality across all targets including Bengali/Hindi.
No content filtering — translates any text faithfully.
"""

from __future__ import annotations

import logging
import os
import time

from aya_safety.backends import register_backend
from aya_safety.backends.base import MTBackend
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

# Google Translate uses ISO 639-1 codes (mostly)
_LANG_MAP = {
    "zh": "zh-CN",
    "zh-TW": "zh-TW",
}


@register_backend("google_translate")
class GoogleTranslateBackend(MTBackend):
    """Google Cloud Translation API v3."""

    def __init__(
        self,
        language_registry: LanguageRegistry | None = None,
        **kwargs,
    ) -> None:
        self._lang_registry = language_registry or LanguageRegistry()
        self._client = None

    @property
    def name(self) -> str:
        return "google_translate"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True  # 130+ languages

    async def startup(self) -> None:
        from google.cloud import translate_v3 as translate

        self._client = translate.TranslationServiceAsyncClient()
        self._project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._parent = f"projects/{self._project_id}/locations/global"
        logger.info("Google Translate client ready")

    async def shutdown(self) -> None:
        self._client = None

    @api_retry
    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        tgt = _LANG_MAP.get(target_lang, target_lang)
        src = _LANG_MAP.get(source_lang, source_lang)

        start = time.perf_counter()

        try:
            response = await self._client.translate_text(
                parent=self._parent,
                contents=[record.source_text],
                source_language_code=src,
                target_language_code=tgt,
            )

            translation = response.translations[0].translated_text
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
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                raise RateLimitError(f"Google Translate rate limit: {e}")
            if "500" in err_str or "503" in err_str:
                raise ServerError(f"Google Translate server error: {e}")
            raise
