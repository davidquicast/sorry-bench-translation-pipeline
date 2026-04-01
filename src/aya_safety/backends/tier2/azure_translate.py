"""
Azure Translator backend.

$10/million characters, 2M characters/month free tier.
Best value among dedicated MT APIs. Supports all 10 target languages.
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

# Azure uses ISO 639-1 codes with some exceptions
_AZURE_LANG_MAP = {
    "zh": "zh-Hans",
    "zh-TW": "zh-Hant",
    "nb": "nb",
}


@register_backend("azure_translate")
class AzureTranslateBackend(MTBackend):
    """Microsoft Azure Translator API."""

    def __init__(
        self,
        language_registry: LanguageRegistry | None = None,
        **kwargs,
    ) -> None:
        self._lang_registry = language_registry or LanguageRegistry()
        self._key: str = ""
        self._region: str = ""
        self._endpoint = "https://api.cognitive.microsofttranslator.com"

    @property
    def name(self) -> str:
        return "azure_translate"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True  # Azure supports 100+ languages

    async def startup(self) -> None:
        self._key = os.environ.get("AZURE_TRANSLATE_KEY", "")
        self._region = os.environ.get("AZURE_TRANSLATE_REGION", "")
        if not self._key:
            raise ValueError("AZURE_TRANSLATE_KEY environment variable not set")
        logger.info("Azure Translator ready")

    async def shutdown(self) -> None:
        pass

    @api_retry
    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        import asyncio
        import json
        import urllib.request

        tgt = _AZURE_LANG_MAP.get(target_lang, target_lang)
        src = _AZURE_LANG_MAP.get(source_lang, source_lang)

        start = time.perf_counter()

        url = f"{self._endpoint}/translate?api-version=3.0&from={src}&to={tgt}"
        body = json.dumps([{"Text": record.source_text}]).encode("utf-8")
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Ocp-Apim-Subscription-Region": self._region,
            "Content-Type": "application/json",
        }

        def _do_request():
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = await asyncio.to_thread(_do_request)
            translation = result[0]["translations"][0]["text"]
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
            if "429" in err_str:
                raise RateLimitError(f"Azure rate limit: {e}")
            if "500" in err_str or "503" in err_str:
                raise ServerError(f"Azure server error: {e}")
            raise
