"""
DeepL API backend.

$25/million characters. Highest quality for European languages.

WARNING: DeepL does NOT support Bengali (bn) or Hindi (hi).
This backend will log a loud warning and return an error for those languages.
"""

from __future__ import annotations

import logging
import os
import time

import deepl

from aya_safety.backends import register_backend
from aya_safety.backends.base import MTBackend
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

# DeepL target language codes
_DEEPL_LANG_MAP = {
    "ar": "AR", "bg": "BG", "cs": "CS", "da": "DA", "de": "DE",
    "el": "EL", "en": "EN-US", "es": "ES", "et": "ET", "fi": "FI",
    "fr": "FR", "hu": "HU", "id": "ID", "it": "IT", "ja": "JA",
    "ko": "KO", "lt": "LT", "lv": "LV", "nb": "NB", "nl": "NL",
    "pl": "PL", "pt": "PT-BR", "ro": "RO", "ru": "RU", "sk": "SK",
    "sl": "SL", "sv": "SV", "tr": "TR", "uk": "UK", "zh": "ZH-HANS",
}

# Languages NOT supported by DeepL
_UNSUPPORTED = {"bn", "hi", "am", "ha", "ig", "mg", "sn", "sw", "wo", "xh", "yo", "zu",
                "pa", "gu", "mr", "ta", "te", "ne", "ur", "fa", "he", "mt",
                "tl", "ms", "vi", "jv", "km", "th", "lo", "my"}


@register_backend("deepl")
class DeepLBackend(MTBackend):
    """DeepL translation API. Best quality for European languages."""

    def __init__(
        self,
        language_registry: LanguageRegistry | None = None,
        **kwargs,
    ) -> None:
        self._lang_registry = language_registry or LanguageRegistry()
        self._translator: deepl.Translator | None = None

    @property
    def name(self) -> str:
        return "deepl"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        if lang_code in _UNSUPPORTED:
            return False
        return lang_code in _DEEPL_LANG_MAP

    async def startup(self) -> None:
        api_key = os.environ.get("DEEPL_API_KEY")
        if not api_key:
            raise ValueError("DEEPL_API_KEY environment variable not set")
        self._translator = deepl.Translator(api_key)
        logger.info("DeepL translator ready")

    async def shutdown(self) -> None:
        self._translator = None

    @api_retry
    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        if target_lang in _UNSUPPORTED:
            lang_name = self._lang_registry.resolve_display_name(target_lang)
            logger.warning(
                "DeepL does NOT support %s (%s). "
                "Use NLLB-200 or GemmaX2-28 instead.",
                lang_name, target_lang,
            )
            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation="",
                backend_name=self.name,
                status=TranslationStatus.TIER2_FAILED,
                error_message=f"DeepL does not support {lang_name}",
            )

        tgt = _DEEPL_LANG_MAP.get(target_lang, target_lang.upper())

        start = time.perf_counter()

        import asyncio
        try:
            result = await asyncio.to_thread(
                self._translator.translate_text,
                record.source_text,
                target_lang=tgt,
            )

            translation = result.text
            elapsed_ms = (time.perf_counter() - start) * 1000

            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER2_COMPLETE,
                latency_ms=elapsed_ms,
            )

        except deepl.QuotaExceededException:
            raise RateLimitError("DeepL quota exceeded")
        except deepl.DeepLException as e:
            if "429" in str(e):
                raise RateLimitError(f"DeepL rate limit: {e}")
            raise ServerError(f"DeepL error: {e}")
