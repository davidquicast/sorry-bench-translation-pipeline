"""
Google Gemini backend: Gemini 2.5 Flash, Gemini 2.5 Pro.

Gemini 2.5 Flash matches GPT-4o-mini pricing with strong multilingual support.
"""

from __future__ import annotations

import logging
import os
import time

from aya_safety.backends import register_backend
from aya_safety.backends.base import LLMBackend
from aya_safety.config import REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
}


@register_backend("gemini")
class GeminiBackend(LLMBackend):
    """Google Gemini translation refinement."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self._model_name = _MODEL_MAP.get(model, model)
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._client = None

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True

    async def startup(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(
            self._model_name,
            system_instruction=REFINEMENT_SYSTEM_PROMPT,
        )
        logger.info("Gemini client ready (model=%s)", self._model_name)

    async def shutdown(self) -> None:
        self._client = None

    @api_retry
    async def refine(
        self,
        record: TranslationRecord,
        mt_output: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        import asyncio

        target_name = self._lang_registry.resolve_display_name(target_lang)
        user_msg = REFINEMENT_USER_TEMPLATE.format(
            target_language=target_name,
            source_text=record.source_text,
            mt_output=mt_output,
        )

        est_tokens = len(record.source_text.split())
        max_tokens = int(est_tokens * 1.5 + 50)

        start = time.perf_counter()

        try:
            import google.generativeai as genai
            generation_config = genai.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=max_tokens,
            )

            # Gemini SDK is sync — run in thread
            response = await asyncio.to_thread(
                self._client.generate_content,
                user_msg,
                generation_config=generation_config,
            )

            translation = response.text.strip()
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
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                raise RateLimitError(f"Gemini rate limit: {e}")
            if "500" in err_str or "503" in err_str:
                raise ServerError(f"Gemini server error: {e}")
            raise
