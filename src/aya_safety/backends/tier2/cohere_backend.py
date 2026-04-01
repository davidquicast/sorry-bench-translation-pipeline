"""
Cohere backend: Command A, Command R+, Command R, Aya Expanse.

Primary Tier 2 backend for the XSafetyRep project. Aya Expanse is from
the same model family as Tiny Aya — the pipeline can use a model from
the same family to translate the benchmark that evaluates it.
"""

from __future__ import annotations

import logging
import os
import time

import cohere

from aya_safety.backends import register_backend
from aya_safety.backends.base import LLMBackend
from aya_safety.config import REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

# Cohere model name mapping
_MODEL_MAP = {
    "command-a": "command-a-03-2025",
    "command-r-plus": "command-r-plus-08-2024",
    "command-r": "command-r-08-2024",
    "aya-expanse": "c4ai-aya-expanse-8b",
    "aya-expanse-32b": "c4ai-aya-expanse-32b",
}

# Aya Expanse covers 23 languages including all 9 XSafetyRep targets
_AYA_EXPANSE_LANGS = {
    "ar", "bn", "de", "en", "es", "fa", "fr", "hi", "id", "it", "ja",
    "ko", "nl", "pl", "pt", "ro", "ru", "th", "tr", "uk", "vi", "zh", "cs",
}


@register_backend("cohere")
class CohereBackend(LLMBackend):
    """Cohere translation refinement via Command A / Aya Expanse."""

    def __init__(
        self,
        model: str = "command-a",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self._model_key = model
        self._model_name = _MODEL_MAP.get(model, model)
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._client: cohere.AsyncClientV2 | None = None

    @property
    def name(self) -> str:
        return f"cohere:{self._model_key}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return lang_code in _AYA_EXPANSE_LANGS or lang_code in self._lang_registry

    async def startup(self) -> None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        self._client = cohere.AsyncClientV2(api_key=api_key)
        logger.info("Cohere client ready (model=%s)", self._model_name)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @api_retry
    async def refine(
        self,
        record: TranslationRecord,
        mt_output: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        target_name = self._lang_registry.resolve_display_name(target_lang)
        user_msg = REFINEMENT_USER_TEMPLATE.format(
            target_language=target_name,
            source_text=record.source_text,
            mt_output=mt_output,
        )

        # Dynamic max tokens: source_token_count * 1.5 + 50
        est_tokens = len(record.source_text.split())
        max_tokens = int(est_tokens * 1.5 + 50)

        start = time.perf_counter()

        try:
            response = await self._client.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=max_tokens,
            )

            translation = response.message.content[0].text.strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER2_COMPLETE,
                latency_ms=elapsed_ms,
                token_count_in=response.usage.tokens.input_tokens if response.usage else 0,
                token_count_out=response.usage.tokens.output_tokens if response.usage else 0,
            )

        except cohere.TooManyRequestsError:
            raise RateLimitError("Cohere rate limit exceeded")
        except cohere.InternalServerError:
            raise ServerError("Cohere server error")
