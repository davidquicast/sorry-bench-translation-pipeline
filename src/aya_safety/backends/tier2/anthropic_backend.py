"""
Anthropic backend: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Haiku.

Note: Claude exhibits the highest refusal sensitivity among major LLMs
(SafeConstellations, Maskey et al., 2025). The refinement framing
mitigates this, but fallback to other backends is expected more often.
"""

from __future__ import annotations

import logging
import os
import time

import anthropic

from aya_safety.backends import register_backend
from aya_safety.backends.base import LLMBackend
from aya_safety.config import REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
    "claude-3-5-haiku": "claude-3-5-haiku-latest",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


@register_backend("anthropic")
class AnthropicBackend(LLMBackend):
    """Anthropic Claude translation refinement."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self._model_name = _MODEL_MAP.get(model, model)
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self._model_name}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True

    async def startup(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("Anthropic client ready (model=%s)", self._model_name)

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

        est_tokens = len(record.source_text.split())
        max_tokens = int(est_tokens * 1.5 + 50)

        start = time.perf_counter()

        try:
            response = await self._client.messages.create(
                model=self._model_name,
                system=REFINEMENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=self._temperature,
                max_tokens=max_tokens,
            )

            translation = response.content[0].text.strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER2_COMPLETE,
                latency_ms=elapsed_ms,
                token_count_in=response.usage.input_tokens,
                token_count_out=response.usage.output_tokens,
            )

        except anthropic.RateLimitError:
            raise RateLimitError("Anthropic rate limit exceeded")
        except anthropic.InternalServerError:
            raise ServerError("Anthropic server error")
