"""
OpenAI backend: GPT-4o, GPT-4o-mini, with optional Batch API (50% discount).

The Batch API uploads all requests as JSONL, processes them async (up to 24h),
and returns results at half price.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import openai

from aya_safety.backends import register_backend
from aya_safety.backends.base import LLMBackend
from aya_safety.config import REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.retry.policies import RateLimitError, ServerError, api_retry

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}


@register_backend("openai")
class OpenAIBackend(LLMBackend):
    """OpenAI GPT-4o / GPT-4o-mini translation refinement."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self._model_name = _MODEL_MAP.get(model, model)
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._client: openai.AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True  # GPT-4o supports all target languages

    async def startup(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI client ready (model=%s)", self._model_name)

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
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=max_tokens,
            )

            translation = response.choices[0].message.content.strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            return TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER2_COMPLETE,
                latency_ms=elapsed_ms,
                token_count_in=response.usage.prompt_tokens if response.usage else 0,
                token_count_out=response.usage.completion_tokens if response.usage else 0,
            )

        except openai.RateLimitError:
            raise RateLimitError("OpenAI rate limit exceeded")
        except openai.InternalServerError:
            raise ServerError("OpenAI server error")


@register_backend("openai-batch")
class OpenAIBatchBackend(LLMBackend):
    """
    OpenAI Batch API for 50% discount on large translation runs.

    Uploads all requests as JSONL, polls for completion, downloads results.
    Designed for batch processing of entire languages at once.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self._model_name = _MODEL_MAP.get(model, model)
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._client: openai.AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return f"openai-batch:{self._model_name}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True

    async def startup(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI Batch client ready (model=%s)", self._model_name)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def refine(
        self,
        record: TranslationRecord,
        mt_output: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        # Single-record refinement falls back to standard API call
        std = OpenAIBackend(
            model=self._model_name,
            language_registry=self._lang_registry,
            temperature=self._temperature,
        )
        std._client = self._client
        return await std.refine(record, mt_output, source_lang, target_lang)

    async def translate_batch(
        self,
        records: list[TranslationRecord],
        source_lang: str,
        target_lang: str,
        batch_size: int = 0,  # Ignored — sends all at once
    ) -> list[TranslationResult]:
        """Submit all records via Batch API and wait for results."""
        raise NotImplementedError(
            "OpenAI Batch API translate_batch requires Tier 1 output. "
            "Use the pipeline's Tier 2 flow instead."
        )

    async def submit_batch_refinement(
        self,
        records: list[TranslationRecord],
        mt_outputs: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """
        Submit all refinement requests as a batch and wait for completion.

        This is the primary entry point for batch processing.
        """
        target_name = self._lang_registry.resolve_display_name(target_lang)

        # Build JSONL request file
        lines = []
        for record, mt_out in zip(records, mt_outputs):
            user_msg = REFINEMENT_USER_TEMPLATE.format(
                target_language=target_name,
                source_text=record.source_text,
                mt_output=mt_out,
            )
            est_tokens = len(record.source_text.split())
            max_tokens = int(est_tokens * 1.5 + 50)

            line = {
                "custom_id": record.record_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self._model_name,
                    "messages": [
                        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": self._temperature,
                    "max_tokens": max_tokens,
                },
            }
            lines.append(json.dumps(line))

        # Upload file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        ) as f:
            f.write("\n".join(lines))
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                file_obj = await self._client.files.create(file=f, purpose="batch")

            # Create batch
            batch = await self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info("Batch submitted: %s (%d requests)", batch.id, len(records))

            # Poll for completion
            import asyncio
            wait_seconds = 30
            while batch.status in ("validating", "in_progress", "finalizing"):
                await asyncio.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 1.5, 300)
                batch = await self._client.batches.retrieve(batch.id)
                logger.info("Batch %s status: %s", batch.id, batch.status)

            if batch.status != "completed":
                raise RuntimeError(f"Batch failed with status: {batch.status}")

            # Download results
            output_file = await self._client.files.content(batch.output_file_id)
            result_lines = output_file.text.strip().split("\n")

            # Parse results into map
            results_map: dict[str, TranslationResult] = {}
            for line in result_lines:
                data = json.loads(line)
                custom_id = data["custom_id"]
                response_body = data.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                usage = response_body.get("usage", {})

                if choices:
                    translation = choices[0]["message"]["content"].strip()
                    results_map[custom_id] = TranslationResult(
                        record_id=custom_id,
                        target_lang=target_lang,
                        translation=translation,
                        backend_name=self.name,
                        status=TranslationStatus.TIER2_COMPLETE,
                        token_count_in=usage.get("prompt_tokens", 0),
                        token_count_out=usage.get("completion_tokens", 0),
                    )

            # Return in original order
            results = []
            for record in records:
                if record.record_id in results_map:
                    results.append(results_map[record.record_id])
                else:
                    results.append(TranslationResult(
                        record_id=record.record_id,
                        target_lang=target_lang,
                        translation="",
                        backend_name=self.name,
                        status=TranslationStatus.TIER2_FAILED,
                        error_message="Missing from batch results",
                    ))

            return results

        finally:
            Path(tmp_path).unlink(missing_ok=True)
