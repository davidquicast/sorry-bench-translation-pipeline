"""
NLLB-200 backend via HuggingFace Transformers.

Tier 1 backbone: translates any content without content filters.
Supports facebook/nllb-200-3.3B, distilled-1.3B, and distilled-600M.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from aya_safety.backends import register_backend
from aya_safety.backends.base import MTBackend
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.utils.gpu import select_nllb_model

logger = logging.getLogger(__name__)


@register_backend("nllb")
class NLLBBackend(MTBackend):
    """
    NLLB-200 translation via HuggingFace Transformers.

    No content filters — translates any text faithfully. This is the
    unconditional backbone that guarantees every record gets an output.
    """

    def __init__(
        self,
        model: str = "auto",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        language_registry: LanguageRegistry | None = None,
    ) -> None:
        self._requested_model = model
        self._requested_device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._lang_registry = language_registry or LanguageRegistry()
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._model_name = ""

    @property
    def name(self) -> str:
        return f"nllb:{self._model_name.split('/')[-1]}" if self._model_name else "nllb"

    @property
    def tier(self) -> int:
        return 1

    def supports_language(self, lang_code: str) -> bool:
        try:
            self._lang_registry.resolve_nllb_code(lang_code)
            return True
        except KeyError:
            return False

    async def startup(self) -> None:
        """Load NLLB model and tokenizer."""
        self._model_name, self._device = select_nllb_model(self._requested_model)
        if self._requested_device != "auto":
            self._device = self._requested_device

        logger.info("Loading %s on %s...", self._model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()

        logger.info("NLLB model loaded: %s", self._model_name)

    async def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("NLLB model unloaded")

    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        results = await self.translate_batch([record], source_lang, target_lang)
        return results[0]

    async def translate_batch(
        self,
        records: list[TranslationRecord],
        source_lang: str,
        target_lang: str,
        batch_size: int | None = None,
    ) -> list[TranslationResult]:
        """
        Batched NLLB translation. Runs in a thread to avoid blocking the event loop.
        """
        bs = batch_size or self._batch_size
        src_nllb = self._lang_registry.resolve_nllb_code(source_lang)
        tgt_nllb = self._lang_registry.resolve_nllb_code(target_lang)

        all_results: list[TranslationResult] = []

        for i in range(0, len(records), bs):
            batch = records[i:i + bs]
            results = await asyncio.to_thread(
                self._translate_batch_sync, batch, src_nllb, tgt_nllb, target_lang,
            )
            all_results.extend(results)

        return all_results

    def _translate_batch_sync(
        self,
        records: list[TranslationRecord],
        src_nllb: str,
        tgt_nllb: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """Synchronous batched translation on the model."""
        texts = [r.source_text for r in records]
        start = time.perf_counter()

        self._tokenizer.src_lang = src_nllb
        tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_nllb)

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=self._max_length,
            )

        translations = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        per_record_ms = elapsed_ms / len(records)
        for record, translation in zip(records, translations):
            results.append(TranslationResult(
                record_id=record.record_id,
                target_lang=target_lang,
                translation=translation,
                backend_name=self.name,
                status=TranslationStatus.TIER1_COMPLETE,
                latency_ms=per_record_ms,
                token_count_in=len(record.source_text.split()),
                token_count_out=len(translation.split()),
            ))

        return results
