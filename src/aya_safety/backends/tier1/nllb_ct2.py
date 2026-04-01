"""
NLLB-200 backend via CTranslate2 for optimized inference (3-5x speedup).

Requires a CTranslate2-converted NLLB model. If not available, falls back
to the standard transformers backend.
"""

from __future__ import annotations

import asyncio
import logging
import time

from aya_safety.backends import register_backend
from aya_safety.backends.base import MTBackend
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.utils.gpu import select_device

logger = logging.getLogger(__name__)


@register_backend("nllb-ct2")
class NLLBCT2Backend(MTBackend):
    """
    NLLB-200 translation via CTranslate2 for optimized inference.

    CTranslate2 provides 3-5x speedup over standard transformers inference
    through INT8/FP16 quantization and optimized kernels.
    """

    def __init__(
        self,
        model_path: str = "facebook/nllb-200-3.3B",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 64,
        language_registry: LanguageRegistry | None = None,
    ) -> None:
        self._model_path = model_path
        self._requested_device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._lang_registry = language_registry or LanguageRegistry()
        self._translator = None
        self._tokenizer = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "nllb-ct2"

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
        import ctranslate2
        from transformers import AutoTokenizer

        self._device = select_device(self._requested_device)
        ct2_device = "cuda" if self._device == "cuda" else "cpu"

        logger.info("Loading CTranslate2 NLLB model from %s on %s...", self._model_path, ct2_device)

        self._translator = ctranslate2.Translator(
            self._model_path,
            device=ct2_device,
            compute_type="int8_float16" if ct2_device == "cuda" else "int8",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        logger.info("CTranslate2 NLLB model loaded")

    async def shutdown(self) -> None:
        del self._translator
        del self._tokenizer
        self._translator = None
        self._tokenizer = None
        logger.info("CTranslate2 model unloaded")

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
        self._tokenizer.src_lang = src_nllb

        texts = [r.source_text for r in records]
        start = time.perf_counter()

        # Tokenize
        tokenized = [
            self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text)
            )
            for text in texts
        ]

        # Translate
        ct2_results = self._translator.translate_batch(
            tokenized,
            target_prefix=[[tgt_nllb]] * len(tokenized),
            max_decoding_length=self._max_length,
            beam_size=4,
        )

        # Decode
        translations = []
        for result in ct2_results:
            tokens = result.hypotheses[0]
            # Remove target language prefix token
            if tokens and tokens[0] == tgt_nllb:
                tokens = tokens[1:]
            text = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(tokens),
                skip_special_tokens=True,
            )
            translations.append(text)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_record_ms = elapsed_ms / len(records)

        results = []
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
