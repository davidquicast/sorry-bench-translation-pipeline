"""
Local HuggingFace causal LM backend: any model specified by model_id.

GemmaX2-28-9B (Xiaomi Research, NAACL 2025) is the best open-source option
for Bengali/Hindi per the research report. Supports all 28 XSafetyRep
target languages.
"""

from __future__ import annotations

import asyncio
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aya_safety.backends import register_backend
from aya_safety.backends.base import LLMBackend
from aya_safety.config import REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE
from aya_safety.data.schemas import TranslationRecord, TranslationResult, TranslationStatus
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.utils.gpu import select_device

logger = logging.getLogger(__name__)


@register_backend("local_hf")
class LocalHFBackend(LLMBackend):
    """Local HuggingFace causal LM for translation refinement."""

    def __init__(
        self,
        model: str = "ModelSpace/GemmaX2-28-9B-v0.1",
        language_registry: LanguageRegistry | None = None,
        temperature: float = 0.0,
        device: str = "auto",
        **kwargs,
    ) -> None:
        self._model_name = model
        self._temperature = temperature
        self._lang_registry = language_registry or LanguageRegistry()
        self._requested_device = device
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        short = self._model_name.split("/")[-1]
        return f"local_hf:{short}"

    @property
    def tier(self) -> int:
        return 2

    def supports_language(self, lang_code: str) -> bool:
        return True  # Depends on the specific model; assume yes

    async def startup(self) -> None:
        self._device = select_device(self._requested_device)
        logger.info("Loading %s on %s...", self._model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
        )
        self._model.eval()
        logger.info("Local HF model loaded: %s", self._model_name)

    async def shutdown(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if self._device == "cuda":
            torch.cuda.empty_cache()

    async def refine(
        self,
        record: TranslationRecord,
        mt_output: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        return await asyncio.to_thread(
            self._refine_sync, record, mt_output, source_lang, target_lang,
        )

    def _refine_sync(
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

        # Build chat template if tokenizer supports it
        if self._tokenizer.chat_template:
            messages = [
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"{REFINEMENT_SYSTEM_PROMPT}\n\n{user_msg}"

        est_tokens = len(record.source_text.split())
        max_new_tokens = int(est_tokens * 1.5 + 50)

        start = time.perf_counter()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self._temperature if self._temperature > 0 else None,
                do_sample=self._temperature > 0,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        translation = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elapsed_ms = (time.perf_counter() - start) * 1000

        return TranslationResult(
            record_id=record.record_id,
            target_lang=target_lang,
            translation=translation,
            backend_name=self.name,
            status=TranslationStatus.TIER2_COMPLETE,
            latency_ms=elapsed_ms,
            token_count_in=input_len,
            token_count_out=len(new_tokens),
        )
