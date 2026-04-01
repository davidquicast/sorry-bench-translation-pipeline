"""
Abstract base classes for all translation backends and QA metrics.

Every backend — local MT model, LLM API, MT API — implements MTBackend.
LLM refinement backends additionally implement LLMBackend.refine().
QA metrics implement QAMetric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aya_safety.data.schemas import QAScore, TranslationRecord, TranslationResult


class MTBackend(ABC):
    """Abstract interface for all Tier 1 and Tier 2 translation backends."""

    @abstractmethod
    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Translate a single record."""

    async def translate_batch(
        self,
        records: list[TranslationRecord],
        source_lang: str,
        target_lang: str,
        batch_size: int = 32,
    ) -> list[TranslationResult]:
        """
        Translate a batch of records.

        Default implementation calls translate() sequentially.
        Backends with true batching (NLLB, CT2) should override this.
        """
        results = []
        for record in records:
            result = await self.translate(record, source_lang, target_lang)
            results.append(result)
        return results

    @abstractmethod
    def supports_language(self, lang_code: str) -> bool:
        """Check if this backend supports the given language code."""

    @abstractmethod
    async def startup(self) -> None:
        """Initialize resources (load model, create API client)."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier, e.g. 'nllb-3.3B', 'cohere', 'openai'."""

    @property
    @abstractmethod
    def tier(self) -> int:
        """1 for MT backbone, 2 for LLM refinement / MT API."""


class LLMBackend(MTBackend):
    """
    Extended interface for Tier 2 LLM backends that refine MT output.

    The key method is refine(), which takes a source text + MT output
    and produces an improved translation.
    """

    @abstractmethod
    async def refine(
        self,
        record: TranslationRecord,
        mt_output: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Refine an existing MT output using the fixed refinement prompt.

        This is the primary Tier 2 operation. The "improve MT output"
        framing reduces refusal rates compared to direct translation.
        """

    async def translate(
        self,
        record: TranslationRecord,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """LLM backends default to raising — use refine() instead."""
        raise NotImplementedError(
            f"{self.name}: LLM backends use refine(), not translate(). "
            "Pass the Tier 1 MT output to refine()."
        )

    @property
    def tier(self) -> int:
        return 2


class QAMetric(ABC):
    """Abstract interface for Tier 3 quality assessment metrics."""

    @abstractmethod
    async def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        reference: str | None = None,
    ) -> QAScore:
        """Score a single translation."""

    async def score_batch(
        self,
        sources: list[str],
        translations: list[str],
        source_lang: str,
        target_lang: str,
        references: list[str] | None = None,
    ) -> list[QAScore]:
        """
        Score a batch of translations.

        Default implementation calls score() sequentially.
        Metrics with true batching (COMET) should override this.
        """
        results = []
        refs = references or [None] * len(sources)
        for src, trans, ref in zip(sources, translations, refs):
            result = await self.score(src, trans, source_lang, target_lang, ref)
            results.append(result)
        return results

    @abstractmethod
    async def startup(self) -> None:
        """Load model or initialize client."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name, e.g. 'comet', 'chrf', 'gemba'."""
