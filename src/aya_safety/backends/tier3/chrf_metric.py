"""
chrF++ metric via SacreBLEU.

Character n-gram + word bigram metric. Less sensitive to tokenization than
BLEU and better for morphologically rich languages (Arabic, Hindi, Bengali,
Japanese). The ALMA-R paper recommends abandoning BLEU as primary metric.
"""

from __future__ import annotations

import asyncio
import logging

import sacrebleu

from aya_safety.backends import register_metric
from aya_safety.backends.base import QAMetric
from aya_safety.data.schemas import QAScore

logger = logging.getLogger(__name__)


@register_metric("chrf")
class ChrFMetric(QAMetric):
    """chrF++ via SacreBLEU (character n-grams + word bigrams)."""

    def __init__(self, **kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        return "chrf"

    async def startup(self) -> None:
        logger.info("chrF++ metric ready (via SacreBLEU)")

    async def shutdown(self) -> None:
        pass

    async def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        reference: str | None = None,
    ) -> QAScore:
        # chrF++ requires a reference; use source as pseudo-reference if none provided
        ref = reference if reference else source
        result = await asyncio.to_thread(
            sacrebleu.sentence_chrf, translation, [ref],
        )

        return QAScore(
            metric_name="chrf",
            score=result.score,
            details={"char_order": 6, "word_order": 2},
        )

    async def score_batch(
        self,
        sources: list[str],
        translations: list[str],
        source_lang: str,
        target_lang: str,
        references: list[str] | None = None,
    ) -> list[QAScore]:
        refs = references if references else sources
        scores = []
        for trans, ref in zip(translations, refs):
            result = sacrebleu.sentence_chrf(trans, [ref])
            scores.append(QAScore(
                metric_name="chrf",
                score=result.score,
                details={"char_order": 6, "word_order": 2},
            ))
        return scores
