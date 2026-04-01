"""
Back-translation + BERTScore metric.

Round-trip translation (target → source) evaluated via BERTScore provides
a practical reference-free quality check. Moon et al. (2020; arxiv 2004.13937)
showed this approach outperformed WMT 2019 QE task submissions.
"""

from __future__ import annotations

import asyncio
import logging

from aya_safety.backends import register_metric
from aya_safety.backends.base import QAMetric
from aya_safety.data.schemas import QAScore

logger = logging.getLogger(__name__)


@register_metric("backtranslation")
class BackTranslationMetric(QAMetric):
    """
    Back-translation quality check using BERTScore.

    Process: translation → back-translate to source lang → compare
    with original source via BERTScore F1.
    """

    def __init__(self, config=None, **kwargs) -> None:
        self._bt_backend = None
        self._bert_scorer = None
        self._config = config

    @property
    def name(self) -> str:
        return "backtranslation"

    async def startup(self) -> None:
        from aya_safety.backends import create_backend

        bt_backend_name = "nllb"
        if self._config and hasattr(self._config, "backtranslation_backend"):
            bt_backend_name = self._config.backtranslation_backend

        self._bt_backend = create_backend(bt_backend_name)
        await self._bt_backend.startup()

        import bert_score
        self._bert_score_module = bert_score
        logger.info("Back-translation metric ready (backend=%s)", bt_backend_name)

    async def shutdown(self) -> None:
        if self._bt_backend:
            await self._bt_backend.shutdown()

    async def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        reference: str | None = None,
    ) -> QAScore:
        scores = await self.score_batch(
            [source], [translation], source_lang, target_lang,
        )
        return scores[0]

    async def score_batch(
        self,
        sources: list[str],
        translations: list[str],
        source_lang: str,
        target_lang: str,
        references: list[str] | None = None,
    ) -> list[QAScore]:
        from aya_safety.data.schemas import TranslationRecord

        # Back-translate: target → source
        bt_records = [
            TranslationRecord(
                record_id=f"bt_{i}",
                source_text=trans,
                source_lang=target_lang,
            )
            for i, trans in enumerate(translations)
        ]

        bt_results = await self._bt_backend.translate_batch(
            bt_records, target_lang, source_lang,
        )
        back_translations = [r.translation for r in bt_results]

        # Compute BERTScore between original sources and back-translations
        P, R, F1 = await asyncio.to_thread(
            self._bert_score_module.score,
            back_translations,
            sources,
            lang=source_lang,
            verbose=False,
        )

        scores = []
        for i, f1 in enumerate(F1.tolist()):
            scores.append(QAScore(
                metric_name="backtranslation",
                score=f1,
                details={
                    "precision": P[i].item(),
                    "recall": R[i].item(),
                    "back_translation": back_translations[i][:200],
                },
            ))

        return scores
