"""
Language detection verification via fast-langdetect (FastText-based).

Verifies that the translation is actually in the target language.
80x faster than the langdetect library, supports 176 languages.
"""

from __future__ import annotations

import asyncio
import logging

from aya_safety.backends import register_metric
from aya_safety.backends.base import QAMetric
from aya_safety.data.schemas import QAScore

logger = logging.getLogger(__name__)

# Mapping from ISO codes to fast-langdetect codes
_LANGDETECT_MAP = {
    "zh": "zh",
    "zh-TW": "zh",
    "nb": "no",
}


@register_metric("langdetect")
class LangDetectMetric(QAMetric):
    """Verify translation language matches target via fast-langdetect."""

    def __init__(self, **kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        return "langdetect"

    async def startup(self) -> None:
        from fast_langdetect import detect
        # Warm up the model
        detect("Hello world")
        logger.info("Language detection metric ready")

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
        from fast_langdetect import detect

        result = await asyncio.to_thread(detect, translation)
        detected_lang = result.get("lang", "").lower()
        confidence = result.get("score", 0.0)

        expected = _LANGDETECT_MAP.get(target_lang, target_lang.split("-")[0])
        is_correct = detected_lang == expected

        # Score: 1.0 if correct language detected with high confidence, else scaled
        if is_correct:
            score = confidence
        else:
            score = 0.0

        return QAScore(
            metric_name="langdetect",
            score=score,
            details={
                "detected_lang": detected_lang,
                "expected_lang": expected,
                "confidence": confidence,
                "correct": is_correct,
            },
        )
