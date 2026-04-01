"""
COMET quality metric via Unbabel/wmt22-cometkiwi-da (reference-free).

COMET has the highest correlation with human judgments among automatic metrics
(Rei et al., 2022). The reference-free CometKiwi variant enables quality
estimation without gold references.
"""

from __future__ import annotations

import asyncio
import logging

from aya_safety.backends import register_metric
from aya_safety.backends.base import QAMetric
from aya_safety.data.schemas import QAScore

logger = logging.getLogger(__name__)


@register_metric("comet")
class COMETMetric(QAMetric):
    """COMET reference-free quality estimation."""

    def __init__(self, config=None, **kwargs) -> None:
        model_name = "Unbabel/wmt22-cometkiwi-da"
        if config and hasattr(config, "comet_model"):
            model_name = config.comet_model
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return "comet"

    async def startup(self) -> None:
        from comet import download_model, load_from_checkpoint

        model_path = download_model(self._model_name)
        self._model = load_from_checkpoint(model_path)
        logger.info("COMET model loaded: %s", self._model_name)

    async def shutdown(self) -> None:
        del self._model
        self._model = None

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
            [reference] if reference else None,
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
        data = [
            {"src": src, "mt": mt}
            for src, mt in zip(sources, translations)
        ]
        if references:
            for d, ref in zip(data, references):
                if ref:
                    d["ref"] = ref

        output = await asyncio.to_thread(
            self._model.predict, data, batch_size=32, gpus=0,
        )

        scores = []
        for seg_score in output.scores:
            scores.append(QAScore(
                metric_name="comet",
                score=float(seg_score),
            ))

        logger.debug("COMET batch: %d scores, mean=%.3f", len(scores),
                      sum(s.score for s in scores) / len(scores) if scores else 0)
        return scores
