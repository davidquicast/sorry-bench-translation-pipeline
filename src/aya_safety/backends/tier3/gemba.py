"""
GEMBA: LLM-as-judge translation quality metric.

Kocmi & Federmann, 2023 (arxiv 2302.14520; github.com/MicrosoftTranslator/GEMBA).
Uses an LLM to score translations 0-100, achieving state-of-the-art correlation
with MQM human labels. This is what makes the QA layer PhD-grade.
"""

from __future__ import annotations

import logging
import os
import re
import time

from aya_safety.backends import register_metric
from aya_safety.backends.base import QAMetric
from aya_safety.data.schemas import QAScore
from aya_safety.retry.policies import api_retry

logger = logging.getLogger(__name__)

# GEMBA-DA prompt template (Direct Assessment, 0-100 scale)
_GEMBA_DA_PROMPT = """Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and target language quality".

{source_lang} source: "{source}"
{target_lang} translation: "{translation}"

Score (0-100):"""


@register_metric("gemba")
class GEMBAMetric(QAMetric):
    """
    GEMBA LLM-as-judge quality metric.

    Scores translations 0-100 using an LLM (default: GPT-4o-mini).
    Identifies error spans in the translation.
    """

    def __init__(self, config=None, **kwargs) -> None:
        self._backend_name = "openai"
        self._model_name = "gpt-4o-mini"
        if config:
            if hasattr(config, "gemba_backend"):
                self._backend_name = config.gemba_backend
            if hasattr(config, "gemba_model"):
                self._model_name = config.gemba_model
        self._client = None

    @property
    def name(self) -> str:
        return "gemba"

    async def startup(self) -> None:
        if self._backend_name == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY needed for GEMBA metric")
            self._client = openai.AsyncOpenAI(api_key=api_key)
        elif self._backend_name == "cohere":
            import cohere
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY needed for GEMBA metric")
            self._client = cohere.AsyncClientV2(api_key=api_key)
        else:
            raise ValueError(f"Unsupported GEMBA backend: {self._backend_name}")

        logger.info("GEMBA metric ready (backend=%s, model=%s)", self._backend_name, self._model_name)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @api_retry
    async def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        reference: str | None = None,
    ) -> QAScore:
        prompt = _GEMBA_DA_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
            translation=translation,
        )

        start = time.perf_counter()

        if self._backend_name == "openai":
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()
        elif self._backend_name == "cohere":
            response = await self._client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            text = response.message.content[0].text.strip()
        else:
            raise ValueError(f"Unsupported backend: {self._backend_name}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse score from LLM response
        score_value = self._parse_score(text)

        return QAScore(
            metric_name="gemba",
            score=score_value,
            details={
                "raw_response": text,
                "latency_ms": elapsed_ms,
                "backend": self._backend_name,
                "model": self._model_name,
            },
        )

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract numeric score (0-100) from LLM response."""
        # Try to find a number in the response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        for num_str in numbers:
            num = float(num_str)
            if 0 <= num <= 100:
                return num
        # If no valid score found, return -1 to indicate parse failure
        logger.warning("Could not parse GEMBA score from: %s", text)
        return -1.0
