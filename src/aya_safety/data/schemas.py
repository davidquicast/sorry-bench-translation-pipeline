"""
Core data models used throughout the pipeline.

TranslationRecord → (Tier 1) → TranslationResult → (Tier 2) → TranslationResult
    → PipelineRecord aggregates everything for one source×target pair.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TranslationStatus(str, Enum):
    PENDING = "pending"
    TIER1_COMPLETE = "tier1_complete"
    TIER2_COMPLETE = "tier2_complete"
    TIER2_REFUSED = "tier2_refused"
    TIER2_FAILED = "tier2_failed"
    QA_COMPLETE = "qa_complete"
    FAILED = "failed"


class TranslationRecord(BaseModel):
    """A single source record to translate."""
    record_id: str
    source_text: str
    source_lang: str
    metadata: dict = Field(default_factory=dict)


class TranslationResult(BaseModel):
    """Output from a single translation backend (Tier 1 or Tier 2)."""
    record_id: str
    target_lang: str
    translation: str
    backend_name: str
    status: TranslationStatus
    refusal_detected: bool = False
    error_message: str | None = None
    latency_ms: float = 0.0
    token_count_in: int = 0
    token_count_out: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QAScore(BaseModel):
    """A single quality metric score."""
    metric_name: str
    score: float
    details: dict = Field(default_factory=dict)


class PipelineRecord(BaseModel):
    """Full record tracking a source through all tiers for one target language."""
    record_id: str
    source_text: str
    source_lang: str
    target_lang: str
    tier1_result: TranslationResult | None = None
    tier2_result: TranslationResult | None = None
    final_translation: str | None = None
    qa_scores: list[QAScore] = Field(default_factory=list)
    status: TranslationStatus = TranslationStatus.PENDING
    metadata: dict = Field(default_factory=dict)
    flagged: bool = False
    flag_reasons: list[str] = Field(default_factory=list)

    def to_translation_record(self) -> TranslationRecord:
        return TranslationRecord(
            record_id=self.record_id,
            source_text=self.source_text,
            source_lang=self.source_lang,
            metadata=self.metadata,
        )
