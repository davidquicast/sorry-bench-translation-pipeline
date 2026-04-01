"""
Checkpoint manager: atomic save/load/resume for fault-tolerant pipeline execution.

Checkpoint files are per-language JSONL with companion metadata JSON.
Atomic writes: write to .tmp, then os.replace() to prevent corruption.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from aya_safety.config import CheckpointConfig
from aya_safety.data.schemas import PipelineRecord

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages per-language checkpoints with atomic writes and resume logic."""

    def __init__(self, config: CheckpointConfig, run_id: str, config_hash: str) -> None:
        self.config = config
        self.run_id = run_id
        self.config_hash = config_hash
        self._dir = Path(config.directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records_since_save: dict[str, int] = {}

    def _data_path(self, target_lang: str) -> Path:
        return self._dir / f"{self.run_id}_{target_lang}.jsonl"

    def _meta_path(self, target_lang: str) -> Path:
        return self._dir / f"{self.run_id}_{target_lang}.meta.json"

    def save(
        self,
        records: list[PipelineRecord],
        target_lang: str,
        tier_completed: int,
        total_records: int,
    ) -> None:
        """
        Atomically save checkpoint data and metadata.

        Overwrites the entire checkpoint with current state (not append).
        """
        if not self.config.enabled:
            return

        data_path = self._data_path(target_lang)
        meta_path = self._meta_path(target_lang)

        # Write data (JSONL)
        tmp_data = data_path.with_suffix(".tmp")
        with open(tmp_data, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        os.replace(tmp_data, data_path)

        # Write metadata
        completed = sum(1 for r in records if r.status.value != "pending")
        meta = {
            "run_id": self.run_id,
            "target_lang": target_lang,
            "total_records": total_records,
            "completed_records": completed,
            "tier_completed": tier_completed,
            "config_hash": self.config_hash,
            "last_checkpoint_time": datetime.now(timezone.utc).isoformat(),
        }
        tmp_meta = meta_path.with_suffix(".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path)

        self._records_since_save[target_lang] = 0
        logger.debug(
            "Checkpoint saved: %s, %d/%d records, tier %d",
            target_lang, completed, total_records, tier_completed,
        )

    def load(self, target_lang: str) -> tuple[list[PipelineRecord], dict | None]:
        """
        Load checkpoint for a language.

        Returns:
            (records, metadata) — metadata is None if no checkpoint exists.
        """
        data_path = self._data_path(target_lang)
        meta_path = self._meta_path(target_lang)

        if not data_path.exists():
            return [], None

        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(PipelineRecord.model_validate_json(line))

        meta = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        logger.info(
            "Loaded checkpoint: %s, %d records, tier %s",
            target_lang, len(records), meta.get("tier_completed") if meta else "?",
        )
        return records, meta

    def get_completed_ids(self, target_lang: str) -> set[str]:
        """Return record IDs that are already completed for this language."""
        records, _ = self.load(target_lang)
        return {r.record_id for r in records if r.status.value != "pending"}

    def get_tier_completed(self, target_lang: str) -> int:
        """Return the highest tier completed for this language (0 if none)."""
        _, meta = self.load(target_lang)
        if meta is None:
            return 0
        return meta.get("tier_completed", 0)

    def check_config_compatibility(self, target_lang: str) -> bool:
        """Check if the stored config hash matches the current config."""
        _, meta = self.load(target_lang)
        if meta is None:
            return True
        return meta.get("config_hash") == self.config_hash

    def should_save(self, target_lang: str, new_records: int = 1) -> bool:
        """Whether enough records have been processed since last save."""
        self._records_since_save[target_lang] = (
            self._records_since_save.get(target_lang, 0) + new_records
        )
        return self._records_since_save[target_lang] >= self.config.save_every_n

    def clear(self, target_lang: str) -> None:
        """Remove checkpoint files for a language."""
        for path in (self._data_path(target_lang), self._meta_path(target_lang)):
            if path.exists():
                path.unlink()
        logger.info("Cleared checkpoint for %s", target_lang)

    def find_existing_run(self) -> str | None:
        """Find an existing run ID from checkpoint files in the directory."""
        for path in self._dir.glob("*.meta.json"):
            name = path.stem  # e.g., "20260324_143022_ar"
            # Strip the .meta suffix already handled by .stem
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return meta.get("run_id")
            except (json.JSONDecodeError, KeyError):
                continue
        return None
