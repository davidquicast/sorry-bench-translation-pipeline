"""
Dataset loading: auto-detect HuggingFace dataset ID vs local file path.

Supports CSV, JSON, JSONL, Parquet local files and any HF dataset via
datasets.load_dataset().
"""

from __future__ import annotations

import logging
from pathlib import Path

from aya_safety.config import InputConfig
from aya_safety.data.schemas import TranslationRecord

logger = logging.getLogger(__name__)

# File extensions → HF datasets format string
_EXT_TO_FORMAT = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
    ".tsv": "csv",
}


def _is_local_path(source: str) -> bool:
    """Heuristic: local path if it contains a file extension or path separator."""
    p = Path(source)
    return p.suffix in _EXT_TO_FORMAT or p.exists()


def _extract_text(value) -> str:
    """Extract text from potentially nested structures (e.g. SorryBench 'turns' column)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # SorryBench stores turns as list of strings; take the first
        if value and isinstance(value[0], str):
            return value[0]
        # Nested list: [[str]]
        if value and isinstance(value[0], list) and value[0]:
            return value[0][0]
    return str(value)


def load_dataset(config: InputConfig) -> list[TranslationRecord]:
    """
    Load source dataset and convert to list[TranslationRecord].

    Auto-detects HuggingFace dataset ID vs local file path.
    """
    from datasets import load_dataset as hf_load

    source = config.source
    records: list[TranslationRecord] = []

    if _is_local_path(source):
        path = Path(source)
        fmt = _EXT_TO_FORMAT.get(path.suffix)
        if fmt is None:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        logger.info("Loading local file: %s (format=%s)", path, fmt)
        kwargs = {"data_files": str(path)}
        if fmt == "csv" and path.suffix == ".tsv":
            kwargs["delimiter"] = "\t"
        ds = hf_load(fmt, split="train", **kwargs)
    else:
        logger.info("Loading HuggingFace dataset: %s", source)
        load_kwargs = {}
        if config.subset:
            load_kwargs["name"] = config.subset
        ds = hf_load(source, split=config.split, **load_kwargs)

    logger.info("Loaded %d records", len(ds))

    text_col = config.source_text_column
    id_col = config.id_column
    meta_cols = config.metadata_columns

    if text_col not in ds.column_names:
        raise ValueError(
            f"Source text column '{text_col}' not found. "
            f"Available columns: {ds.column_names}"
        )

    for idx, row in enumerate(ds):
        record_id = str(row[id_col]) if id_col and id_col in ds.column_names else str(idx)
        source_text = _extract_text(row[text_col])

        metadata = {}
        for col in meta_cols:
            if col in ds.column_names:
                metadata[col] = row[col]

        records.append(TranslationRecord(
            record_id=record_id,
            source_text=source_text,
            source_lang=config.source_language,
            metadata=metadata,
        ))

    logger.info("Created %d TranslationRecords (source_lang=%s)", len(records), config.source_language)
    return records
