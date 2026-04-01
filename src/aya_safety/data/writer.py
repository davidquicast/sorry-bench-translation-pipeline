"""
Output writing: JSONL, Parquet, CSV formats + optional HuggingFace Hub push.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from aya_safety.config import OutputConfig
from aya_safety.data.schemas import PipelineRecord

logger = logging.getLogger(__name__)


def _records_to_dicts(records: list[PipelineRecord], include_all_tiers: bool) -> list[dict]:
    """Flatten PipelineRecords to serializable dicts."""
    rows = []
    for r in records:
        row = {
            "record_id": r.record_id,
            "source_text": r.source_text,
            "source_lang": r.source_lang,
            "target_lang": r.target_lang,
            "final_translation": r.final_translation,
            "status": r.status.value,
            "flagged": r.flagged,
        }
        # Metadata
        for k, v in r.metadata.items():
            row[f"meta_{k}"] = v

        if include_all_tiers:
            if r.tier1_result:
                row["tier1_translation"] = r.tier1_result.translation
                row["tier1_backend"] = r.tier1_result.backend_name
                row["tier1_latency_ms"] = r.tier1_result.latency_ms
            if r.tier2_result:
                row["tier2_translation"] = r.tier2_result.translation
                row["tier2_backend"] = r.tier2_result.backend_name
                row["tier2_refusal"] = r.tier2_result.refusal_detected
                row["tier2_latency_ms"] = r.tier2_result.latency_ms

        # QA scores as individual columns
        for qa in r.qa_scores:
            row[f"qa_{qa.metric_name}"] = qa.score

        if r.flag_reasons:
            row["flag_reasons"] = "; ".join(r.flag_reasons)

        rows.append(row)
    return rows


def write_output(
    records: list[PipelineRecord],
    target_lang: str,
    config: OutputConfig,
) -> Path:
    """Write translated records to disk in the configured format."""
    out_dir = Path(config.directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _records_to_dicts(records, config.include_all_tiers)
    filename = f"translations_{target_lang}.{config.format}"
    out_path = out_dir / filename

    if config.format == "jsonl":
        _write_jsonl(rows, out_path)
    elif config.format == "csv":
        _write_csv(rows, out_path)
    elif config.format == "parquet":
        _write_parquet(rows, out_path)
    else:
        raise ValueError(f"Unsupported output format: {config.format}")

    logger.info("Wrote %d records to %s", len(rows), out_path)
    return out_path


def _write_jsonl(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.touch()
        return
    tmp = path.with_suffix(".tmp")
    fieldnames = list(rows[0].keys())
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        # Write empty parquet
        table = pa.table({})
        pq.write_table(table, str(path))
        return
    table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)


def push_to_hub(
    output_dir: str,
    repo_id: str,
    token_env: str = "HF_TOKEN",
) -> None:
    """Push all output files as a HuggingFace dataset."""
    from datasets import Dataset, DatasetDict

    token = os.environ.get(token_env)
    if not token:
        raise ValueError(f"Cannot push to Hub: {token_env} environment variable not set")

    out_path = Path(output_dir)
    splits = {}

    for f in sorted(out_path.glob("translations_*")):
        lang = f.stem.replace("translations_", "")
        if f.suffix == ".parquet":
            ds = Dataset.from_parquet(str(f))
        elif f.suffix == ".jsonl":
            ds = Dataset.from_json(str(f))
        elif f.suffix == ".csv":
            ds = Dataset.from_csv(str(f))
        else:
            continue
        splits[lang] = ds

    if not splits:
        logger.warning("No output files found in %s", out_path)
        return

    dd = DatasetDict(splits)
    dd.push_to_hub(repo_id, token=token)
    logger.info("Pushed dataset to HuggingFace Hub: %s", repo_id)
