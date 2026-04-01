"""
Final report generation: JSON summary + flagged records CSV.

Reports include per-language completion rate, failure/refusal counts per model,
average QA scores, total processing time, and estimated API cost.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from aya_safety.config import PipelineConfig
from aya_safety.data.schemas import PipelineRecord, TranslationStatus

logger = logging.getLogger(__name__)

# Approximate cost per 1K tokens (input + output) by backend
_COST_PER_1K_TOKENS: dict[str, float] = {
    "cohere": 0.002,      # Aya Expanse ~$0.50/$1.50 per M tokens
    "openai": 0.00075,    # GPT-4o-mini ~$0.15/$0.60 per M tokens
    "anthropic": 0.003,   # Claude 3.5 Haiku
    "gemini": 0.00075,    # Gemini 2.5 Flash
}


def generate_report(
    all_results: dict[str, list[PipelineRecord]],
    config: PipelineConfig,
    elapsed_seconds: float,
) -> None:
    """Generate the final pipeline report (JSON + flagged CSV)."""

    report: dict = {
        "summary": {},
        "per_language": {},
        "per_backend": {},
        "qa_metrics": {},
        "timing": {},
        "cost_estimate": {},
    }

    total_records = 0
    total_completed = 0
    total_failed = 0
    total_refused = 0
    total_flagged = 0
    all_flagged_records: list[dict] = []

    backend_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "refusals": 0, "failures": 0,
        "latency_sum": 0.0, "tokens_in": 0, "tokens_out": 0,
    })

    qa_scores_by_metric: dict[str, list[float]] = defaultdict(list)

    for target_lang, records in all_results.items():
        lang_total = len(records)
        lang_completed = 0
        lang_failed = 0
        lang_refused = 0
        lang_flagged = 0

        for record in records:
            total_records += 1

            if record.final_translation:
                lang_completed += 1
                total_completed += 1

            if record.status == TranslationStatus.FAILED:
                lang_failed += 1
                total_failed += 1

            if record.status in (TranslationStatus.TIER2_REFUSED, TranslationStatus.TIER2_FAILED):
                lang_refused += 1
                total_refused += 1

            if record.flagged:
                lang_flagged += 1
                total_flagged += 1
                all_flagged_records.append({
                    "record_id": record.record_id,
                    "target_lang": target_lang,
                    "source_text": record.source_text[:200],
                    "final_translation": (record.final_translation or "")[:200],
                    "status": record.status.value,
                    "flag_reasons": "; ".join(record.flag_reasons),
                    **{f"qa_{s.metric_name}": s.score for s in record.qa_scores},
                })

            # Backend stats
            for result in [record.tier1_result, record.tier2_result]:
                if result:
                    bname = result.backend_name.split(":")[0]
                    backend_stats[bname]["total"] += 1
                    backend_stats[bname]["latency_sum"] += result.latency_ms
                    backend_stats[bname]["tokens_in"] += result.token_count_in
                    backend_stats[bname]["tokens_out"] += result.token_count_out
                    if result.refusal_detected:
                        backend_stats[bname]["refusals"] += 1
                    if result.status in (TranslationStatus.FAILED, TranslationStatus.TIER2_FAILED):
                        backend_stats[bname]["failures"] += 1

            # QA scores
            for qa in record.qa_scores:
                qa_scores_by_metric[qa.metric_name].append(qa.score)

        report["per_language"][target_lang] = {
            "total": lang_total,
            "completed": lang_completed,
            "completion_rate": lang_completed / lang_total if lang_total else 0,
            "failed": lang_failed,
            "refused": lang_refused,
            "flagged": lang_flagged,
        }

    # Summary
    report["summary"] = {
        "total_records": total_records,
        "total_completed": total_completed,
        "completion_rate": total_completed / total_records if total_records else 0,
        "total_failed": total_failed,
        "total_refused": total_refused,
        "total_flagged": total_flagged,
        "languages_processed": len(all_results),
    }

    # Per-backend
    for bname, stats in backend_stats.items():
        total = stats["total"] or 1
        report["per_backend"][bname] = {
            "total_calls": stats["total"],
            "refusals": stats["refusals"],
            "refusal_rate": stats["refusals"] / total,
            "failures": stats["failures"],
            "avg_latency_ms": stats["latency_sum"] / total,
            "total_tokens_in": stats["tokens_in"],
            "total_tokens_out": stats["tokens_out"],
        }

    # QA metrics
    for metric_name, scores in qa_scores_by_metric.items():
        valid_scores = [s for s in scores if s >= 0]
        if valid_scores:
            report["qa_metrics"][metric_name] = {
                "mean": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
                "count": len(valid_scores),
            }

    # Timing
    report["timing"] = {
        "total_seconds": elapsed_seconds,
        "total_minutes": elapsed_seconds / 60,
        "records_per_second": total_records / elapsed_seconds if elapsed_seconds else 0,
    }

    # Cost estimate
    total_cost = 0.0
    for bname, stats in backend_stats.items():
        base_name = bname.split(":")[0]
        cost_rate = _COST_PER_1K_TOKENS.get(base_name, 0)
        total_tokens = stats["tokens_in"] + stats["tokens_out"]
        cost = (total_tokens / 1000) * cost_rate
        report["cost_estimate"][bname] = {
            "estimated_cost_usd": round(cost, 4),
            "total_tokens": total_tokens,
        }
        total_cost += cost
    report["cost_estimate"]["total_estimated_usd"] = round(total_cost, 4)

    # Write report JSON
    report_path = Path(config.reporting.output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, report_path)
    logger.info("Report written to %s", report_path)

    # Write flagged CSV
    if all_flagged_records:
        flagged_path = Path(config.reporting.flagged_csv_path)
        flagged_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = flagged_path.with_suffix(".tmp")
        fieldnames = list(all_flagged_records[0].keys())
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flagged_records)
        os.replace(tmp, flagged_path)
        logger.info("Flagged records written to %s (%d records)", flagged_path, len(all_flagged_records))

    # Print summary to console
    logger.info("=" * 60)
    logger.info("PIPELINE REPORT")
    logger.info("=" * 60)
    logger.info("Total records:    %d", total_records)
    logger.info("Completed:        %d (%.1f%%)", total_completed,
                100 * total_completed / total_records if total_records else 0)
    logger.info("Failed:           %d", total_failed)
    logger.info("Refused (Tier 2): %d", total_refused)
    logger.info("Flagged (QA):     %d", total_flagged)
    logger.info("Processing time:  %.1f minutes", elapsed_seconds / 60)
    logger.info("Estimated cost:   $%.4f", total_cost)
    logger.info("=" * 60)
