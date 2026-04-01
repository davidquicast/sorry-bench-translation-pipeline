"""
TranslationPipeline: top-level orchestrator wiring all three tiers.

Data flow:
    Input → Tier 1 (NLLB) → Tier 2 (LLM refinement) → Tier 3 (QA) → Output

Languages are processed sequentially. Within each language, batches use
controlled concurrency for API calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from aya_safety.backends import create_backend, create_metric
from aya_safety.backends.base import LLMBackend, MTBackend, QAMetric
from aya_safety.checkpoint.manager import CheckpointManager
from aya_safety.config import PipelineConfig
from aya_safety.data.loader import load_dataset
from aya_safety.data.schemas import (
    PipelineRecord,
    QAScore,
    TranslationRecord,
    TranslationStatus,
)
from aya_safety.data.writer import push_to_hub, write_output
from aya_safety.languages.definitions import LanguageSpec
from aya_safety.languages.registry import LanguageRegistry
from aya_safety.reporting.report import generate_report
from aya_safety.retry.rate_limiter import BackendRateLimiter
from aya_safety.retry.refusal import RefusalDetector

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    Main pipeline orchestrator.

    Manages the lifecycle of backends, coordinates the three tiers,
    handles checkpointing, and generates the final report.
    """

    def __init__(
        self,
        config: PipelineConfig,
        resume: bool = False,
        force_resume: bool = False,
    ) -> None:
        self.config = config
        self._resume = resume
        self._force_resume = force_resume

        # Initialize language registry
        self._lang_registry = LanguageRegistry()
        for custom in config.custom_languages:
            self._lang_registry.register(LanguageSpec(
                nllb_code=custom.nllb_code,
                iso_639_1=custom.iso_639_1,
                bcp47=custom.bcp47,
                display_name=custom.display_name,
                script=custom.script,
                resource_level=custom.resource_level,
                group="custom",
                recommended_models=tuple(custom.recommended_models),
            ))

        # Generate or resume run ID
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._checkpoint = CheckpointManager(
            config=config.checkpoint,
            run_id=self._run_id,
            config_hash=config.config_hash(),
        )

        if self._resume:
            existing_id = self._checkpoint.find_existing_run()
            if existing_id:
                self._run_id = existing_id
                self._checkpoint.run_id = existing_id
                logger.info("Resuming run: %s", self._run_id)

        self._refusal_detector = RefusalDetector()
        self._rate_limiter = BackendRateLimiter(config.tiers.tier2.rate_limits)

        # Backends and metrics (initialized during run)
        self._tier1_backend: MTBackend | None = None
        self._tier2_backends: list[MTBackend | LLMBackend] = []
        self._qa_metrics: list[QAMetric] = []

        # Tracking
        self._all_results: dict[str, list[PipelineRecord]] = {}
        self._start_time: float = 0

    async def run(self) -> None:
        """Main entry point: run the full pipeline for all target languages."""
        self._start_time = time.perf_counter()

        logger.info("=" * 70)
        logger.info("Multilingual Translation Pipeline")
        logger.info("Run ID: %s", self._run_id)
        logger.info("Target languages: %s", ", ".join(self.config.target_languages))
        logger.info(
            "Tiers: 1=%s, 2=%s, 3=%s",
            "ON", "ON" if self.config.tiers.tier2.enabled else "OFF",
            "ON" if self.config.tiers.tier3.enabled else "OFF",
        )
        logger.info("=" * 70)

        # Load source data
        records = load_dataset(self.config.input)
        if not records:
            logger.error("No records loaded. Exiting.")
            return

        # Initialize backends
        await self._init_backends()

        try:
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                lang_task = progress.add_task(
                    "Languages", total=len(self.config.target_languages),
                )

                for target_lang in self.config.target_languages:
                    lang_name = self._lang_registry.resolve_display_name(target_lang)
                    progress.update(lang_task, description=f"[bold]{lang_name} ({target_lang})")

                    pipeline_records = await self._process_language(
                        records, target_lang, progress,
                    )
                    self._all_results[target_lang] = pipeline_records

                    # Write output
                    write_output(pipeline_records, target_lang, self.config.output)

                    progress.advance(lang_task)

            # Generate report
            elapsed = time.perf_counter() - self._start_time
            generate_report(
                self._all_results,
                self.config,
                elapsed_seconds=elapsed,
            )

            # Optional: push to Hub
            if self.config.output.push_to_hub and self.config.output.hub_repo_id:
                push_to_hub(
                    self.config.output.directory,
                    self.config.output.hub_repo_id,
                    self.config.output.hub_token_env,
                )

            logger.info("Pipeline complete in %.1f seconds", elapsed)

        finally:
            await self._shutdown_backends()

    async def _init_backends(self) -> None:
        """Initialize all required backends."""
        # Tier 1
        t1_cfg = self.config.tiers.tier1
        self._tier1_backend = create_backend(
            t1_cfg.backend,
            model=t1_cfg.model,
            device=t1_cfg.device,
            max_length=t1_cfg.max_length,
            batch_size=t1_cfg.batch_size,
            language_registry=self._lang_registry,
        )
        await self._tier1_backend.startup()
        logger.info("Tier 1 backend ready: %s", self._tier1_backend.name)

        # Tier 2
        if self.config.tiers.tier2.enabled:
            t2_cfg = self.config.tiers.tier2
            backend_names = [t2_cfg.backend] + t2_cfg.fallback_order

            for bname in backend_names:
                try:
                    model_name = t2_cfg.model.get(bname, "")
                    backend = create_backend(
                        bname,
                        model=model_name,
                        language_registry=self._lang_registry,
                        temperature=t2_cfg.temperature,
                    )
                    await backend.startup()
                    self._tier2_backends.append(backend)
                    logger.info("Tier 2 backend ready: %s", backend.name)
                except Exception as e:
                    logger.warning("Failed to initialize Tier 2 backend '%s': %s", bname, e)

        # Tier 3
        if self.config.tiers.tier3.enabled:
            t3_cfg = self.config.tiers.tier3
            for metric_name in t3_cfg.metrics:
                try:
                    metric = create_metric(metric_name, config=t3_cfg)
                    await metric.startup()
                    self._qa_metrics.append(metric)
                    logger.info("Tier 3 metric ready: %s", metric.name)
                except Exception as e:
                    logger.warning("Failed to initialize metric '%s': %s", metric_name, e)

    async def _shutdown_backends(self) -> None:
        """Shutdown all backends and metrics."""
        if self._tier1_backend:
            await self._tier1_backend.shutdown()
        for backend in self._tier2_backends:
            await backend.shutdown()
        for metric in self._qa_metrics:
            await metric.shutdown()

    async def _process_language(
        self,
        records: list[TranslationRecord],
        target_lang: str,
        progress: Progress,
    ) -> list[PipelineRecord]:
        """Run the full pipeline for one target language."""

        # Check checkpoint compatibility on resume
        if self._resume:
            if not self._checkpoint.check_config_compatibility(target_lang):
                if not self._force_resume:
                    logger.warning(
                        "Config hash mismatch for %s. Use --force-resume to continue.",
                        target_lang,
                    )
                    return []
                logger.warning("Force-resuming %s despite config hash mismatch", target_lang)

        # Load checkpoint or create fresh pipeline records
        completed_tier = self._checkpoint.get_tier_completed(target_lang)
        existing_records, meta = self._checkpoint.load(target_lang)
        existing_ids = {r.record_id for r in existing_records}

        # Build pipeline records: resume existing + create new
        pipeline_records: list[PipelineRecord] = list(existing_records)
        for record in records:
            if record.record_id not in existing_ids:
                pipeline_records.append(PipelineRecord(
                    record_id=record.record_id,
                    source_text=record.source_text,
                    source_lang=record.source_lang,
                    target_lang=target_lang,
                    metadata=record.metadata,
                ))

        total = len(pipeline_records)
        rec_task = progress.add_task(
            f"  {target_lang}", total=total, visible=True,
        )
        already_done = sum(1 for r in pipeline_records if r.status != TranslationStatus.PENDING)
        progress.update(rec_task, completed=already_done)

        # --- Tier 1 ---
        if completed_tier < 1:
            pending_t1 = [r for r in pipeline_records if r.tier1_result is None]
            if pending_t1:
                logger.info("Tier 1: translating %d records to %s", len(pending_t1), target_lang)
                await self._run_tier1(pending_t1, target_lang, progress, rec_task)
                self._checkpoint.save(pipeline_records, target_lang, 1, total)

        # --- Tier 2 ---
        if self.config.tiers.tier2.enabled and completed_tier < 2:
            pending_t2 = [
                r for r in pipeline_records
                if r.tier1_result is not None and r.tier2_result is None
                and r.status not in (TranslationStatus.TIER2_REFUSED, TranslationStatus.TIER2_FAILED)
            ]
            if pending_t2 and self._tier2_backends:
                logger.info("Tier 2: refining %d records for %s", len(pending_t2), target_lang)
                await self._run_tier2(pending_t2, target_lang, progress, rec_task)
                self._checkpoint.save(pipeline_records, target_lang, 2, total)
        elif not self.config.tiers.tier2.enabled:
            # Set final translation to Tier 1 output
            for r in pipeline_records:
                if r.tier1_result and not r.final_translation:
                    r.final_translation = r.tier1_result.translation

        # --- Tier 3 ---
        if self.config.tiers.tier3.enabled and completed_tier < 3:
            qa_pending = [r for r in pipeline_records if r.final_translation and not r.qa_scores]
            if qa_pending and self._qa_metrics:
                logger.info("Tier 3: evaluating %d records for %s", len(qa_pending), target_lang)
                await self._run_tier3(qa_pending, target_lang)
                self._checkpoint.save(pipeline_records, target_lang, 3, total)

        progress.remove_task(rec_task)

        completed = sum(1 for r in pipeline_records if r.final_translation)
        logger.info(
            "%s complete: %d/%d records translated",
            target_lang, completed, total,
        )

        return pipeline_records

    async def _run_tier1(
        self,
        records: list[PipelineRecord],
        target_lang: str,
        progress: Progress,
        task_id,
    ) -> None:
        """Run Tier 1 (NLLB) translation on pending records."""
        batch_size = self.config.tiers.tier1.batch_size
        source_records = [r.to_translation_record() for r in records]

        for i in range(0, len(source_records), batch_size):
            batch_src = source_records[i:i + batch_size]
            batch_pipe = records[i:i + batch_size]

            try:
                results = await self._tier1_backend.translate_batch(
                    batch_src,
                    batch_src[0].source_lang,
                    target_lang,
                    batch_size=batch_size,
                )

                for pipe_rec, result in zip(batch_pipe, results):
                    pipe_rec.tier1_result = result
                    pipe_rec.final_translation = result.translation
                    pipe_rec.status = TranslationStatus.TIER1_COMPLETE

                progress.advance(task_id, len(batch_pipe))

            except Exception as e:
                logger.error("Tier 1 batch error at offset %d: %s", i, e)
                for pipe_rec in batch_pipe:
                    pipe_rec.status = TranslationStatus.FAILED
                    pipe_rec.metadata["error"] = str(e)

            # Periodic checkpoint
            if self._checkpoint.should_save(target_lang, len(batch_pipe)):
                self._checkpoint.save(
                    records, target_lang, 0,
                    len(records),
                )

    async def _run_tier2(
        self,
        records: list[PipelineRecord],
        target_lang: str,
        progress: Progress,
        task_id,
    ) -> None:
        """Run Tier 2 (LLM refinement) with fallback chain."""
        semaphore = asyncio.Semaphore(self.config.tiers.tier2.max_concurrent)

        async def refine_one(pipe_rec: PipelineRecord) -> None:
            async with semaphore:
                mt_output = pipe_rec.tier1_result.translation
                source_rec = pipe_rec.to_translation_record()
                translated = False

                for backend in self._tier2_backends:
                    if not backend.supports_language(target_lang):
                        continue

                    await self._rate_limiter.acquire(backend.name.split(":")[0])

                    try:
                        if isinstance(backend, LLMBackend):
                            result = await backend.refine(
                                source_rec, mt_output,
                                pipe_rec.source_lang, target_lang,
                            )
                        else:
                            result = await backend.translate(
                                source_rec, pipe_rec.source_lang, target_lang,
                            )

                        # Check for refusal
                        if self._refusal_detector.is_refusal(
                            result.translation, target_lang, pipe_rec.source_text,
                        ):
                            result.refusal_detected = True
                            # Try to extract translation from mixed response
                            extracted = self._refusal_detector.extract_translation_if_mixed(
                                result.translation,
                            )
                            if extracted:
                                result.translation = extracted
                                result.refusal_detected = False
                            else:
                                logger.debug(
                                    "Refusal detected from %s for record %s, trying next backend",
                                    backend.name, pipe_rec.record_id,
                                )
                                continue

                        pipe_rec.tier2_result = result
                        pipe_rec.final_translation = result.translation
                        pipe_rec.status = TranslationStatus.TIER2_COMPLETE
                        translated = True
                        break

                    except Exception as e:
                        logger.warning(
                            "Tier 2 error from %s for record %s: %s",
                            backend.name, pipe_rec.record_id, e,
                        )
                        continue

                if not translated:
                    # All backends failed or refused — keep Tier 1 output
                    pipe_rec.final_translation = mt_output
                    pipe_rec.status = TranslationStatus.TIER2_REFUSED
                    logger.debug("All Tier 2 backends refused/failed for record %s", pipe_rec.record_id)

                progress.advance(task_id)

        # Run all refinements with concurrency control
        tasks = [refine_one(r) for r in records]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_tier3(
        self,
        records: list[PipelineRecord],
        target_lang: str,
    ) -> None:
        """Run all QA metrics on translated records."""
        sources = [r.source_text for r in records]
        translations = [r.final_translation for r in records]
        thresholds = self.config.tiers.tier3.thresholds

        threshold_map = {
            "comet": thresholds.comet_min,
            "chrf": thresholds.chrf_min,
            "backtranslation": thresholds.bertscore_min,
            "gemba": thresholds.gemba_min,
        }

        for metric in self._qa_metrics:
            try:
                scores = await metric.score_batch(
                    sources, translations,
                    records[0].source_lang, target_lang,
                )
                for record, score in zip(records, scores):
                    record.qa_scores.append(score)

                    # Flag if below threshold
                    min_score = threshold_map.get(metric.name)
                    if min_score is not None and score.score < min_score:
                        record.flagged = True
                        record.flag_reasons.append(
                            f"{metric.name}={score.score:.3f} < {min_score}"
                        )

            except Exception as e:
                logger.error("Tier 3 metric '%s' failed: %s", metric.name, e)

        # Update status
        for record in records:
            if record.qa_scores:
                record.status = TranslationStatus.QA_COMPLETE
