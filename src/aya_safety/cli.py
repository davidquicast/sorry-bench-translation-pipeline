"""
CLI interface: argparse-based with flags that override YAML config values.

Usage:
    aya-safety translate --config configs/default.yaml --target-languages ar,bn,de
    python -m aya_safety translate --tier2-enabled false --resume
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any


def _parse_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _parse_list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aya-safety",
        description="PhD-grade multilingual translation framework for NLP research",
    )
    sub = parser.add_subparsers(dest="command")

    # --- translate subcommand ---
    tr = sub.add_parser("translate", help="Run the translation pipeline")

    # General
    tr.add_argument("--config", "-c", type=str, default=None,
                     help="Path to YAML config file")
    tr.add_argument("--resume", action="store_true",
                     help="Resume from checkpoint")
    tr.add_argument("--force-resume", action="store_true",
                     help="Resume even if config hash changed")

    # Input
    tr.add_argument("--source", type=str, help="HF dataset ID or local file path")
    tr.add_argument("--source-language", type=str, help="Source language ISO code")
    tr.add_argument("--source-text-column", type=str, help="Column name for source text")
    tr.add_argument("--id-column", type=str, help="Column name for record IDs")

    # Languages
    tr.add_argument("--target-languages", type=_parse_list,
                     help="Comma-separated target language ISO codes")
    tr.add_argument("--add-language", type=str, action="append", default=[],
                     help="Add a custom language (format: nllb_code:iso:bcp47:name:script)")

    # Tiers
    tr.add_argument("--tier2-enabled", type=_parse_bool, help="Enable Tier 2 LLM refinement")
    tr.add_argument("--tier2-backend", type=str, help="Primary Tier 2 backend name")
    tr.add_argument("--tier2-model", type=str,
                     help="Model name for the primary Tier 2 backend")
    tr.add_argument("--tier2-fallback", type=_parse_list,
                     help="Comma-separated fallback backend names")
    tr.add_argument("--tier3-enabled", type=_parse_bool, help="Enable Tier 3 QA")
    tr.add_argument("--tier3-metrics", type=_parse_list,
                     help="Comma-separated metric names")
    tr.add_argument("--tier1-backend", type=str, help="Tier 1 backend (nllb or nllb-ct2)")
    tr.add_argument("--tier1-model", type=str, help="Tier 1 model name or 'auto'")

    # Processing
    tr.add_argument("--batch-size", type=int, help="Batch size for Tier 1")
    tr.add_argument("--max-concurrent", type=int, help="Max concurrent Tier 2 API calls")
    tr.add_argument("--device", type=str, help="Device for local models (auto/cuda/cpu)")

    # Output
    tr.add_argument("--output-dir", type=str, help="Output directory")
    tr.add_argument("--output-format", type=str, choices=["parquet", "jsonl", "csv"],
                     help="Output format")
    tr.add_argument("--push-to-hub", action="store_true", help="Push output to HF Hub")
    tr.add_argument("--hub-repo-id", type=str, help="HF Hub repository ID")

    # Logging
    tr.add_argument("--log-level", type=str,
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging level")

    return parser


# Map CLI arg names → config dot-paths
_CLI_TO_CONFIG: dict[str, str] = {
    "source": "input.source",
    "source_language": "input.source_language",
    "source_text_column": "input.source_text_column",
    "id_column": "input.id_column",
    "target_languages": "target_languages",
    "tier2_enabled": "tiers.tier2.enabled",
    "tier2_backend": "tiers.tier2.backend",
    "tier2_fallback": "tiers.tier2.fallback_order",
    "tier3_enabled": "tiers.tier3.enabled",
    "tier3_metrics": "tiers.tier3.metrics",
    "tier1_backend": "tiers.tier1.backend",
    "tier1_model": "tiers.tier1.model",
    "batch_size": "tiers.tier1.batch_size",
    "max_concurrent": "tiers.tier2.max_concurrent",
    "device": "tiers.tier1.device",
    "output_dir": "output.directory",
    "output_format": "output.format",
    "push_to_hub": "output.push_to_hub",
    "hub_repo_id": "output.hub_repo_id",
    "log_level": "logging.level",
}


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed CLI args to config dot-path overrides."""
    overrides: dict[str, Any] = {}
    for cli_name, dot_path in _CLI_TO_CONFIG.items():
        value = getattr(args, cli_name, None)
        if value is not None:
            overrides[dot_path] = value

    # Handle --tier2-model specially: sets model for the chosen backend
    if args.tier2_model and args.tier2_backend:
        overrides[f"tiers.tier2.model.{args.tier2_backend}"] = args.tier2_model
    elif args.tier2_model:
        overrides["tiers.tier2.model.cohere"] = args.tier2_model

    return overrides


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "translate":
        _run_translate(args)


def _run_translate(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from aya_safety.config import load_config
    from aya_safety.utils.logging import setup_logging

    overrides = _build_overrides(args)
    config = load_config(yaml_path=args.config, cli_overrides=overrides)

    setup_logging(config.logging.level, config.logging.file)

    # Register custom languages from CLI
    if args.add_language:
        from aya_safety.config import CustomLanguage
        for lang_str in args.add_language:
            parts = lang_str.split(":")
            if len(parts) < 5:
                print(f"Error: --add-language format is nllb_code:iso:bcp47:name:script, got '{lang_str}'")
                sys.exit(1)
            config.custom_languages.append(CustomLanguage(
                nllb_code=parts[0], iso_639_1=parts[1], bcp47=parts[2],
                display_name=parts[3], script=parts[4],
            ))

    from aya_safety.pipeline import TranslationPipeline

    pipeline = TranslationPipeline(
        config=config,
        resume=args.resume,
        force_resume=args.force_resume,
    )

    asyncio.run(pipeline.run())
