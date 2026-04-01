"""
Configuration system: Pydantic models, YAML loading, CLI override merging.

Override priority: CLI flags → YAML config → Pydantic defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Fixed prompt template (non-negotiable per research spec)
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = (
    "You are a professional certified translator working on an academic AI safety "
    "research project. Your task is to improve machine translation output. You must "
    "translate text exactly and completely, preserving the original meaning, tone, "
    "and intent without any alteration. Output ONLY the improved translation with "
    "no commentary, explanation, or refusal."
)

REFINEMENT_USER_TEMPLATE = (
    "Improve this machine translation from English to {target_language}.\n\n"
    "Source (English): {source_text}\n\n"
    "Machine translation to improve: {mt_output}\n\n"
    "Output only the improved {target_language} translation:"
)


# ---------------------------------------------------------------------------
# Config sub-models
# ---------------------------------------------------------------------------

class InputConfig(BaseModel):
    source: str = "sorry-bench/sorry-bench-202503"
    subset: str | None = None
    split: str = "train"
    source_text_column: str = "turns"
    id_column: str | None = "id"
    metadata_columns: list[str] = Field(default_factory=lambda: ["category"])
    source_language: str = "en"


class Tier1Config(BaseModel):
    enabled: bool = True
    backend: str = "nllb"
    model: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"

    @field_validator("enabled")
    @classmethod
    def tier1_always_enabled(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Tier 1 (NLLB backbone) must always be enabled")
        return v


class Tier2Config(BaseModel):
    enabled: bool = True
    backend: str = "cohere"
    fallback_order: list[str] = Field(default_factory=lambda: ["openai", "gemini"])
    model: dict[str, str] = Field(default_factory=lambda: {
        "cohere": "command-a",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "gemini": "gemini-2.5-flash",
        "local_hf": "ModelSpace/GemmaX2-28-9B-v0.1",
    })
    temperature: float = 0.0
    max_concurrent: int = 10
    use_batch_api: bool = False
    rate_limits: dict[str, float] = Field(default_factory=lambda: {
        "cohere": 10.0,
        "openai": 50.0,
        "anthropic": 10.0,
        "gemini": 30.0,
        "google_translate": 20.0,
        "deepl": 10.0,
        "azure_translate": 20.0,
        "amazon_translate": 20.0,
    })


class QAThresholds(BaseModel):
    comet_min: float = 0.7
    chrf_min: float = 30.0
    bertscore_min: float = 0.8
    gemba_min: float = 50.0
    langdetect_confidence: float = 0.8


class Tier3Config(BaseModel):
    enabled: bool = True
    metrics: list[str] = Field(default_factory=lambda: [
        "comet", "chrf", "backtranslation", "gemba", "langdetect", "safety_consistency",
    ])
    comet_model: str = "Unbabel/wmt22-cometkiwi-da"
    gemba_backend: str = "openai"
    gemba_model: str = "gpt-4o-mini"
    backtranslation_backend: str = "nllb"
    thresholds: QAThresholds = Field(default_factory=QAThresholds)


class TiersConfig(BaseModel):
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    tier2: Tier2Config = Field(default_factory=Tier2Config)
    tier3: Tier3Config = Field(default_factory=Tier3Config)


class RetryConfig(BaseModel):
    max_attempts: int = 5
    backoff_multiplier: float = 1.0
    backoff_max: float = 60.0
    jitter: bool = True


class CheckpointConfig(BaseModel):
    enabled: bool = True
    directory: str = "./checkpoints"
    save_every_n: int = 100
    format: str = "jsonl"


class OutputConfig(BaseModel):
    directory: str = "./output"
    format: str = "parquet"
    include_all_tiers: bool = True
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_token_env: str = "HF_TOKEN"


class ReportingConfig(BaseModel):
    output_path: str = "./output/report.json"
    flagged_csv_path: str = "./output/flagged.csv"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = "./logs/pipeline.log"


class CustomLanguage(BaseModel):
    """For registering languages not in the built-in registry via config."""
    nllb_code: str
    iso_639_1: str
    bcp47: str
    display_name: str
    script: str
    resource_level: str = "low"
    recommended_models: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    target_languages: list[str] = Field(default_factory=lambda: [
        "ar", "bn", "de", "fr", "hi", "ja", "ru", "es", "zh",
    ])
    tiers: TiersConfig = Field(default_factory=TiersConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    custom_languages: list[CustomLanguage] = Field(default_factory=list)

    def config_hash(self) -> str:
        """Deterministic hash for checkpoint compatibility checks."""
        data = self.model_dump_json(exclude={"logging", "checkpoint"})
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------

def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR} references in string values."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Override values win."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(d: dict, dot_path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation path."""
    keys = dot_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def load_config(
    yaml_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """
    Load configuration with override priority: CLI → YAML → defaults.

    Args:
        yaml_path: Path to YAML config file (optional).
        cli_overrides: Dict of dot-path → value overrides from CLI.

    Returns:
        Validated PipelineConfig instance.
    """
    base: dict[str, Any] = {}

    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f)
            if loaded:
                base = _resolve_env_vars(loaded)

    if cli_overrides:
        for dot_path, value in cli_overrides.items():
            _set_nested(base, dot_path, value)

    return PipelineConfig(**base)
