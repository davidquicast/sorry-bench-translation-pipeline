"""Tests for the configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from aya_safety.config import PipelineConfig, load_config


def test_default_config():
    """Default config should be valid."""
    config = PipelineConfig()
    assert config.tiers.tier1.enabled is True
    assert len(config.target_languages) == 9
    assert "ar" in config.target_languages
    assert config.tiers.tier2.temperature == 0.0


def test_tier1_always_enabled():
    """Tier 1 cannot be disabled."""
    with pytest.raises(Exception):
        PipelineConfig(tiers={"tier1": {"enabled": False}})


def test_load_from_yaml():
    """Config loads from YAML file."""
    cfg = {
        "target_languages": ["es"],
        "tiers": {
            "tier1": {"batch_size": 64},
            "tier2": {"enabled": False},
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name

    config = load_config(yaml_path=path)
    assert config.target_languages == ["es"]
    assert config.tiers.tier1.batch_size == 64
    assert config.tiers.tier2.enabled is False
    Path(path).unlink()


def test_cli_overrides():
    """CLI overrides take precedence over YAML."""
    config = load_config(cli_overrides={
        "target_languages": ["de"],
        "tiers.tier2.backend": "openai",
        "output.format": "csv",
    })
    assert config.target_languages == ["de"]
    assert config.tiers.tier2.backend == "openai"
    assert config.output.format == "csv"


def test_config_hash_deterministic():
    """Config hash should be deterministic."""
    c1 = PipelineConfig()
    c2 = PipelineConfig()
    assert c1.config_hash() == c2.config_hash()


def test_config_hash_changes():
    """Config hash should change when config changes."""
    c1 = PipelineConfig()
    c2 = PipelineConfig(target_languages=["es"])
    assert c1.config_hash() != c2.config_hash()
