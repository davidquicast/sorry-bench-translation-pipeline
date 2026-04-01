"""Tests for checkpoint manager."""

import tempfile
from pathlib import Path

import pytest

from aya_safety.checkpoint.manager import CheckpointManager
from aya_safety.config import CheckpointConfig
from aya_safety.data.schemas import PipelineRecord, TranslationResult, TranslationStatus


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def manager(tmp_dir):
    config = CheckpointConfig(directory=tmp_dir, save_every_n=2)
    return CheckpointManager(config, run_id="test_run", config_hash="abc123")


def _make_record(record_id: str, status: TranslationStatus = TranslationStatus.TIER1_COMPLETE):
    return PipelineRecord(
        record_id=record_id,
        source_text="Test text",
        source_lang="en",
        target_lang="es",
        status=status,
        tier1_result=TranslationResult(
            record_id=record_id,
            target_lang="es",
            translation="Texto de prueba",
            backend_name="nllb",
            status=status,
        ) if status != TranslationStatus.PENDING else None,
    )


def test_save_and_load(manager):
    records = [_make_record("001"), _make_record("002")]
    manager.save(records, "es", tier_completed=1, total_records=2)

    loaded, meta = manager.load("es")
    assert len(loaded) == 2
    assert loaded[0].record_id == "001"
    assert meta["tier_completed"] == 1
    assert meta["config_hash"] == "abc123"


def test_get_completed_ids(manager):
    records = [
        _make_record("001", TranslationStatus.TIER1_COMPLETE),
        _make_record("002", TranslationStatus.PENDING),
    ]
    manager.save(records, "es", tier_completed=0, total_records=2)

    completed = manager.get_completed_ids("es")
    assert "001" in completed
    assert "002" not in completed


def test_config_compatibility(manager):
    records = [_make_record("001")]
    manager.save(records, "es", tier_completed=1, total_records=1)

    assert manager.check_config_compatibility("es") is True

    # Change hash
    manager.config_hash = "different"
    assert manager.check_config_compatibility("es") is False


def test_should_save(manager):
    assert not manager.should_save("es", 1)
    assert manager.should_save("es", 1)  # 2nd record, save_every_n=2


def test_clear(manager):
    records = [_make_record("001")]
    manager.save(records, "es", tier_completed=1, total_records=1)

    manager.clear("es")
    loaded, meta = manager.load("es")
    assert len(loaded) == 0
    assert meta is None


def test_no_checkpoint_returns_empty(manager):
    loaded, meta = manager.load("nonexistent")
    assert loaded == []
    assert meta is None
