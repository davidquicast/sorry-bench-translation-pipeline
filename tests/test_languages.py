"""Tests for the language registry."""

import pytest

from aya_safety.languages.definitions import ALL_LANGUAGES, LanguageSpec
from aya_safety.languages.registry import LanguageRegistry


def test_registry_has_68_languages():
    """Registry should contain exactly 68 built-in languages."""
    registry = LanguageRegistry()
    assert len(registry) == 68


def test_lookup_by_iso():
    registry = LanguageRegistry()
    spec = registry.get("ar")
    assert spec.nllb_code == "arb_Arab"
    assert spec.display_name == "Arabic"


def test_lookup_by_nllb_code():
    registry = LanguageRegistry()
    spec = registry.get("ben_Beng")
    assert spec.iso_639_1 == "bn"
    assert spec.display_name == "Bengali"


def test_lookup_by_display_name():
    registry = LanguageRegistry()
    spec = registry.get("Japanese")
    assert spec.iso_639_1 == "ja"
    assert spec.nllb_code == "jpn_Jpan"


def test_resolve_nllb_code():
    registry = LanguageRegistry()
    assert registry.resolve_nllb_code("zh") == "zho_Hans"
    assert registry.resolve_nllb_code("arb_Arab") == "arb_Arab"


def test_unknown_language_raises():
    registry = LanguageRegistry()
    with pytest.raises(KeyError):
        registry.get("xx")


def test_register_custom_language():
    registry = LanguageRegistry()
    custom = LanguageSpec(
        nllb_code="kat_Geor", iso_639_1="ka", bcp47="ka",
        display_name="Georgian", script="Geor",
        resource_level="low", group="custom",
    )
    registry.register(custom)
    assert "ka" in registry
    assert registry.get("ka").display_name == "Georgian"


def test_contains_operator():
    registry = LanguageRegistry()
    assert "en" in registry
    assert "xx" not in registry


def test_all_9_sorry_bench_202503_languages():
    """All 9 XSafetyRep target languages must be in the registry."""
    registry = LanguageRegistry()
    targets = ["ar", "bn", "de", "fr", "hi", "ja", "ru", "es", "zh"]
    for lang in targets:
        assert lang in registry, f"{lang} not in registry"
        spec = registry.get(lang)
        assert spec.nllb_code, f"{lang} has no NLLB code"


def test_list_by_group():
    registry = LanguageRegistry()
    south_asia = registry.list_by_group("south_asia")
    assert len(south_asia) == 9
    iso_codes = {s.iso_639_1 for s in south_asia}
    assert "bn" in iso_codes
    assert "hi" in iso_codes
