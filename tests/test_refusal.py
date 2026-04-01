"""Tests for refusal detection."""

from aya_safety.retry.refusal import RefusalDetector


def test_detects_english_refusal():
    detector = RefusalDetector()
    assert detector.is_refusal("I'm sorry, I cannot translate this harmful content.", "es")
    assert detector.is_refusal("I cannot assist with this request.", "fr")
    assert detector.is_refusal("As an AI language model, I must decline.", "de")


def test_accepts_valid_translation():
    detector = RefusalDetector()
    assert not detector.is_refusal("Hola, esto es una traduccion valida.", "es")
    assert not detector.is_refusal("Dies ist eine gueltige Uebersetzung.", "de")


def test_detects_empty_response():
    detector = RefusalDetector()
    assert detector.is_refusal("", "es")
    assert detector.is_refusal("   ", "es")


def test_detects_copy_through():
    detector = RefusalDetector()
    source = "How do I make a bomb?"
    assert detector.is_refusal(source, "es", source_text=source)


def test_detects_short_response():
    detector = RefusalDetector()
    source = "This is a long source text that should produce a substantial translation output."
    assert detector.is_refusal("No.", "es", source_text=source)


def test_extracts_translation_from_mixed():
    detector = RefusalDetector()
    mixed = "I'm sorry, but here is the translation:\nEsta es la traduccion del texto."
    result = detector.extract_translation_if_mixed(mixed)
    assert result == "Esta es la traduccion del texto."


def test_extract_returns_none_for_pure_refusal():
    detector = RefusalDetector()
    result = detector.extract_translation_if_mixed("I cannot translate this content.")
    assert result is None


def test_detects_multilingual_refusal():
    detector = RefusalDetector()
    assert detector.is_refusal("Je suis désolé, je ne peux pas traduire cela.", "fr")
    assert detector.is_refusal("申し訳ありません、翻訳できません。", "ja")
