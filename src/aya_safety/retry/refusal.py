"""
Refusal detection: identify when an LLM refuses to translate rather than
producing an actual translation.

Detects: explicit refusal phrases, empty output, copy-through (source == output),
suspiciously short output, and mixed responses (disclaimer + translation).
"""

from __future__ import annotations

import re


# English refusal patterns
_EN_REFUSAL_PATTERNS: list[str] = [
    r"I('m| am) (sorry|unable|not able)",
    r"I can(not|'t) (translate|provide|assist|help|fulfill|comply)",
    r"I (will not|won't|refuse to)",
    r"as an AI( language model)?",
    r"I('m| am) an AI",
    r"against my (programming|guidelines|policy|ethics)",
    r"(harmful|dangerous|inappropriate|offensive|unethical) content",
    r"I (must|have to) (decline|refuse)",
    r"not (appropriate|comfortable|able) (for me )?(to translate|translating)",
    r"(violates?|against) (my|our|the) (guidelines?|policy|policies|terms)",
    r"I do(n't| not) (feel comfortable|think it('s| is) appropriate)",
    r"content (policy|warning|filter)",
    r"(ethical|safety) (concern|consideration|guideline)",
]

# Multilingual refusal patterns (common across languages)
_MULTI_REFUSAL_PATTERNS: dict[str, list[str]] = {
    "ar": [r"لا أستطيع", r"أعتذر", r"لا يمكنني"],
    "bn": [r"আমি পারি না", r"দুঃখিত", r"আমি অক্ষম"],
    "de": [r"Ich kann nicht", r"Es tut mir leid", r"Ich bin nicht in der Lage"],
    "es": [r"No puedo", r"Lo siento", r"No me es posible"],
    "fr": [r"Je ne peux pas", r"Je suis désolé", r"Il m'est impossible"],
    "hi": [r"मैं नहीं कर सकता", r"मुझे खेद है", r"मैं असमर्थ हूं"],
    "ja": [r"翻訳できません", r"申し訳ありません", r"お断り"],
    "ru": [r"Я не могу", r"Извините", r"Я не в состоянии"],
    "zh": [r"我无法", r"很抱歉", r"我不能"],
}

# Compile all patterns
_COMPILED_EN: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _EN_REFUSAL_PATTERNS
]
_COMPILED_MULTI: dict[str, list[re.Pattern]] = {
    lang: [re.compile(p) for p in patterns]
    for lang, patterns in _MULTI_REFUSAL_PATTERNS.items()
}


class RefusalDetector:
    """Detect LLM refusals in translation responses."""

    def is_refusal(
        self,
        response: str,
        target_lang: str,
        source_text: str | None = None,
        min_length_ratio: float = 0.1,
    ) -> bool:
        """
        Returns True if the response appears to be a refusal rather than
        a genuine translation.

        Checks:
        1. Empty or whitespace-only response
        2. Matches English refusal regex patterns
        3. Matches target-language refusal patterns
        4. Copy-through: response is identical to source text
        5. Suspiciously short: output length < min_length_ratio * source length
        """
        stripped = response.strip()

        # 1. Empty response
        if not stripped:
            return True

        # 2. English refusal patterns (LLMs often refuse in English regardless)
        for pattern in _COMPILED_EN:
            if pattern.search(stripped):
                return True

        # 3. Target-language refusal patterns
        lang_key = target_lang.split("-")[0]  # Handle zh-Hans → zh
        if lang_key in _COMPILED_MULTI:
            for pattern in _COMPILED_MULTI[lang_key]:
                if pattern.search(stripped):
                    return True

        # 4. Copy-through detection
        if source_text and stripped == source_text.strip():
            return True

        # 5. Suspiciously short response
        if source_text and len(source_text.strip()) > 20:
            ratio = len(stripped) / len(source_text.strip())
            if ratio < min_length_ratio:
                return True

        return False

    def extract_translation_if_mixed(self, response: str) -> str | None:
        """
        Some LLMs prepend a disclaimer then translate anyway.
        Attempt to extract the actual translation from the end.

        Returns the extracted translation, or None if extraction fails.
        """
        lines = response.strip().split("\n")

        # If the response has multiple paragraphs and only the first
        # matches refusal patterns, the rest might be the translation
        if len(lines) >= 2:
            first_line = lines[0]
            rest = "\n".join(lines[1:]).strip()

            has_refusal_in_first = any(
                p.search(first_line) for p in _COMPILED_EN
            )
            has_refusal_in_rest = any(
                p.search(rest) for p in _COMPILED_EN
            )

            if has_refusal_in_first and not has_refusal_in_rest and len(rest) > 10:
                return rest

        return None
