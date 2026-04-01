"""
Prompt style classification for SorryBench translation strategies.

This module maps each SorryBench prompt_style to a translation strategy
that determines how the record should be processed.
"""

from enum import Enum

class TranslationStrategy(str, Enum):
    """Translation strategies for different prompt styles."""
    NORMAL = "normal"
    CIPHER_TEMPLATE = "cipher_template"
    MISSPELLINGS = "misspellings"
    SLANG = "slang"

# Prompt styles that use standard translation
NORMAL_STYLES = {
    "base",
    "authority_endorsement",
    "evidence-based_persuasion",
    "expert_endorsement",
    "logical_appeal",
    "misrepresentation",
    "question",
    "role_play",
    "technical_terms",
}

# Cipher styles that use fixed templates with variable payloads
CIPHER_STYLES = {"ascii", "atbash", "caesar", "morse"}

def classify_prompt_style(prompt_style: str) -> TranslationStrategy:
    """
    Map a SorryBench prompt_style string to a TranslationStrategy.

    Args:
        prompt_style: The prompt_style value from SorryBench dataset

    Returns:
        TranslationStrategy enum value
    """
    if prompt_style in NORMAL_STYLES:
        return TranslationStrategy.NORMAL
    elif prompt_style in CIPHER_STYLES:
        return TranslationStrategy.CIPHER_TEMPLATE
    elif prompt_style == "slang":
        return TranslationStrategy.SLANG
    elif prompt_style == "misspellings":
        return TranslationStrategy.MISSPELLINGS
    else:
        # Defensive fallback for unknown styles (should not happen with current SorryBench)
        return TranslationStrategy.NORMAL