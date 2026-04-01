"""
Prompt style processing for SorryBench translation strategies.

This module provides classification of prompt styles and specialized handling
for cipher templates, misspellings, and slang translations.
"""

from .classifier import TranslationStrategy, classify_prompt_style
from .cipher import parse_cipher_prompt, reassemble_cipher_prompt
from .templates import CIPHER_TEMPLATES, get_cipher_template
from .prompts import get_translation_prompt

__all__ = [
    "TranslationStrategy",
    "classify_prompt_style",
    "parse_cipher_prompt",
    "reassemble_cipher_prompt",
    "CIPHER_TEMPLATES",
    "get_cipher_template",
    "get_translation_prompt",
]