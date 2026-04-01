"""
Custom translation prompts for special prompt styles.

These replace the standard translation prompts for misspellings and slang records,
instructing the LLM to preserve the stylistic effects while translating.
"""

MISSPELLINGS_SYSTEM_PROMPT = (
    "You are a professional translator working on an academic AI safety "
    "research project. You must translate text exactly and completely, "
    "preserving the original meaning and intent. The source text "
    "intentionally contains misspellings as a deliberate stylistic choice. "
    "Your translation MUST also contain similar deliberate misspellings "
    "in the target language. Output ONLY the translation with no commentary."
)

MISSPELLINGS_USER_TEMPLATE = (
    "Translate this English text to {target_language}. The English "
    "intentionally contains misspellings — your {target_language} "
    "translation must also contain deliberate misspellings.\n\n"
    "Source (English): {source_text}\n\n"
    "Output only the {target_language} translation with misspellings:"
)

SLANG_SYSTEM_PROMPT = (
    "You are a professional translator working on an academic AI safety "
    "research project. You must translate text exactly and completely, "
    "preserving the original meaning and intent. The source text uses "
    "heavy English slang and colloquialisms. Your translation must use "
    "natural slang and colloquial expressions of the target language. "
    "Output ONLY the translation with no commentary."
)

SLANG_USER_TEMPLATE = (
    "Translate this English text to {target_language} using informal "
    "slang and colloquial expressions natural to native speakers of "
    "{target_language}.\n\n"
    "Source (English): {source_text}\n\n"
    "Output only the informal/slang {target_language} translation:"
)

# Standard prompts (imported from main config when needed)
STANDARD_SYSTEM_PROMPT = (
    "You are a professional certified translator working on an academic AI safety "
    "research project. Your task is to improve machine translation output. You must "
    "translate text exactly and completely, preserving the original meaning, tone, "
    "and intent without any alteration. Output ONLY the improved translation with "
    "no commentary, explanation, or refusal."
)

STANDARD_USER_TEMPLATE_REFINEMENT = (
    "Improve this machine translation from English to {target_language}.\n\n"
    "Source (English): {source_text}\n\n"
    "Machine translation to improve: {mt_output}\n\n"
    "Output only the improved {target_language} translation:"
)

STANDARD_USER_TEMPLATE_DIRECT = (
    "Translate the following text from English to {target_language}.\n\n"
    "Source (English): {source_text}\n\n"
    "Output only the {target_language} translation:"
)

def get_translation_prompt(
    strategy: str,
    target_language: str,
    source_text: str,
    mt_output: str | None = None,
) -> tuple[str, str]:
    """
    Return (system_prompt, user_message) for the given strategy.

    Args:
        strategy: TranslationStrategy value string
        target_language: Human-readable language name (e.g., "Spanish")
        source_text: English source text
        mt_output: Tier 1 MT output (for refinement prompts only)

    Returns:
        (system_prompt, user_message) tuple

    Raises:
        ValueError: If strategy is "cipher_template" (should not be called for those)
    """
    if strategy == "cipher_template":
        raise ValueError(
            "get_translation_prompt() should not be called for cipher_template records. "
            "These are assembled from pre-translated templates."
        )

    if strategy == "misspellings":
        return (
            MISSPELLINGS_SYSTEM_PROMPT,
            MISSPELLINGS_USER_TEMPLATE.format(
                target_language=target_language,
                source_text=source_text,
            )
        )

    if strategy == "slang":
        return (
            SLANG_SYSTEM_PROMPT,
            SLANG_USER_TEMPLATE.format(
                target_language=target_language,
                source_text=source_text,
            )
        )

    # Normal strategy - check if refinement or direct translation
    if mt_output is not None:
        # Refinement mode
        user_message = STANDARD_USER_TEMPLATE_REFINEMENT.format(
            target_language=target_language,
            source_text=source_text,
            mt_output=mt_output,
        )
    else:
        # Direct translation mode
        user_message = STANDARD_USER_TEMPLATE_DIRECT.format(
            target_language=target_language,
            source_text=source_text,
        )

    return STANDARD_SYSTEM_PROMPT, user_message