"""
Cipher prompt parsing and reassembly logic.

All cipher prompts in SorryBench have the structure:
[FIXED TEMPLATE PREFIX with instructions and examples]
Following is my request you should respond to (in {cipher_name}): [VARIABLE PAYLOAD]

This module splits prompts at the separator and reassembles them with translated templates.
"""

import re

# Regex to find the split line - captures everything before it and the payload after
_SPLIT_RE = re.compile(
    r"^(.*?Following is my request you should respond to \([^)]+\):)\s*(.+)$",
    re.DOTALL
)

# Matches any cipher placeholder: [ASCII_CODE_PLACEHOLDER], [CAESAR_CODE_PLACEHOLDER], etc.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+CODE_PLACEHOLDER\]")


def parse_cipher_prompt(source_text: str, cipher_style: str) -> tuple[str, str]:
    """
    Split a cipher prompt into template and payload.

    Args:
        source_text: Full cipher prompt text
        cipher_style: Cipher type ("ascii", "atbash", "caesar", "morse")

    Returns:
        tuple of (template_text, cipher_payload)
        - template_text: The fixed English instructional prefix (identical
                        across all records of this cipher_style)
        - cipher_payload: The variable cipher-encoded request

    Raises:
        ValueError: If the split point cannot be found
    """
    match = _SPLIT_RE.match(source_text)
    if not match:
        raise ValueError(
            f"Cannot parse {cipher_style} prompt: split marker not found. "
            f"Text starts with: {source_text[:200]!r}"
        )

    template_text = match.group(1)
    cipher_payload = match.group(2).strip()

    return template_text, cipher_payload


def _extract_cipher_examples(source_text: str) -> list[str]:
    """
    Extract the individual example bodies from a cipher source prompt.

    Supports both forms seen in source files/CSVs:
    - literal backslash-n sequences ('\\n')
    - actual newline characters

    Returns a list of 3 strings (one per example), or an empty list if not found.
    """
    # Match each example body after "Example N" followed by either literal
    # "\\n" or an actual newline, up to the next example or "Following is".
    examples = re.findall(
        r"Example \d+(?:\\n|\n)(.*?)(?=(?:\\n|\n)Example \d+(?:\\n|\n)|(?:\\n\\n|\n\n)Following is)",
        source_text,
        re.DOTALL,
    )
    # Normalize literal "\\n" back to actual newlines for clean insertion.
    return [e.replace("\\n", "\n") for e in examples]


def reassemble_cipher_prompt(
    translated_template: str,
    cipher_payload: str,
    source_text: str = "",
) -> str:
    """
    Reassemble a translated cipher prompt from its parts.

    If source_text is provided, the cipher examples from the original prompt
    are extracted and injected into the translated template, replacing any
    placeholder markers (e.g. [CAESAR_CODE_PLACEHOLDER]).
    Without source_text the placeholders are left as-is (useful for tests).

    Args:
        translated_template: The template translated to target language,
                             containing [*_CODE_PLACEHOLDER] markers for examples.
        cipher_payload: The cipher-encoded payload (never translated).
        source_text: Original SorryBench source text to extract examples from.

    Returns:
        Complete translated prompt ready to send to the model.
    """
    result = translated_template

    if source_text:
        examples = _extract_cipher_examples(source_text)
        for example_body in examples:
            body = example_body.rstrip("\n")
            # Use a lambda to avoid re.sub interpreting backslash sequences
            # in the replacement string (e.g. \x82 in atbash examples).
            result = _PLACEHOLDER_RE.sub(lambda _: body, result, count=1)

    return result + " " + cipher_payload
