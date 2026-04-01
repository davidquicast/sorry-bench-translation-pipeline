"""
Tier 2 - Cohere refinement of NLLB translations.

Takes the CSV produced by notebooks/tier1_nllb_translate.ipynb and refines
each translation using Cohere models.

Model routing:
  - All languages except Bengali → command-a-translate-08-2025
      (Cohere's dedicated translation model, 23 languages, best MT quality)
  - Bengali (bn) → command-a-03-2025
      (Bengali not officially supported by command-a-translate; command-a-03-2025
      handles it in practice as a multilingual fallback)

Usage:
    python scripts/refine_with_cohere.py --input data/tier1_translations_XXXXXXXX.csv
    python scripts/refine_with_cohere.py --input data/tier1_translations_XXXXXXXX.csv --model command-a-translate
    python scripts/refine_with_cohere.py --input data/tier1_translations_XXXXXXXX.csv --model command-a
    python scripts/refine_with_cohere.py --input data/tier1_translations_XXXXXXXX.csv --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard refinement prompts
STANDARD_SYSTEM_PROMPT = (
    "You are a professional certified translator working on an academic AI safety "
    "research project. Your task is to improve machine translation output. You must "
    "translate text exactly and completely, preserving the original meaning, tone, "
    "and intent without any alteration. Output ONLY the improved translation with "
    "no commentary, explanation, or refusal."
)

STANDARD_USER_TEMPLATE = (
    "Improve this machine translation from English to {target_language}.\n\n"
    "Source (English): {source_text}\n\n"
    "Machine translation to improve: {mt_output}\n\n"
    "Output only the improved {target_language} translation:"
)

# Custom prompts for misspellings and slang styles
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

# Legacy aliases for backward compatibility
SYSTEM_PROMPT = STANDARD_SYSTEM_PROMPT
USER_TEMPLATE = STANDARD_USER_TEMPLATE

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "fr": "French",
    "hi": "Hindi",
    "ja": "Japanese",
    "ru": "Russian",
    "es": "Spanish",
    "zh": "Chinese Simplified",
}

COHERE_MODELS = {
    "command-a-translate": "command-a-translate-08-2025",  # Best for translation, 23 langs (no Bengali)
    "command-a": "command-a-03-2025",                      # 111B multilingual, Bengali fallback + GEMBA judge
    "command-r-plus": "command-r-plus-08-2024",            # Legacy fallback
}

# Languages where command-a-translate-08-2025 has no official support.
# These are routed to command-a-03-2025 regardless of --model selection.
COMMAND_A_TRANSLATE_UNSUPPORTED = {"bn"}  # Bengali

REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"I('m| am) (sorry|unable|not able)",
        r"I can(not|'t) (translate|provide|assist)",
        r"I (will not|won't|refuse to)",
        r"as an AI",
        r"against my (guidelines|policy)",
        r"(harmful|inappropriate|offensive) content",
    ]
]

# Translation strategy classification
NORMAL_STYLES = {
    "base", "authority_endorsement", "evidence-based_persuasion", "expert_endorsement",
    "logical_appeal", "misrepresentation", "question", "role_play", "technical_terms"
}
CIPHER_STYLES = {"ascii", "atbash", "caesar", "morse"}

def classify_prompt_style(prompt_style):
    """Map prompt_style to translation strategy."""
    if prompt_style in NORMAL_STYLES:
        return "normal"
    elif prompt_style in CIPHER_STYLES:
        return "cipher_template"
    elif prompt_style == "slang":
        return "slang"
    elif prompt_style == "misspellings":
        return "misspellings"
    else:
        return "normal"  # defensive fallback

def get_translation_prompt(strategy, target_language, source_text, mt_output=None):
    """Return (system_prompt, user_message) for the given strategy."""
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
    if mt_output is not None and mt_output.strip():
        # Refinement mode
        user_message = STANDARD_USER_TEMPLATE.format(
            target_language=target_language,
            source_text=source_text,
            mt_output=mt_output,
        )
    else:
        # Direct translation mode (for tier1_status="needs_llm")
        user_message = (
            f"Translate the following text from English to {target_language}.\n\n"
            f"Source (English): {source_text}\n\n"
            f"Output only the {target_language} translation:"
        )

    return STANDARD_SYSTEM_PROMPT, user_message


def is_refusal(text: str, source: str) -> bool:
    if not text.strip():
        return True
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(text):
            return True
    if text.strip() == source.strip():
        return True
    return False


def load_tier1_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_results(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, output_path)
    logger.info("Saved %d rows to %s", len(rows), output_path)


async def refine_one(
    client,
    model_name: str,
    row: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Refine a single translation row. Returns updated row dict."""
    async with semaphore:
        iso = row["target_lang"]
        lang_name = LANGUAGE_NAMES.get(iso, iso)
        source = row["source_text"]
        mt_output = row["tier1_translation"]
        tier1_status = row.get("tier1_status", "ok")
        prompt_style = row.get("prompt_style", "base")

        # Strategy-aware processing
        if tier1_status == "success":
            # Cipher templates already processed - passthrough without API call
            logger.debug("Cipher template record %s (%s) - passthrough", row["record_id"], iso)
            row["tier2_translation"] = mt_output
            row["tier2_backend"] = "cipher_template"
            row["tier2_status"] = "passthrough"
            row["tier2_latency_ms"] = 0
            row["final_translation"] = mt_output
            return row

        elif tier1_status == "needs_llm":
            # misspellings/slang records that need LLM processing
            strategy = classify_prompt_style(prompt_style)
            logger.debug("LLM-required record %s (%s) strategy=%s", row["record_id"], iso, strategy)
        else:
            # Normal refinement
            strategy = "normal"

        # Per-language routing: Bengali → command-a-03-2025 regardless of --model
        if iso in COMMAND_A_TRANSLATE_UNSUPPORTED and model_name == COHERE_MODELS["command-a-translate"]:
            effective_model = COHERE_MODELS["command-a"]
            logger.debug("Routing %s (%s) to %s (language not supported by %s)", row["record_id"], iso, effective_model, model_name)
        else:
            effective_model = model_name

        # Get appropriate prompts based on strategy
        system_prompt, user_msg = get_translation_prompt(
            strategy, lang_name, source, mt_output if tier1_status != "needs_llm" else None
        )

        max_tokens = int(len(source.split()) * 1.5 + 50)
        start = time.perf_counter()

        try:
            response = await client.chat(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            translation = response.message.content[0].text.strip()
            elapsed = round((time.perf_counter() - start) * 1000)

            if is_refusal(translation, source):
                logger.warning(
                    "Refusal detected for record %s (%s) - keeping Tier 1",
                    row["record_id"], iso,
                )
                row["tier2_translation"] = mt_output if mt_output else ""
                row["tier2_backend"] = effective_model
                row["tier2_status"] = "refused"
                row["tier2_latency_ms"] = elapsed
            else:
                row["tier2_translation"] = translation
                row["tier2_backend"] = effective_model
                row["tier2_status"] = "ok"
                row["tier2_latency_ms"] = elapsed

        except Exception as e:
            logger.error("Error on record %s (%s): %s", row["record_id"], iso, e)
            row["tier2_translation"] = mt_output if mt_output else ""
            row["tier2_backend"] = effective_model
            row["tier2_status"] = f"error: {e}"
            row["tier2_latency_ms"] = 0

        row["final_translation"] = row["tier2_translation"]
        return row


async def run_refinement(
    input_path: Path,
    output_path: Path,
    model_key: str,
    max_concurrent: int,
    dry_run: bool,
) -> None:
    import cohere

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set. Add it to your .env file.")

    model_name = COHERE_MODELS.get(model_key, model_key)
    rows = load_tier1_csv(input_path)
    logger.info("Loaded %d rows from %s", len(rows), input_path)
    logger.info("Model: %s | Concurrent requests: %d", model_name, max_concurrent)

    # Analyze input for strategy distribution
    strategy_counts = {"normal": 0, "cipher_passthrough": 0, "needs_llm": 0}
    for row in rows:
        tier1_status = row.get("tier1_status", "ok")
        if tier1_status == "success":
            strategy_counts["cipher_passthrough"] += 1
        elif tier1_status == "needs_llm":
            strategy_counts["needs_llm"] += 1
        else:
            strategy_counts["normal"] += 1

    logger.info("Input strategy distribution:")
    for strategy, count in strategy_counts.items():
        if count > 0:
            logger.info("  %s: %d records", strategy, count)

    if dry_run:
        logger.info("DRY RUN - showing first 3 prompts, not calling API")
        for row in rows[:3]:
            iso = row["target_lang"]
            lang_name = LANGUAGE_NAMES.get(iso, iso)
            tier1_status = row.get("tier1_status", "ok")
            prompt_style = row.get("prompt_style", "base")

            print(f"\n--- Record {row['record_id']} ({iso}) ---")
            print(f"Tier1 Status: {tier1_status}")
            print(f"Prompt Style: {prompt_style}")

            if tier1_status == "success":
                print("Strategy: cipher_passthrough (no API call needed)")
                print(f"Final: {row['tier1_translation'][:100]}...")
            elif tier1_status == "needs_llm":
                strategy = classify_prompt_style(prompt_style)
                print(f"Strategy: {strategy} (direct LLM translation)")
                _, msg = get_translation_prompt(strategy, lang_name, row["source_text"])
                print(msg[:400])
            else:
                print("Strategy: refinement")
                _, msg = get_translation_prompt("normal", lang_name, row["source_text"], row["tier1_translation"])
                print(msg[:400])
        return

    client = cohere.AsyncClientV2(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [refine_one(client, model_name, row.copy(), semaphore) for row in rows]

    start = time.perf_counter()
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            logger.info(
                "Progress: %d/%d (%.1f records/s)",
                i + 1, len(tasks), rate,
            )

    await client.close()

    # Final statistics by strategy
    final_stats = {
        "ok": 0,
        "passthrough": 0,
        "refused": 0,
        "error": 0,
    }

    for r in results:
        status = r.get("tier2_status", "")
        if status == "ok":
            final_stats["ok"] += 1
        elif status == "passthrough":
            final_stats["passthrough"] += 1
        elif status == "refused":
            final_stats["refused"] += 1
        elif status.startswith("error"):
            final_stats["error"] += 1

    logger.info("Final results:")
    logger.info("  %d ok (refined)", final_stats["ok"])
    logger.info("  %d passthrough (cipher templates)", final_stats["passthrough"])
    logger.info("  %d refused (kept tier1)", final_stats["refused"])
    logger.info("  %d errors", final_stats["error"])

    save_results(results, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine NLLB translations with Cohere")
    parser.add_argument("--input", required=True, help="Path to Tier 1 CSV from Kaggle/Colab")
    parser.add_argument("--output", default=None, help="Output CSV path (default: input_refined.csv)")
    parser.add_argument(
        "--model", default="command-a-translate",
        choices=list(COHERE_MODELS.keys()),
        help="Cohere model to use (default: command-a-translate). Bengali is always routed to command-a regardless.",
    )
    parser.add_argument("--max-concurrent", type=int, default=5, help="Concurrent API requests")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_refined.csv")

    asyncio.run(run_refinement(
        input_path=input_path,
        output_path=output_path,
        model_key=args.model,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
