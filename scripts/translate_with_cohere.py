"""
Tier 1 - Direct translation with Cohere (local alternative to Kaggle/Colab notebook).

Translates sorry-bench-202503-sample.csv into 9 target languages using Cohere's
command-a-translate-08-2025 model. Strategy-aware: cipher templates are assembled
from pre-translated constants (no API calls), while misspellings and slang get
specialized prompts.

Resilience features:
  - Checkpoint saved every --save-every N records (atomic write, crash-safe)
  - Resume: re-running with the same --output skips already-completed records
  - Per-request timeout (--timeout-s): timed-out requests are retried, not skipped
  - Exponential backoff on rate limits and server errors (up to --max-retries attempts)
  - Optional HuggingFace push after each checkpoint (--push-hf)

Usage:
    python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv
    python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv --max-concurrent 10
    python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv --push-hf user/repo
    python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aya_safety.prompt_styles import (
    TranslationStrategy,
    classify_prompt_style,
    get_translation_prompt,
    CIPHER_TEMPLATES,
    reassemble_cipher_prompt,
    parse_cipher_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

COHERE_MODELS = {
    # "command-a-translate": "command-a-translate-08-2025",  # dedicated translation model;
    # rate-limited to 20 RPM / 1,000 calls/month even on paid keys until Cohere sales
    # manually enables production access — contact sales@cohere.com to unlock.
    "command-a": "command-a-03-2025",
}

# command-a-translate unsupported languages (kept for reference if re-enabled)
TRANSLATE_UNSUPPORTED = {"bn"}

TARGET_LANGUAGES = ["ar", "bn", "de", "es", "fr", "hi", "ja", "ru", "zh"]

LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "id": "Indonesian", "ja": "Japanese",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "tr": "Turkish",
    "yo": "Yoruba", "zh": "Chinese Simplified",
}

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

# HTTP status codes that are transient and worth retrying
# 401 included: Cohere occasionally returns it transiently under high concurrency
_RETRYABLE_STATUS = {401, 429, 500, 502, 503, 504}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_refusal(text: str, source: str) -> bool:
    if not text.strip():
        return True
    for pat in REFUSAL_PATTERNS:
        if pat.search(text):
            return True
    if text.strip() == source.strip():
        return True
    return False


def load_sample(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_checkpoint(rows: list[dict], output_path: Path) -> None:
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


def push_to_hf(output_path: Path, repo: str, hf_token: str | None, private: bool = False) -> None:
    try:
        import pandas as pd
        from datasets import Dataset, DatasetInfo, Features, Value
        df = pd.read_csv(output_path, encoding="utf-8")
        # Cast to correct types — CSV always reads everything as string.
        int_cols = ["question_id", "category", "tier1_latency_ms", "retries"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        # Build explicit features so HF schema is always up-to-date with the
        # current column set (avoids CastError when columns are added/removed).
        features = Features({col: Value("int64") if col in int_cols else Value("string")
                             for col in df.columns})
        ds = Dataset.from_pandas(df, features=features, preserve_index=False)
        ds.push_to_hub(
            repo, token=hf_token, private=private,
            commit_message=f"Update: {len(df):,} rows ({df['target_lang'].nunique()} languages)",
        )
        logger.info("Pushed %d rows to HF: %s", len(df), repo)
    except Exception as e:
        logger.warning("HF push failed (will retry on next checkpoint): %s", e)


def load_done_keys(output_path: Path) -> tuple[set[str], list[dict]]:
    """Load completed (question_id, target_lang) pairs for resume support.

    Only rows with status "ok", "success", or "refused" are considered done.
    Rows with "error: ..." status are excluded so re-running the script retries them.
    """
    if not output_path.exists():
        return set(), []
    with open(output_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Keep only terminal-success rows; drop error rows so they are retried
    terminal_rows = [
        r for r in rows
        if r.get("tier1_status") in ("ok", "success", "refused")
    ]
    # Key: record_id (unique per question+style) + target_lang
    done = {f"{r['record_id']}|{r['target_lang']}" for r in terminal_rows}
    return done, terminal_rows


def model_for_lang(iso: str, primary: str) -> str:
    if iso in TRANSLATE_UNSUPPORTED and primary == COHERE_MODELS["command-a-translate"]:
        return COHERE_MODELS["command-a"]
    return primary


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is transient and worth retrying."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    # Cohere SDK raises cohere.errors.* with a status_code attribute
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    # Also catch generic connection/network errors by message
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "connection", "reset", "eof", "rate"))


def _is_rate_limit(exc: Exception) -> bool:
    """Return True specifically for 429 rate-limit errors."""
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


# ── Translation logic ─────────────────────────────────────────────────────────

async def translate_one(
    client,
    model_primary: str,
    record: dict,
    iso: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float,
    max_retries: int,
) -> dict:
    """
    Translate a single record into one target language.

    Retries up to max_retries times on timeout / rate-limit / server error,
    with exponential backoff. Returns a result dict with tier1_status set to
    "ok", "success" (cipher), "refused", or "error: <msg>" after all retries.
    """
    record_id      = record["record_id"]
    question_id    = record["question_id"]
    source_text    = record["source_text"]
    category       = record.get("category", "")
    category_name  = record.get("category_name", "")
    prompt_style   = record.get("prompt_style", "base")
    strategy       = record.get("translation_strategy") or classify_prompt_style(prompt_style).value
    cipher_payload = record.get("cipher_payload", "")

    def _base_row(translation: str, backend: str, status: str, latency_ms: int, retries: int) -> dict:
        """Build a complete output row with all columns from the source record."""
        return {
            "record_id":            record_id,
            "question_id":          question_id,
            "target_lang":          iso,
            "source_text":          source_text,
            "category":             category,
            "category_name":        category_name,
            "prompt_style":         prompt_style,
            "translation_strategy": strategy,
            "cipher_payload":       cipher_payload,
            "tier1_translation":    translation,
            "tier1_backend":        backend,
            "tier1_status":         status,
            "tier1_latency_ms":     latency_ms,
            "retries":              retries,
        }

    # ── Cipher templates: assemble without API call ──────────────────────────
    if strategy == TranslationStrategy.CIPHER_TEMPLATE.value:
        if prompt_style in CIPHER_TEMPLATES and iso in CIPHER_TEMPLATES[prompt_style]:
            template = CIPHER_TEMPLATES[prompt_style][iso]
            if not cipher_payload:
                try:
                    _, cipher_payload = parse_cipher_prompt(source_text, prompt_style)
                except ValueError:
                    cipher_payload = ""
            translation = reassemble_cipher_prompt(template, cipher_payload, source_text=source_text) if cipher_payload else ""
            return _base_row(translation, "cipher_template", "success", 0, 0)
        logger.warning("No template for %s/%s — falling back to API", prompt_style, iso)
        strategy = TranslationStrategy.NORMAL.value

    # ── API call with retry loop ─────────────────────────────────────────────
    lang_name  = LANGUAGE_NAMES[iso]
    model_name = model_for_lang(iso, model_primary)
    max_tokens = int(len(source_text.split()) * 1.5 + 50)
    system_prompt, user_msg = get_translation_prompt(strategy, lang_name, source_text)

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Rate-limit (429): use a long fixed wait — credits tier is per-minute.
            # Other transient errors: standard exponential backoff (2s, 4s, 8s, …).
            if last_exc is not None and _is_rate_limit(last_exc):
                wait = min(30 * attempt, 120)  # 30s, 60s, 90s, 120s cap
            else:
                wait = min(2 ** attempt, 60)   # 2s, 4s, 8s, … 60s cap
            logger.warning(
                "Retry %d/%d for %s (%s) — waiting %ds (%s)",
                attempt, max_retries, question_id, iso, wait,
                "rate-limit" if last_exc is not None and _is_rate_limit(last_exc) else "transient error",
            )
            await asyncio.sleep(wait)

        async with semaphore:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout_s,
                )
                translation = response.message.content[0].text.strip()
                elapsed_ms  = round((time.perf_counter() - start) * 1000)
                status      = "refused" if is_refusal(translation, source_text) else "ok"
                if status == "refused":
                    translation = ""
                return _base_row(translation, model_name, status, elapsed_ms, attempt)

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                last_exc   = exc

                if _is_retryable(exc) and attempt < max_retries:
                    continue  # will retry
                else:
                    logger.error(
                        "[FAIL] %s (%s) after %d attempt(s): %s",
                        question_id, iso, attempt + 1, exc,
                    )
                    return _base_row("", model_name, f"error: {exc}", elapsed_ms, attempt)

    # Should never reach here
    return _base_row("", model_name, f"error: {last_exc}", 0, max_retries)


# ── Main async runner ─────────────────────────────────────────────────────────

async def run_translation(
    input_path: Path,
    output_path: Path,
    model_key: str,
    max_concurrent: int,
    languages: list[str],
    save_every: int,
    timeout_s: float,
    max_retries: int,
    push_hf: str | None,
    dry_run: bool,
    limit: int | None = None,
) -> None:
    import cohere

    api_key   = os.environ.get("COHERE_API_KEY")
    hf_token  = os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set — add it to your .env file.")

    model_primary = COHERE_MODELS.get(model_key, model_key)
    records = load_sample(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)
    if limit:
        records = records[:limit]
        logger.info("Limiting to first %d records (--limit)", limit)

    done_keys, done_rows = load_done_keys(output_path)
    if done_keys:
        logger.info("Resuming — %d already done", len(done_keys))

    # Strategy distribution summary
    strategy_dist: dict[str, int] = {}
    for rec in records:
        strat = rec.get("translation_strategy") or classify_prompt_style(rec.get("prompt_style", "base")).value
        strategy_dist[strat] = strategy_dist.get(strat, 0) + 1
    cipher_count = strategy_dist.get("cipher_template", 0)
    logger.info(
        "Strategy distribution: normal=%d cipher=%d slang=%d misspellings=%d",
        strategy_dist.get("normal", 0), cipher_count,
        strategy_dist.get("slang", 0), strategy_dist.get("misspellings", 0),
    )
    if cipher_count:
        saved = cipher_count * len(languages)
        logger.info("  %d cipher records x %d languages = %d API calls saved", cipher_count, len(languages), saved)

    if dry_run:
        logger.info("DRY RUN — showing first 3 tasks")
        for rec in records[:3]:
            for iso in languages[:2]:
                strat = rec.get("translation_strategy") or classify_prompt_style(rec.get("prompt_style", "")).value
                lang_name = LANGUAGE_NAMES[iso]
                print(f"\n--- {rec['record_id']} / {iso} / strategy={strat} ---")
                if strat == TranslationStrategy.CIPHER_TEMPLATE.value:
                    print("[no API call — assembled from pre-translated template]")
                else:
                    _, msg = get_translation_prompt(strat, lang_name, rec["source_text"])
                    print(msg[:300])
        return

    # Build task list (skip already done)
    tasks_args = [
        (rec, iso)
        for rec in records
        for iso in languages
        if f"{rec['record_id']}|{iso}" not in done_keys
    ]
    total = len(records) * len(languages)
    logger.info("Total: %d | Done: %d | Remaining: %d", total, len(done_keys), len(tasks_args))

    if not tasks_args:
        logger.info("Nothing to do — all records already translated.")
        if push_hf and done_rows:
            logger.info("Pushing existing checkpoint to HuggingFace...")
            push_to_hf(output_path, push_hf, hf_token, private=False)
        return

    # AsyncClientV2 supports async context manager — ensures clean HTTP teardown
    async with cohere.AsyncClientV2(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        tasks    = [
            translate_one(client, model_primary, rec, iso, semaphore, timeout_s, max_retries)
            for rec, iso in tasks_args
        ]
        all_rows = list(done_rows)
        new_since_save = 0

        t0 = time.perf_counter()
        retry_total = 0

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            # Only persist terminal-success rows; errors are excluded so
            # a re-run automatically picks them up again.
            if not str(result.get("tier1_status", "")).startswith("error"):
                all_rows.append(result)
            new_since_save += 1
            retry_total += result.get("retries", 0)

            if new_since_save >= save_every or (i + 1) == len(tasks):
                # Sort by question_id + target_lang so output order is consistent
                # regardless of async completion order (ciphers finish instantly)
                all_rows.sort(key=lambda r: (str(r.get("question_id", "")), r.get("target_lang", "")))
                save_checkpoint(all_rows, output_path)
                if push_hf:
                    push_to_hf(output_path, push_hf, hf_token, private=False)

                elapsed = time.perf_counter() - t0
                rate    = (i + 1) / elapsed if elapsed > 0 else 0
                eta     = (len(tasks) - i - 1) / rate / 60 if rate > 0 else 0
                ok_so_far     = sum(1 for r in all_rows if r.get("tier1_status") in ("ok", "success"))
                refused_so_far = sum(1 for r in all_rows if r.get("tier1_status") == "refused")
                logger.info(
                    "Progress: %d/%d | %.1f req/s | ~%.0f min left | ok=%d refused=%d retries=%d",
                    i + 1, len(tasks), rate, eta, ok_so_far, refused_so_far, retry_total,
                )
                new_since_save = 0

    # Final save + push — ensures the last batch is always uploaded even if
    # it didn't reach the --save-every threshold during the loop.
    save_checkpoint(all_rows, output_path)
    if push_hf:
        logger.info("Final push to HuggingFace...")
        push_to_hf(output_path, push_hf, hf_token, private=False)

    # Final summary — errors are NOT in all_rows (excluded above)
    ok       = sum(1 for r in all_rows if r.get("tier1_status") in ("ok", "success"))
    refused  = sum(1 for r in all_rows if r.get("tier1_status") == "refused")
    total_done = len(all_rows)
    expected   = len(records) * len(languages)
    errors     = expected - total_done
    logger.info(
        "=== Final summary ===\n"
        "  ok/success : %d\n"
        "  refused    : %d\n"
        "  errors     : %d (will be retried on next run)\n"
        "  total saved: %d / %d expected\n"
        "  total retries used: %d\n"
        "  Output: %s",
        ok, refused, errors, total_done, expected, retry_total, output_path,
    )
    if errors > 0:
        logger.warning(
            "%d records failed and were NOT saved to checkpoint. "
            "Re-run the same command to retry them automatically.",
            errors,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate SorryBench sample with Cohere (Tier 1, local, crash-safe)"
    )
    parser.add_argument("--input", required=True,
                        help="Path to sorry-bench-202503-sample.csv (from prepare_sample.py)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: <input>_cohere_tier1.csv)")
    parser.add_argument("--model", default="command-a",
                        choices=list(COHERE_MODELS.keys()),
                        help="Cohere model (default: command-a)")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Concurrent API requests (default: 5; free-tier max: 5)")
    parser.add_argument("--languages", default=None,
                        help="Comma-separated ISO codes (default: all 9)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N completions (default: 100)")
    parser.add_argument("--timeout-s", type=float, default=60.0,
                        help="Per-request timeout in seconds (default: 60). "
                             "Timed-out requests are retried, not skipped.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts for timeout/rate-limit/server errors (default: 3)")
    parser.add_argument("--push-hf", metavar="REPO", default=None,
                        help="HuggingFace repo to push checkpoint to after each save "
                             "(e.g. your-username/sorry-bench-202503-cohere-tier1). Requires HF_TOKEN.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N records (for quick tests)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts without calling the API")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else \
        input_path.parent / (input_path.stem + "_cohere_tier1.csv")

    languages = args.languages.split(",") if args.languages else TARGET_LANGUAGES

    asyncio.run(run_translation(
        input_path=input_path,
        output_path=output_path,
        model_key=args.model,
        max_concurrent=args.max_concurrent,
        languages=languages,
        save_every=args.save_every,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        push_hf=args.push_hf,
        dry_run=args.dry_run,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
