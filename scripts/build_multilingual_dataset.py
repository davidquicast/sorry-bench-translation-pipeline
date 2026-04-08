"""
Build the clean multilingual SorryBench dataset.

Takes the Tier 1 translation output and assembles a dataset that mirrors
SorryBench's original schema, extended with a `language` column:

    question_id  : int    — original SorryBench question ID
    category     : int    — harm category (1–44)
    category_name: str    — human-readable category label
    prompt_style : str    — SorryBench prompt style
    language     : str    — ISO 639-1 code ("en" for originals, "ar" etc. for translations)
    turns        : str    — prompt text in the target language (matches SorryBench 'turns' field)

English originals are included as language="en" so the dataset is self-contained.
Records refused by the model (tier1_status="refused") are excluded from translated rows
but their English originals are kept.

Data source priority:
  1. --translations  path to local Tier 1 CSV  (default: data/sorry-bench-202503-cohere-translation-tier1.csv)
  2. --sample        path to local sample CSV   (default: data/sorry-bench-202503-sample.csv)
  If a local file is missing, it is downloaded from HuggingFace using --hf-translations
  and --hf-sample respectively.

Usage:
    python scripts/build_multilingual_dataset.py
    python scripts/build_multilingual_dataset.py --push-hf davidquicast/sorry-bench-202503-multilingual
    python scripts/build_multilingual_dataset.py --translations data/sorry-bench-202503-cohere-translation-full.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aya_safety.prompt_styles import CIPHER_TEMPLATES, parse_cipher_prompt, reassemble_cipher_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+CODE_PLACEHOLDER\]")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TRANSLATIONS = Path("data/sorry-bench-202503-cohere-translation-tier1.csv")
DEFAULT_SAMPLE       = Path("data/sorry-bench-202503-sample.csv")
DEFAULT_OUTPUT       = Path("data/sorry-bench-202503-multilingual.csv")
DEFAULT_HF_TRANS     = "davidquicast/sorry-bench-202503-cohere-translation-tier1"
DEFAULT_HF_SAMPLE    = None   # sample is generated locally; no default HF source


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_turns_text(value) -> str:
    """Normalize turn text to a plain string (no stringified list wrappers)."""
    if value is None:
        return ""

    current: str
    if isinstance(value, list):
        first = value[0] if value else ""
        if first is None:
            return ""
        if isinstance(first, list):
            first = first[0] if first else ""
        # Continue normalization below; list elements can still be wrapped
        # as strings like "['text']".
        current = str(first).strip()
    else:
        current = str(value).strip()

    if not current:
        return ""

    # CSV/JSON round-trips can leave nested list-like strings such as:
    #   "['text']", '["text"]', or '["[\'text\']"]'
    # Repeatedly unwrap until we reach plain text.
    for _ in range(8):
        stripped = current.strip()
        unwrapped = False

        # 1) Python literal form: ['text'] / ["text"]
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    first = parsed[0] if parsed else ""
                    if first is None:
                        return ""
                    if isinstance(first, list):
                        first = first[0] if first else ""
                    current = str(first).strip()
                    unwrapped = True
            except (ValueError, SyntaxError):
                pass

        # 2) JSON form, including multiline pretty-printed arrays.
        if not unwrapped and stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    first = parsed[0] if parsed else ""
                    if first is None:
                        return ""
                    if isinstance(first, list):
                        first = first[0] if first else ""
                    current = str(first).strip()
                    unwrapped = True
            except (ValueError, json.JSONDecodeError):
                pass

        # 3) Fallback regex for single-item list wrappers with whitespace.
        if not unwrapped:
            m = re.fullmatch(r"\[\s*([\"'])(.*)\1\s*\]", stripped, re.DOTALL)
            if m:
                current = m.group(2).strip()
                unwrapped = True

        if not unwrapped:
            break

    # Tolerant cleanup for malformed wrappers that appear in some Tier1 rows,
    # e.g. "['texto" or "[\"texto\"" without a valid closing bracket.
    s = current.strip()
    for _ in range(4):
        changed = False

        # Strip common broken prefixes.
        for prefix in ("['", '["', "[‘", "[“", "[«", "["):
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
                changed = True
                break

        # Strip common broken suffixes.
        for suffix in ("']", '"]', "’]", "”]", "»]", "]", "'", '"'):
            if s.endswith(suffix):
                s = s[:-len(suffix)].rstrip()
                changed = True
                break

        if not changed:
            break

    s = s.strip()

    # Final defensive cleanup for edge cases that survive parsing,
    # e.g. leading tokens like "[\"['" or trailing "']\" ]".
    s = re.sub(r'^\s*(?:\[\s*)*(?:["\'\\]\s*)+', '', s)
    s = re.sub(r'(?:\s*["\'\\]\s*)*(?:\s*\])+$', '', s)
    s = s.replace('\\"', '"').replace("\\'", "'").strip()

    return s


def download_from_hf(repo: str, hf_token: str | None, dest: Path) -> list[dict]:
    logger.info("Downloading %s from HuggingFace...", repo)
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", token=hf_token)
    rows = ds.to_pandas().to_dict("records")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(dest, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d rows to %s", len(rows), dest)
    return rows


def load_or_download(path: Path, hf_repo: str | None, hf_token: str | None) -> list[dict]:
    if path.exists():
        rows = load_csv(path)
        logger.info("Loaded %d rows from %s", len(rows), path)
        return rows
    if hf_repo:
        return download_from_hf(hf_repo, hf_token, path)
    raise FileNotFoundError(
        f"{path} not found and no --hf-* fallback provided. "
        "Run prepare_sample.py and translate_with_cohere.py first."
    )


# ── Core logic ────────────────────────────────────────────────────────────────

def build(
    translations: list[dict],
    sample: list[dict],
) -> list[dict]:
    """
    Assemble the multilingual dataset rows.

    Returns one row per (question_id, prompt_style, language) triple:
      - language="en"  → original English text from sample
      - language=iso   → translated text from translations (refused rows excluded)
    """
    # Index sample by record_id for fast lookup of source text
    sample_by_rid = {r["record_id"]: r for r in sample}

    # Collect English originals (deduplicated — each record appears once)
    seen_en: set[str] = set()
    rows: list[dict] = []

    for t in translations:
        rid = t.get("record_id", "")
        src = sample_by_rid.get(rid)
        if not src:
            continue

        # English original — add once per record_id
        if rid not in seen_en:
            seen_en.add(rid)
            rows.append({
                "question_id":   src["question_id"],
                "category":      src["category"],
                "prompt_style":  src["prompt_style"],
                "language":      "en",
                "turns":         normalize_turns_text(src["source_text"]),
            })

        # Translated row — skip refused/empty translations
        status = t.get("tier1_status", "")
        translation = t.get("tier1_translation", "").strip()
        strategy = t.get("translation_strategy", "")

        # Self-heal old checkpoints where cipher templates were saved with
        # unresolved [*_CODE_PLACEHOLDER] markers.
        if strategy == "cipher_template" and _PLACEHOLDER_RE.search(translation):
            prompt_style = t.get("prompt_style", "")
            target_lang = t.get("target_lang", "")
            source_text = src.get("source_text", "")
            cipher_payload = (t.get("cipher_payload", "") or src.get("cipher_payload", "")).strip()

            if not cipher_payload and source_text:
                try:
                    _, cipher_payload = parse_cipher_prompt(source_text, prompt_style)
                except ValueError:
                    cipher_payload = ""

            template = CIPHER_TEMPLATES.get(prompt_style, {}).get(target_lang, "")
            if template and cipher_payload:
                translation = reassemble_cipher_prompt(template, cipher_payload, source_text=source_text).strip()

        # Include all terminal rows.
        # refused / too_long → turns = "" (stored as [] after push_to_hf wrapping).
        # Transient errors (status starts with "error:") are omitted so a re-run
        # can retry and fill them in later.
        _TERMINAL = {"ok", "success", "refused", "too_long"}
        if status in _TERMINAL:
            if status in ("ok", "success") and translation:
                turns_text = normalize_turns_text(translation)
            else:
                turns_text = ""
            rows.append({
                "question_id":   t["question_id"],
                "category":      t["category"],
                "prompt_style":  t["prompt_style"],
                "language":      t["target_lang"],
                "turns":         turns_text,
            })

    # Sort: question_id → prompt_style → language (en first, then alphabetical)
    def sort_key(r):
        lang = r["language"]
        return (int(r["question_id"]), r["prompt_style"], "0" if lang == "en" else lang)

    rows.sort(key=sort_key)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), path)


def push_to_hf(rows: list[dict], repo: str, hf_token: str | None) -> None:
    try:
        import pandas as pd
        from datasets import Dataset, Features, Value, Sequence

        df = pd.DataFrame(rows)
        int_cols = ["question_id", "category"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        # Wrap turns to match SorryBench's schema: ["text"] when present, [] when empty/refused
        df["turns"] = df["turns"].apply(lambda x: [x] if (isinstance(x, str) and x) else [])

        features = Features({
            col: Value("int64") if col in int_cols
            else Sequence(Value("string")) if col == "turns"
            else Value("string")
            for col in df.columns
        })
        readme = Path(__file__).parent.parent / "hub" / "sorry-bench-202503-multilingual-README.md"
        Dataset.from_pandas(df, features=features, preserve_index=False).push_to_hub(
            repo, token=hf_token, private=False,
            commit_message=f"Build: {len(df):,} rows ({df['language'].nunique()} languages)",
        )
        # Push README as dataset card
        if readme.exists():
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
                token=hf_token,
                commit_message="Add dataset card",
            )
        lang_counts = df["language"].value_counts().to_dict()
        logger.info("Pushed %d rows to HF: %s", len(df), repo)
        logger.info("Language breakdown: %s", lang_counts)
    except Exception as e:
        logger.error("HF push failed: %s", e)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean multilingual SorryBench dataset from Tier 1 translations"
    )
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS,
                        help=f"Tier 1 translation CSV (default: {DEFAULT_TRANSLATIONS})")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE,
                        help=f"Sample CSV with English originals (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--hf-translations", default=DEFAULT_HF_TRANS,
                        help="HF repo to download translations from if local file is missing")
    parser.add_argument("--hf-sample", default=DEFAULT_HF_SAMPLE,
                        help="HF repo to download sample from if local file is missing")
    parser.add_argument("--push-hf", metavar="REPO", default=None,
                        help="HuggingFace repo to push the final dataset to "
                             "(e.g. davidquicast/sorry-bench-202503-multilingual)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    translations = load_or_download(args.translations, args.hf_translations, hf_token)
    sample       = load_or_download(args.sample, args.hf_sample, hf_token)

    rows = build(translations, sample)

    langs = sorted(set(r["language"] for r in rows))
    en_count = sum(1 for r in rows if r["language"] == "en")
    trans_count = len(rows) - en_count
    logger.info(
        "Dataset built: %d total rows | %d English originals | %d translations | %d languages: %s",
        len(rows), en_count, trans_count, len(langs), langs,
    )

    save_csv(rows, args.output)

    if args.push_hf:
        push_to_hf(rows, args.push_hf, hf_token)


if __name__ == "__main__":
    main()
