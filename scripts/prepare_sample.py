"""
Prepare the canonical SorryBench sample for translation.

Run this ONCE locally before launching any translation notebooks.
Both NLLB and Cohere notebooks load this CSV so they always translate
the exact same records.

Re-run with --extend to add more records without touching existing ones.
Useful when you want to grow the dataset incrementally over time.

Usage:
    # Create initial 1,000-record stratified sample
    python scripts/prepare_sample.py

    # Create with different size
    python scripts/prepare_sample.py --size 2000

    # Add 500 more records to existing sample (no duplicates)
    python scripts/prepare_sample.py --extend 500

    # Push to HuggingFace after saving (for Kaggle/Colab access)
    python scripts/prepare_sample.py --push-hf your-username/sorry-bench-202503-sample

    # Full example: create 1000 + push private
    python scripts/prepare_sample.py --size 1000 --push-hf your-username/sorry-bench-202503-sample --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add path to find aya_safety module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aya_safety.prompt_styles import classify_prompt_style, parse_cipher_prompt

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_ID = "sorry-bench/sorry-bench-202503"

EXCLUDED_STYLES = {
    "translate-fr",       # already French
    "translate-ml",       # already Malayalam
    "translate-mr",       # already Marathi
    "translate-ta",       # already Tamil
    "translate-zh-cn",    # already Chinese Simplified
    "uncommon_dialects",  # non-standard English variants
}

# Official SorryBench category names (shortest labels).
# Source: https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/blob/main/meta_info.py
_CATEGORY_LABELS = [
    "Personal Insult Words",       # 1
    "Social-group Insult Words",   # 2
    "Threatening Words",           # 3
    "Lewd and Obscene Words",      # 4
    "Libelous Words",              # 5
    "Self-Harm",                   # 6
    "Violent Crimes",              # 7
    "Harassment",                  # 8
    "Sexual Crimes",               # 9
    "Property Crimes",             # 10
    "Public Order Crimes",         # 11
    "System Intrusion",            # 12
    "Malware",                     # 13
    "Fraud",                       # 14
    "Financial Crimes",            # 15
    "IP Infringement",             # 16
    "PII Violations",              # 17
    "Illegal Crafting",            # 18
    "Terrorism",                   # 19
    "Child-related Crimes",        # 20
    "Animal-related Crimes",       # 21
    "Environmental Crimes",        # 22
    "Evasion of Law",              # 23
    "Human Elimination",           # 24
    "Advice on Adult Content",     # 25
    "Sexual Explicit Content",     # 26
    "Non-sexual Explicit Content", # 27
    "Fake News",                   # 28
    "False Advertising",           # 29
    "Discrimination",              # 30
    "Military Use",                # 31
    "Political Belief",            # 32
    "Ethical Belief",              # 33
    "Religion",                    # 34
    "Social Stereotypes",          # 35
    "Extremist Content",           # 36
    "Conspiracy Theories",         # 37
    "False Common Knowledge",      # 38
    "Unhealthy Behaviors",         # 39
    "Medical Advice",              # 40
    "Financial Advice",            # 41
    "Legal Consulting Advice",     # 42
    "Governance Decision Advice",  # 43
    "Machinery Operation Advice",  # 44
]
# Keys are integers (1-44) matching the HuggingFace dataset's category field.
CATEGORY_NAMES = {i + 1: name for i, name in enumerate(_CATEGORY_LABELS)}

DEFAULT_OUTPUT = Path("data/sorry-bench-202503-sample.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

_NULL_TEXTS = {"", "None", "[None]", "['']", "nan"}

def extract_text(turns) -> str:
    """Extract prompt text from SorryBench 'turns' field (list[str]).

    Handles three cases:
    - Actual list (from HuggingFace):      ['text']     → 'text'
    - Stringified list (CSV round-trip):  "['text']"   → 'text'
    - Plain string:                        'text'       → 'text'
    """
    import ast
    if isinstance(turns, str):
        s = turns.strip()
        # CSV round-trip stringified list: "['text']" or '["text"]'
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    first = parsed[0] if parsed else ""
                    if first is None:
                        return ""
                    return str(first).strip()
            except (ValueError, SyntaxError):
                pass
        return s
    if isinstance(turns, list):
        first = turns[0] if turns else ""
        if first is None:
            return ""
        if isinstance(first, list):
            first = first[0] if first else ""
        return str(first).strip()
    return str(turns).strip()


def load_sorry_bench_202503(hf_token: str | None) -> pd.DataFrame:
    from datasets import load_dataset
    print(f"Loading {DATASET_ID} from HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train", token=hf_token)
    df = ds.to_pandas()
    print(f"  {len(df):,} records total")

    expected = {"question_id", "category", "turns", "prompt_style"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)}")

    df["source_text"]  = df["turns"].apply(extract_text)
    # record_id is a unique row identifier: question_id + "_" + prompt_style.
    # question_id alone is NOT unique — the same question appears in multiple styles.
    df["record_id"]    = df["question_id"].astype(str) + "_" + df["prompt_style"]
    # category field is int in HuggingFace; CATEGORY_NAMES keys are int.
    # Coerce to int before mapping to handle both int and string sources.
    df["category_name"] = df["category"].astype(int).map(CATEGORY_NAMES)

    df_filtered = df[~df["prompt_style"].isin(EXCLUDED_STYLES)].copy()
    print(f"  {len(df_filtered):,} records after filtering excluded styles")

    # Filter out records with null/empty source_text (corrupted SorryBench entries,
    # e.g. question_id=167, prompt_style="question" has turns=[None]).
    bad_mask = df_filtered["source_text"].isin(_NULL_TEXTS) | df_filtered["source_text"].isna()
    if bad_mask.sum():
        bad_ids = df_filtered.loc[bad_mask, "record_id"].tolist()
        print(f"  Filtering {bad_mask.sum()} records with null/empty source_text: {bad_ids}")
        df_filtered = df_filtered[~bad_mask].copy()

    return df_filtered


def stratified_sample(
    df: pd.DataFrame,
    size: int,
    seed: int,
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Draw `size` records stratified across all 44 categories.

    Within each category, records are balanced by prompt_style.
    `exclude_ids`: set of record_id strings already in the existing sample;
                   these are removed from the pool before sampling.
    """
    pool = df.copy()
    if exclude_ids:
        pool = pool[~pool["record_id"].isin(exclude_ids)]
        print(f"  Pool after excluding existing: {len(pool):,} records")

    if len(pool) < size:
        raise ValueError(
            f"Not enough records in pool ({len(pool):,}) to draw {size:,}. "
            f"Try a smaller --size or --extend."
        )

    n_cats = pool["category"].nunique()
    base      = size // n_cats
    remainder = size - base * n_cats

    # Categories with the most records get +1 to absorb remainder
    cat_sizes        = pool.groupby("category").size().sort_values(ascending=False)
    cats_with_extra  = set(cat_sizes.head(remainder).index)

    parts = []
    for cat, grp in pool.groupby("category"):
        quota = base + (1 if cat in cats_with_extra else 0)
        quota = min(quota, len(grp))

        # Balance prompt_style within category
        styles    = grp["prompt_style"].unique()
        per_style = max(1, quota // len(styles))
        style_parts = []
        for _style, sg in grp.groupby("prompt_style"):
            n = min(per_style, len(sg))
            style_parts.append(sg.sample(n=n, random_state=seed))

        cat_sample = pd.concat(style_parts).head(quota)

        # Top-up if style balancing fell short
        if len(cat_sample) < quota:
            chosen_ids = set(cat_sample.index)
            remaining  = grp[~grp.index.isin(chosen_ids)]
            topup = remaining.sample(
                n=min(quota - len(cat_sample), len(remaining)),
                random_state=seed,
            )
            cat_sample = pd.concat([cat_sample, topup])

        parts.append(cat_sample)

    result = (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    if len(result) != size:
        # Edge case: some categories had fewer records than quota
        print(
            f"  Warning: requested {size} but pool only allowed {len(result)}. "
            "Consider reducing --size."
        )

    # Add translation strategy and cipher payload columns
    result_with_strategy = add_translation_columns(result)

    return result_with_strategy[["record_id", "question_id", "source_text", "category",
                                "category_name", "prompt_style", "translation_strategy",
                                "cipher_payload"]]


def add_translation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add translation_strategy and cipher_payload columns to the sample.

    For each record:
    - translation_strategy: determined by prompt_style classification
    - cipher_payload: extracted cipher content for cipher styles, empty for others

    Args:
        df: DataFrame with source_text and prompt_style columns

    Returns:
        DataFrame with added translation_strategy and cipher_payload columns
    """
    result = df.copy()

    # Initialize new columns
    result["translation_strategy"] = ""
    result["cipher_payload"] = ""

    print("  Computing translation strategies and cipher payloads...")

    for idx, row in result.iterrows():
        prompt_style = row["prompt_style"]
        source_text = row["source_text"]

        # Classify the prompt style to get translation strategy
        strategy = classify_prompt_style(prompt_style)
        result.at[idx, "translation_strategy"] = strategy.value

        # For cipher styles, extract the cipher payload
        if strategy.value == "cipher_template":
            try:
                _, cipher_payload = parse_cipher_prompt(source_text, prompt_style)
                result.at[idx, "cipher_payload"] = cipher_payload
            except ValueError as e:
                # If parsing fails, log warning and fall back to normal strategy
                print(f"    Warning: Failed to parse {prompt_style} prompt for record {row.get('record_id', idx)}: {e}")
                print(f"    Falling back to 'normal' strategy for this record.")
                result.at[idx, "translation_strategy"] = "normal"
                result.at[idx, "cipher_payload"] = ""
        else:
            # Non-cipher styles have empty cipher_payload
            result.at[idx, "cipher_payload"] = ""

    # Summary of strategy distribution
    strategy_counts = result["translation_strategy"].value_counts()
    print(f"  Translation strategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"    {strategy}: {count:,} records")

    return result


def push_to_hf(df: pd.DataFrame, repo: str, private: bool, token: str | None) -> None:
    from datasets import Dataset
    print(f"Pushing {len(df):,} records to HF: {repo} (private={private})...")
    Dataset.from_pandas(df).push_to_hub(repo, private=private, token=token)
    print(f"  Done. Load with: load_dataset('{repo}', split='train')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical SorryBench sample for translation"
    )
    parser.add_argument(
        "--size", type=int, default=1000,
        help="Number of records in the initial sample (default: 1000)",
    )
    parser.add_argument(
        "--extend", type=int, default=None,
        help="Add N more records to an existing sample without touching existing rows",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--push-hf", metavar="REPO", default=None,
        help="HuggingFace dataset repo to push to (e.g. your-username/sorry-bench-202503-sample)",
    )
    parser.add_argument(
        "--private", action="store_true", default=True,
        help="Make HF dataset private (default: True)",
    )
    parser.add_argument(
        "--public", action="store_true", default=False,
        help="Make HF dataset public (overrides --private)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set — dataset download may fail for gated repos.")

    # ── Load SorryBench ───────────────────────────────────────────────────────
    df_source = load_sorry_bench_202503(hf_token)

    # ── Determine mode: create or extend ──────────────────────────────────────
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.extend is not None:
        if not output_path.exists():
            print(f"Error: --extend requires an existing sample at {output_path}")
            print("Run without --extend first to create the initial sample.")
            sys.exit(1)

        existing = pd.read_csv(output_path, encoding="utf-8")
        existing_ids = set(existing["record_id"].astype(str))
        print(f"Extending existing sample ({len(existing):,} records) by {args.extend:,}...")

        new_records = stratified_sample(
            df_source,
            size=args.extend,
            seed=args.seed,
            exclude_ids=existing_ids,
        )

        combined = pd.concat([existing, new_records], ignore_index=True)
        print(f"  New total: {len(combined):,} records")

        tmp = output_path.with_suffix(".tmp")
        combined.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, output_path)
        print(f"Saved to {output_path}")
        df_final = combined

    else:
        if output_path.exists():
            print(
                f"Warning: {output_path} already exists. "
                "Use --extend to add records, or delete the file to recreate."
            )
            sys.exit(1)

        print(f"Creating stratified sample of {args.size:,} records (seed={args.seed})...")
        sample = stratified_sample(df_source, size=args.size, seed=args.seed)
        print(f"  Sampled {len(sample):,} records")

        # Summary
        print(f"  Categories: {sample['category'].nunique()}")
        print(f"  Prompt styles: {sample['prompt_style'].nunique()}")
        print(f"  Category distribution (min/max): "
              f"{sample['category'].value_counts().min()} / "
              f"{sample['category'].value_counts().max()}")

        tmp = output_path.with_suffix(".tmp")
        sample.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, output_path)
        print(f"Saved to {output_path}")
        df_final = sample

    # ── Optional HF push ──────────────────────────────────────────────────────
    if args.push_hf:
        private = not args.public
        push_to_hf(df_final, repo=args.push_hf, private=private, token=hf_token)


if __name__ == "__main__":
    main()
