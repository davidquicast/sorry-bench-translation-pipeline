# Multilingual Translation Pipeline for Safety Benchmarks

[![Source](https://img.shields.io/badge/HuggingFace-sorry--bench--202503-orange?logo=huggingface)](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503)
[![Tier 1 Translations](https://img.shields.io/badge/HuggingFace-cohere--translation--tier1-blue?logo=huggingface)](https://huggingface.co/datasets/tiny-aya-safety/sorry-bench-202503-cohere-translation-tier1)
[![Multilingual Clean](https://img.shields.io/badge/HuggingFace-multilingual--clean-green?logo=huggingface)](https://huggingface.co/datasets/tiny-aya-safety/sorry-bench-202503-multilingual)

Translation pipeline for NLP safety research. Translates safety benchmarks across languages without data loss from LLM refusals.

**Primary use case:** Translating [SorryBench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) (Xie et al., ICLR 2025) into 9 languages for the [XSafetyRep](docs/safety%20representations%20across%20languages%20-%20tiny%20aya.pdf) project, which studies how [Tiny Aya](docs/tiny_aya_tech_report.pdf) (Abagyan et al., 2025; [arXiv 2506.10766](https://arxiv.org/abs/2506.10766)) encodes safety across languages.

**Notable design detail:** One optional backend is [Aya Expanse](https://huggingface.co/CohereLabs/aya-expanse-8b) (Cohere), from the same model family as Tiny Aya: meaning the pipeline can use a model from the same family to translate the benchmark that will evaluate it.

**Generated datasets:**

- [sorry-bench-202503-cohere-translation-tier1](https://huggingface.co/datasets/tiny-aya-safety/sorry-bench-202503-cohere-translation-tier1): full translation log with metadata (strategy, status, latency, cipher payload) for 6,596 records x 9 languages = 59,364 rows
- [sorry-bench-202503-multilingual](https://huggingface.co/datasets/tiny-aya-safety/sorry-bench-202503-multilingual): clean dataset mirroring SorryBench schema with a `language` column, 65,960 rows (6,596 records x 10 languages including English)

> **Note on `src/aya_safety/`:** this package contains a full three-tier pipeline framework (NLLB backends, LLM refinement, QA metrics, CLI, checkpoint manager). It was written as a forward-looking architecture but has not been tested end-to-end. The scripts in `scripts/` and the notebooks in `notebooks/` are what was actually run and validated.

---

## Quickstart (local, Cohere only)

```bash
# 1. Install
pip install -r requirements-cohere.txt
cp .env.example .env   # add COHERE_API_KEY and HF_TOKEN

# 2. Generate sample (all available records, stratified across 44 categories)
python scripts/prepare_sample.py --extend 9999 --output data/sorry-bench-202503-sample.csv

# 3. Translate into 9 languages (strategy-aware, ~50 000+ API calls)
python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv --output data/sorry-bench-202503-cohere-translation-tier1.csv
# → data/sorry-bench-202503-cohere-translation-tier1.csv

# 4. Build the clean multilingual dataset (mirrors SorryBench schema + language column)
python scripts/build_multilingual_dataset.py --push-hf your-username/sorry-bench-202503-multilingual
# → data/sorry-bench-202503-multilingual.csv  (English originals + 9 translations per record)
```

That's it. Cipher-style prompts (ascii/atbash/caesar/morse) are assembled from pre-translated templates: no API calls. Misspellings and slang get specialized prompts that preserve the stylistic effects in the target language.

If you want NLLB-200 as the backbone + Cohere refinement, see [Running on Kaggle or Colab](#running-on-kaggle-or-colab).

---

## The Three-Tier Architecture

The framework uses a cascading three-tier pipeline that architecturally solves the LLM refusal problem: where LLMs refuse 36.7–46.7% of translation requests containing harmful content (Maskey et al., 2025; [SafeConstellations](https://arxiv.org/abs/2508.11290)).

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT DATASET                             │
│        HuggingFace dataset ID or local CSV/JSON/Parquet          │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1: NLLB-200 (Backbone: always runs, $0)                  │
│                                                                  │
│  • Dedicated MT model with NO content filters                    │
│  • Every record ALWAYS gets a translation: nothing is lost      │
│  • facebook/nllb-200-3.3B (primary), 1.3B, or 600M              │
│  • Auto-selects model by available VRAM                          │
│  • Optional CTranslate2 optimization (3-5× speedup)             │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  TIER 2: LLM Refinement (Optional: $2-112)                     │
│                                                                  │
│  • Framed as "improve this MT output", NOT "translate this"      │
│  • This framing dramatically reduces refusal rates               │
│  • Any detected refusal → automatic fallback to Tier 1 output   │
│  • Fallback chain: primary → fallback1 → fallback2 → Tier 1     │
│  • Backends: Cohere, OpenAI, Anthropic, Gemini, local HF,       │
│    Google Translate, DeepL, Azure, Amazon                        │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  TIER 3: Quality Assurance (Optional: $0-10)                    │
│                                                                  │
│  • COMET (Unbabel/wmt22-cometkiwi-da): reference-free           │
│  • chrF++ (SacreBLEU): character n-grams                        │
│  • Back-translation + BERTScore: round-trip consistency         │
│  • GEMBA (Kocmi & Federmann, 2023): LLM-as-judge (0-100)       │
│  • Language detection (fast-langdetect)                          │
│  • Safety label consistency (round-trip category preservation)   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: Parquet/JSONL/CSV + Report + Flagged Records CSV        │
│  Optional: push to HuggingFace Hub                               │
└──────────────────────────────────────────────────────────────────┘
```

### How the refusal problem is solved

1. **NLLB-200 is the unconditional backbone.** It has no content filters (Meta, 2022). Every record always gets a translation. Nothing is ever lost to LLM refusal.

2. **LLM refinement is framed as "improve this MT output"**, not "translate this harmful text." This framing reduces refusal rates dramatically because the LLM is editing an existing translation rather than generating harmful content from scratch.

3. **Any detected refusal automatically falls back to the Tier 1 MT output.** Refusal detection uses multilingual regex patterns, copy-through detection, and length anomaly checks. Every fallback is logged with full metadata.

This architecture is validated by PolyGuard (Kumar et al., 2025; [arXiv 2504.04377](https://arxiv.org/abs/2504.04377)) and LinguaSafe's TATER framework (Ning et al., 2025; [arXiv 2508.12733](https://arxiv.org/abs/2508.12733)), which reduced Bengali translation errors from 71% to 12%.

---

## Prompt-Style-Aware Translation

SorryBench encodes harmful intent using 21 stylistic variants beyond plain English. Translating all styles identically produces wrong results and wastes API calls. The pipeline classifies each record and applies a different strategy per style group:

| Style group | Examples | Strategy | Why |
|---|---|---|---|
| **Normal** (9 styles) | `base`, `role_play`, `question`, `logical_appeal` | Translate as-is | Standard MT/LLM translation |
| **Cipher** (4 styles) | `ascii`, `atbash`, `caesar`, `morse` | Assemble from pre-translated template, no API call | The cipher instruction template is identical for every record of that style: only the encoded payload changes. Translating the template once per language and splicing the payload saves ~3,000 API calls across 1,000 records. |
| **Misspellings** | `misspellings` | LLM with custom prompt | Preserve intentional spelling errors in the target language (not correct them) |
| **Slang** | `slang` | LLM with custom prompt | Use natural slang of the target language, not a literal translation |
| **Excluded** (6 styles) | `translate-fr`, `uncommon_dialects` | Filtered out | Already non-English source text |

**What `prepare_sample.py` does:** before any translation, it classifies each record and stores `translation_strategy` and `cipher_payload` columns in the sample CSV. Every downstream script and notebook reads these columns directly: no repeated classification logic.

**Morse table note:** the A-Z cipher table stays Latin in ALL languages because the encoded content is English text. Only the surrounding instructional prose is translated.

---

## Output Schema

Every output CSV produced by the pipeline follows this column order:

| Column | Description |
|---|---|
| `question_id` | Original SorryBench question ID |
| `target_lang` | ISO 639-1 code of the target language |
| `source_text` | Original English prompt |
| `category` | SorryBench category number (1–44) |
| `category_name` | Human-readable category label |
| `prompt_style` | SorryBench prompt style (e.g. `base`, `caesar`) |
| `translation_strategy` | `normal` / `cipher_template` / `misspellings` / `slang` |
| `cipher_payload` | Encoded payload for cipher records (empty otherwise) |
| `tier1_translation` | Tier 1 translation output |
| `tier1_backend` | Model or method used (`command-a-03-2025`, `cipher_template`, etc.) |
| `tier1_status` | `ok` / `success` (cipher) / `refused` / `needs_llm` (NLLB → misspellings/slang) |
| `tier1_latency_ms` | Request time in milliseconds (0 for cipher) |
| `retries` | Number of retry attempts (0 on first-try success) |

After Tier 2 refinement (`refine_with_cohere.py`), columns `tier2_translation`, `tier2_backend`, `tier2_status`, `tier2_latency_ms`, and `final_translation` are appended.

---

## Resilience and Checkpointing

`translate_with_cohere.py` is designed to run unattended on large datasets:

- **Atomic saves**: checkpoint written to `.tmp` then renamed; a crash mid-write never corrupts the output.
- **Resume**: re-running the same command skips already-completed records (`ok`, `success`, `refused`). Only errors are retried automatically.
- **Retry logic**: transient failures (timeout, 5xx) retry with exponential backoff (2 s, 4 s, 8 s…). Rate-limit errors (429) use a longer wait (30 s, 60 s, 90 s) because Cohere's limit is per-minute, not per-second.
- **Error exclusion**: records that exhaust all retries are intentionally excluded from the checkpoint. The next run will attempt them again, with no manual cleanup needed.
- **HF push**: pass `--push-hf user/repo` to upload the checkpoint to HuggingFace after every `--save-every` completions (default: 100). The dataset is pushed as public by default.

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys (Cohere, OpenAI, etc.)
```

If you only have Cohere and do not want to run local models, use the lighter install instead:

```bash
pip install -r requirements-cohere.txt
```

---

## Running on Kaggle or Colab

### Option A: NLLB-200 (free GPU, recommended)

NLLB-200-3.3B requires ~7 GB VRAM and runs free on Kaggle (2× T4, 30 h/week) or Google Colab (1× T4). It has no content filters and never refuses.

**Step 1.** Locally, generate the canonical sample and push it to HuggingFace:

```bash
python scripts/prepare_sample.py --size 1000 --push-hf your-username/sorry-bench-202503-sample
```

This ensures that if you run multiple notebooks in parallel they all translate the **exact same records**.

**Step 2.** Upload `notebooks/tier1_nllb_translate.ipynb` to Kaggle or Colab and open it.

**Step 3.** Enable GPU:
- Kaggle: Settings → Accelerator → GPU T4 × 2
- Colab: Runtime → Change runtime type → T4 GPU

**Step 4.** In Cell 3, set:
```python
PLATFORM   = "kaggle"    # or "colab"
PRECISION  = "half"      # float16, fits on both platforms
SAMPLE_CSV = "your-username/sorry-bench-202503-sample"   # HF dataset from Step 1
```
Set `N_RECORDS = 5` for a quick test (45 translations), or leave as `None` for the full sample.

**Step 5.** Add your `HF_TOKEN` to Kaggle/Colab Secrets, then run all cells.

**Step 6.** Place the downloaded checkpoint CSV in `data/` and run Tier 2 refinement:

```bash
python scripts/refine_with_cohere.py --input data/tier1_checkpoint.csv
```

NLLB model is auto-selected based on available VRAM:

| Total VRAM (float16) | Model selected | Quality |
|----------------------|----------------|---------|
| ≥ 7 GB | nllb-200-3.3B | Best |
| 2.5–7 GB | nllb-200-distilled-1.3B | Good |
| CPU only | nllb-200-distilled-600M | Acceptable |

---

### Option B: Cohere API (no GPU required, runs locally)

Use this if you don't have GPU access, or want higher-quality Tier 1 output and plan to skip Tier 2 refinement.

**Step 1.** Generate the sample:

```bash
python scripts/prepare_sample.py --size 1000 --output data/sorry-bench-202503-sample.csv
```

**Step 2.** Translate directly from your terminal:

```bash
python scripts/translate_with_cohere.py --input data/sorry-bench-202503-sample.csv --output data/sorry-bench-202503-cohere-translation-tier1.csv
# Output: data/sorry-bench-202503-cohere-translation-tier1.csv
```

Options:
- `--max-concurrent 10`: increase throughput with a paid Cohere key
- `--push-hf user/repo`: upload checkpoint to HuggingFace after every 100 completions (public by default)
- `--languages es,fr,de`: translate only specific languages
- `--max-retries 5`: more retry attempts for flaky connections
- `--dry-run`: preview prompts without calling the API

Model routing is automatic:
- All languages → `command-a-03-2025` (Cohere's multilingual model, 500 RPM on production keys)

> **Note on `command-a-translate-08-2025`:** Cohere's dedicated translation model exists and is supported in the codebase, but even paid production keys are capped at 20 RPM / 1,000 calls per month until Cohere sales manually enables production access for your account. Contact `sales@cohere.com` to unlock it. Once enabled, pass `--model command-a-translate` to switch.

**Kaggle/Colab alternative:** if you prefer a notebook environment, use `notebooks/tier1_cohere_translate.ipynb` with `SAMPLE_CSV` pointing to your CSV or HF dataset.

---

### Running both notebooks in parallel

Generate the sample once, set `SAMPLE_CSV` to the same HF dataset in both notebooks, and launch them simultaneously. Both will translate the same records; their checkpoint CSVs are independent.

```bash
# Once locally:
python scripts/prepare_sample.py --size 1000 --push-hf your-username/sorry-bench-202503-sample
# Then set SAMPLE_CSV = "your-username/sorry-bench-202503-sample" in both notebooks
```

---

### Growing the dataset incrementally

```bash
# Add 500 more records without touching the existing 1000:
python scripts/prepare_sample.py --extend 500 --push-hf your-username/sorry-bench-202503-sample
```

Both notebooks will resume from their existing checkpoints and only process the new records.

---

## Available Modes

| Mode | Description | Cost (83K records) | Keys needed |
|------|-------------|-------------------|-------------|
| Tier 1 only (NLLB) | NLLB-200 baseline, no GPU cost | **$0** | None |
| Tier 1 only (Cohere) | `command-a-translate-08-2025` direct translation | **~$10–20** | `COHERE_API_KEY` |
| Tier 1 + 2 (Cohere) | NLLB → `command-a-translate-08-2025` refinement | **~$10–20** | `COHERE_API_KEY` |
| Tier 1 + 2 (OpenAI) | NLLB → GPT-4o-mini refinement | **~$2–5** | `OPENAI_API_KEY` |
| Tier 1 + 2 (OpenAI Batch) | NLLB → GPT-4o-mini batch (50% off) | **~$1–3** | `OPENAI_API_KEY` |
| Tier 1 + 2 (Gemini) | NLLB → Gemini 2.5 Flash refinement | **~$5** | `GOOGLE_API_KEY` |
| Tier 1 + 2 + 3 | Full pipeline with QA scoring | +$0–10 for GEMBA | LLM key + metric keys |

---

## Worked Example: Translating SorryBench for XSafetyRep

### Step 0: Prepare the sample (run locally once)

```bash
# Create 1,000-record stratified sample and upload to HuggingFace
python scripts/prepare_sample.py --size 1000 --push-hf your-username/sorry-bench-202503-sample

# Later, grow the dataset without duplicates:
python scripts/prepare_sample.py --extend 500 --push-hf your-username/sorry-bench-202503-sample
```

Both notebooks read from this sample so they always translate the same records.

### Step 1: Estimate costs

For 6,596 records × 9 languages = 59,364 translations:

| Backend | Type | Estimated Cost | Refusal Risk |
|---------|------|---------------|--------------|
| NLLB-200 (Kaggle/Colab) | Local MT model | **$0** | None |
| Cohere command-a-translate-08-2025 | LLM API | **~$1–3** | Low |
| GPT-4o-mini Batch | LLM API | **~$0.30** | Moderate |
| Gemini 2.5 Flash | LLM API | **~$0.50** | Low-Moderate |

### Step 2: Run Tier 1 baseline

```bash
aya-safety translate \
  --config configs/sorry-bench-202503.yaml \
  --tier2-enabled false \
  --tier3-enabled false
```

This translates all sampled records into 9 languages using NLLB-200 at zero cost.

### Step 3: Run full pipeline

```bash
aya-safety translate \
  --config configs/sorry-bench-202503.yaml \
  --tier2-backend cohere \
  --resume
```

The `--resume` flag picks up from the Tier 1 checkpoint: NLLB translations are reused, and Tier 2 refines them.

### Step 4: Inspect the report

```bash
cat output/sorry-bench-202503/report.json | python -m json.tool
```

The report shows per-language completion rates, refusal counts per backend, average QA scores, and estimated costs.

### Step 5: Review flagged records

```bash
head output/sorry-bench-202503/flagged.csv
```

Records with QA scores below thresholds are flagged for manual review.

### Step 6 (optional): Push to HuggingFace Hub

```bash
aya-safety translate \
  --config configs/sorry-bench-202503.yaml \
  --push-to-hub \
  --hub-repo-id your-username/sorry-bench-202503-multilingual \
  --resume
```

---

## Cost Reference Table

For reference at full scale: 9,240 records × 9 languages = 83,160 translations:

| Service | Type | Total Cost | Cost/Record | Refusal Risk |
|---------|------|-----------|-------------|--------------|
| NLLB-200 (Kaggle/Colab) | Local model | **$0** | $0 | None |
| GPT-4o-mini (Batch API) | LLM | **~$2–5** | $0.00003 | Moderate |
| Gemini 2.5 Flash | LLM | **~$5** | $0.00006 | Low-Moderate |
| Cohere command-a-translate-08-2025 | LLM | **~$10–20** | $0.00015 | Low |
| Claude 3 Haiku | LLM | **~$9** | $0.00011 | High |
| GPT-4o | LLM | **~$78** | $0.00094 | Moderate |
| Claude 3.5 Sonnet | LLM | **~$112** | $0.00135 | High |
| Google Cloud Translation | MT API | **~$500** | $0.006 | None |
| DeepL (7 langs only) | MT API | **~$625** | $0.0075 | None |
| Azure Translator | MT API | **~$2,250** | $0.027 | None |

LLM APIs are 10-100× cheaper than dedicated MT APIs due to token-vs-character pricing.

---

## Model Recommendations by Language Group

| Language Group | Best Open-Source | Best API | Gaps |
|----------------|-----------------|----------|------|
| European (DE, FR, ES, RU) | TowerInstruct-13B, ALMA-R | DeepL, GPT-4o | None |
| CJK (JA, ZH) | GemmaX2-28-9B (Xiaomi, NAACL 2025; [arXiv 2502.02481](https://arxiv.org/abs/2502.02481)) | Google Translate, GPT-4o | DeepL recently added JA/ZH |
| South Asian (BN, HI) | NLLB-200-3.3B, GemmaX2-28 | Google Translate, Azure | **DeepL does NOT support BN or HI** |
| Arabic (AR) | NLLB-200-3.3B, GemmaX2-28 | Google Translate, Azure | DeepL Arabic support is recent |
| African | **NLLB-200 only viable option** | Google Translate, Azure | Most LLMs and DeepL lack coverage |

**Bengali warning:** Bengali is the hardest target language in this set. LinguaSafe found 71% error rates for vanilla Bengali translation. DeepL doesn't support it. NLLB-200 and GemmaX2-28-9B are the only reliable open-source options.

---

## Adding a Custom Language

### Via YAML config

```yaml
custom_languages:
  - nllb_code: "kat_Geor"
    iso_639_1: "ka"
    bcp47: "ka"
    display_name: "Georgian"
    script: "Geor"
    resource_level: "low"
    recommended_models: ["nllb-200-3.3B"]
```

### Via CLI

```bash
aya-safety translate \
  --config configs/default.yaml \
  --add-language kat_Geor:ka:ka:Georgian:Geor \
  --target-languages ka
```

---

## Prompt Engineering Rationale

The refinement prompt is fixed and based on research findings:

- **Simple, direct prompts outperform complex ones**: Jiao et al. (2023; [arXiv 2301.08745](https://arxiv.org/abs/2301.08745)) tested multiple templates; simple wins.
- **Chain-of-thought prompting degrades translation quality**: induces word-by-word translation behavior (He et al., 2023; [arXiv 2303.13780](https://arxiv.org/abs/2303.13780)).
- **English prompts for all target languages**: the research finds English prompts produce better translations than target-language prompts.
- **Professional translator persona**: reduces refusal in content-sensitive contexts.
- **Temperature 0.0**: deterministic, faithful translation with minimal verbosity.

---

## Quality Metrics

| Metric | Source | What It Measures |
|--------|--------|------------------|
| **COMET** | Unbabel/wmt22-cometkiwi-da | Reference-free quality estimation; highest correlation with human judgments |
| **chrF++** | SacreBLEU | Character n-grams + word bigrams; robust for morphologically rich languages |
| **Back-translation + BERTScore** | Moon et al. (2020; [arXiv 2004.13937](https://arxiv.org/abs/2004.13937)) | Round-trip semantic consistency check |
| **GEMBA** | Kocmi & Federmann (2023; [arXiv 2302.14520](https://arxiv.org/abs/2302.14520); [GitHub](https://github.com/MicrosoftTranslator/GEMBA)) | LLM-as-judge scoring 0-100; state-of-the-art MQM correlation |
| **Language detection** | fast-langdetect (FastText-based) | Verifies translation is in the target language |
| **Safety consistency** | CultureGuard (2025; [arXiv 2508.01710](https://arxiv.org/html/2508.01710)) | Safety label preserved after round-trip translation |

---

## Checkpointing and Resume

The pipeline saves atomic checkpoints (write to `.tmp`, then `os.replace()`) per language after each batch. If interrupted:

```bash
aya-safety translate --config configs/sorry-bench-202503.yaml --resume
```

This resumes from the exact point of interruption. No translations are duplicated or lost.

Checkpoint files are stored in `./checkpoints/` as JSONL + metadata JSON per language. Config hash validation prevents accidental resume with incompatible settings (use `--force-resume` to override).

---

## Building the Clean Multilingual Dataset

After translation, `build_multilingual_dataset.py` assembles a dataset that mirrors SorryBench's original schema, extended with a `language` column. English originals are included so the dataset is self-contained.

### Output schema

| Column | Type | Description |
|---|---|---|
| `question_id` | int | Original SorryBench question ID |
| `category` | int | Harm category (1–44) |
| `prompt_style` | str | SorryBench prompt style |
| `language` | str | ISO 639-1 code (`en` for originals, `ar`/`bn`/… for translations) |
| `turns` | str | Prompt text in the target language |

Refused translations are excluded. Each record yields 1 English row + up to 9 translated rows.

### Usage

```bash
# Build from local files (default paths)
python scripts/build_multilingual_dataset.py

# Build and push to HuggingFace
python scripts/build_multilingual_dataset.py --push-hf your-username/sorry-bench-202503-multilingual

# Custom paths
python scripts/build_multilingual_dataset.py \
  --translations data/sorry-bench-202503-cohere-translation-tier1.csv \
  --sample data/sorry-bench-202503-sample.csv \
  --output data/sorry-bench-202503-multilingual.csv \
  --push-hf your-username/sorry-bench-202503-multilingual
```

If a local file is missing, it is downloaded automatically from HuggingFace using `--hf-translations` (default: `davidquicast/sorry-bench-202503-cohere-translation-tier1`).

---

## Translating the Full SorryBench Dataset

The published datasets cover the full SorryBench dataset after filtering excluded styles: **6,596 records x 9 languages = 59,364 translations**, plus 6,596 English originals = 65,960 rows total.

To reproduce from scratch:

**Step 2: Translate:**

```bash
python scripts/translate_with_cohere.py \
  --input data/sorry-bench-202503-full.csv \
  --output data/sorry-bench-202503-cohere-translation-full.csv \
  --max-concurrent 5 \
  --max-retries 5 \
  --push-hf your-username/sorry-bench-202503-cohere-translation-full
```

At 500 RPM (`command-a-03-2025` production), 83,000 calls takes roughly 3–4 hours with `--max-concurrent 5`. The script checkpoints every 100 completions and resumes automatically if interrupted.

**Step 3: Build the clean multilingual dataset:**

```bash
python scripts/build_multilingual_dataset.py \
  --translations data/sorry-bench-202503-cohere-translation-full.csv \
  --sample data/sorry-bench-202503-full.csv \
  --output data/sorry-bench-202503-multilingual-full.csv \
  --push-hf your-username/sorry-bench-202503-multilingual-full
```

> **Note on `command-a-translate-08-2025`:** Cohere's dedicated translation model is production-limited to 20 RPM / 1,000 calls/month until sales approval. For full-scale translation use `command-a-03-2025` (the default). Pass `--model command-a-translate` once your account is upgraded.

---

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `output/translations_{lang}.parquet` | Parquet/JSONL/CSV | All records with Tier 1, Tier 2, and QA scores |
| `output/report.json` | JSON | Per-language stats, per-backend stats, QA summary, timing, cost |
| `output/flagged.csv` | CSV | Records below QA thresholds for manual review |
| `checkpoints/{run_id}_{lang}.jsonl` | JSONL | Checkpoint data for resume |

---

## Full CLI Reference

```bash
aya-safety translate [OPTIONS]

Options:
  --config, -c PATH          YAML config file
  --resume                   Resume from checkpoint
  --force-resume             Resume even if config hash changed
  --source TEXT               HF dataset ID or local file path
  --source-language TEXT      Source language ISO code
  --target-languages LIST     Comma-separated target language codes
  --add-language TEXT          Custom language (nllb_code:iso:bcp47:name:script)
  --tier1-backend TEXT        Tier 1 backend (nllb, nllb-ct2)
  --tier1-model TEXT          Tier 1 model name or 'auto'
  --tier2-enabled BOOL        Enable/disable Tier 2
  --tier2-backend TEXT        Primary Tier 2 backend
  --tier2-model TEXT          Model for primary Tier 2 backend
  --tier2-fallback LIST       Comma-separated fallback backends
  --tier3-enabled BOOL        Enable/disable Tier 3
  --tier3-metrics LIST        Comma-separated metric names
  --batch-size INT            Batch size for Tier 1
  --max-concurrent INT        Max concurrent Tier 2 API calls
  --device TEXT               Device (auto, cuda, cpu)
  --output-dir PATH           Output directory
  --output-format FORMAT      Output format (parquet, jsonl, csv)
  --push-to-hub              Push to HuggingFace Hub
  --hub-repo-id TEXT          HF Hub repository ID
  --log-level LEVEL           Logging level (DEBUG, INFO, WARNING, ERROR)
```

---

## Bibliography

- Abagyan, D. et al. (2025). Tiny Aya: Bridging Scale and Multilingual Depth. [arXiv 2506.10766](https://arxiv.org/abs/2506.10766)
- Costa-jussà, M. R. et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. Meta AI. [arXiv 2207.04672](https://arxiv.org/abs/2207.04672)
- Friedrich, F. et al. (2024). M-ALERT: A Multilingual Benchmark for Safety Evaluation. [arXiv 2412.15035](https://arxiv.org/abs/2412.15035)
- He, Z. et al. (2023). Exploring Human-Like Translation Strategy with Large Language Models. [arXiv 2303.13780](https://arxiv.org/abs/2303.13780); TACL [doi.org/10.1162/tacl_a_00642](https://doi.org/10.1162/tacl_a_00642)
- Jiao, W. et al. (2023). Is ChatGPT a Good Translator? [arXiv 2301.08745](https://arxiv.org/abs/2301.08745)
- Kocmi, T. & Federmann, C. (2023). Large Language Models Are State-of-the-Art Evaluators of Translation Quality (GEMBA). [arXiv 2302.14520](https://arxiv.org/abs/2302.14520); [GitHub](https://github.com/MicrosoftTranslator/GEMBA)
- Kumar, A. et al. (2025). PolyGuard: A Multilingual Safety Moderation Tool. [arXiv 2504.04377](https://arxiv.org/abs/2504.04377)
- Maskey, S. et al. (2025). SafeConstellations: Measuring Cross-Lingual Safety Alignment. [arXiv 2508.11290](https://arxiv.org/abs/2508.11290)
- Moon, H. et al. (2020). Revisiting Round-Trip Translation for Quality Estimation. [arXiv 2004.13937](https://arxiv.org/abs/2004.13937)
- Ning, X. et al. (2025). LinguaSafe: TATER Framework for Multilingual Safety Translation. [arXiv 2508.12733](https://arxiv.org/abs/2508.12733)
- Rei, R. et al. (2022). COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task. Unbabel. [GitHub](https://github.com/Unbabel/COMET)
- Singh, N. et al. (2024). Aya Collection: Templating and Translating for Multilingual NLP. ACL 2024. [arXiv 2402.06619](https://arxiv.org/abs/2402.06619)
- Wang, Y. et al. (2024). All Languages Matter: On the Multilingual Safety of LLMs (XSafety). ACL Findings 2024. [arXiv 2310.00905](https://arxiv.org/abs/2310.00905)
- Xie, T. et al. (2025). SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal. ICLR 2025. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/9622163c87b67fd5a4a0ec3247cf356e-Paper-Conference.pdf)
- Xu, H. et al. (2024). ALMA-R: Contrastive Preference Optimization for Translation. ICML 2024. [arXiv 2401.08417](https://arxiv.org/abs/2401.08417)
- Xu, H. et al. (2025). X-ALMA: Plug & Play Modules and Training Recipes for Multilingual LLM. ICLR 2025. [arXiv 2410.03115](https://arxiv.org/abs/2410.03115)
- Xiaomi Research (2025). GemmaX2-28: Large-Scale Multilingual NMT. NAACL 2025. [arXiv 2502.02481](https://arxiv.org/abs/2502.02481)
- CultureGuard (2025). Cross-Lingual Safety Consistency Filter. [arXiv 2508.01710](https://arxiv.org/html/2508.01710)
- Feng, Y. et al. (2025). TEaR: Translate, Estimate, and Refine. NAACL Findings 2025. [arXiv 2402.16379](https://arxiv.org/abs/2402.16379)
- Deng, Y. et al. (2024). MultiJail: Multilingual Jailbreak Attacks. ICLR 2024. [arXiv 2310.06474](https://arxiv.org/abs/2310.06474)
