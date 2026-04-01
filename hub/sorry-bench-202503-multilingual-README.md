---
language:
- en
- ar
- bn
- de
- es
- fr
- hi
- ja
- ru
- zh
license: mit
task_categories:
- text-generation
tags:
- safety
- multilingual
- sorry-bench
- jailbreak
- harmful-content
---

# sorry-bench-202503-multilingual

Multilingual version of [SorryBench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) — a benchmark for evaluating LLM safety refusals across 44 harm categories and 21 prompt styles.

This dataset contains 1,000 stratified English prompts from SorryBench translated into 9 languages, plus the original English, for a total of 10,000 rows.

## Schema

| Column | Type | Description |
|---|---|---|
| `question_id` | int | Original SorryBench question ID |
| `category` | int | Harm category (1–44) |
| `prompt_style` | string | SorryBench prompt style (e.g. `base`, `role_play`, `ascii`, `morse`) |
| `language` | string | ISO 639-1 code (`en` for originals, `ar`/`bn`/etc. for translations) |
| `turns` | string | Prompt text in the target language |

## Languages

English (`en`), Arabic (`ar`), Bengali (`bn`), German (`de`), Spanish (`es`), French (`fr`), Hindi (`hi`), Japanese (`ja`), Russian (`ru`), Chinese Simplified (`zh`)

## Translation method

- **Normal/role-play/slang/misspelling styles** — translated with Cohere `command-a-03-2025`
- **Cipher styles** (ascii, atbash, caesar, morse) — instructional template translated per language; cipher payload preserved as-is (it encodes English content)

## Source

Sampled from `sorry-bench/sorry-bench-202503` (stratified across all 44 categories and 21 prompt styles). Records with null source text in the original dataset were excluded.
