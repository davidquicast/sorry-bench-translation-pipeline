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

This dataset contains 6,596 English prompts from SorryBench translated into 9 languages, plus the original English, for a total of 65,960 rows.

## Schema

| Column | Type | Description |
|---|---|---|
| `question_id` | int | Original SorryBench question ID |
| `category` | int | Harm category (1-44) |
| `prompt_style` | string | SorryBench prompt style (e.g. `base`, `role_play`, `ascii`, `morse`) |
| `language` | string | ISO 639-1 code (`en` for originals, `ar`/`bn`/etc. for translations) |
| `turns` | list[string] | Prompt text in the target language. Empty list `[]` for refused or untranslatable records |

## Languages

English (`en`), Arabic (`ar`), Bengali (`bn`), German (`de`), Spanish (`es`), French (`fr`), Hindi (`hi`), Japanese (`ja`), Russian (`ru`), Chinese Simplified (`zh`)

## Translation method

- **Normal/role-play/slang/misspelling styles** -- translated with Cohere `command-a-03-2025`
- **Cipher styles** (ascii, atbash, caesar, morse) -- instructional template translated per language; cipher payload preserved as-is (it encodes English content)
- **Refused or too-long records** -- included with `turns = []` to preserve coverage

## Excluded prompt styles

The following 6 prompt styles from the original SorryBench (9,240 rows) were excluded, reducing the base to 6,596 records:

| Style | Reason |
|---|---|
| `translate-fr` | Prompt already written in French |
| `translate-ml` | Prompt already written in Malayalam |
| `translate-mr` | Prompt already written in Marathi |
| `translate-ta` | Prompt already written in Tamil |
| `translate-zh-cn` | Prompt already written in Chinese Simplified |
| `uncommon_dialects` | Non-standard English variants not suitable for translation |

Translating these into a third language would produce semantically inconsistent or duplicate content.

## Source and pipeline

Prompts sampled from `sorry-bench/sorry-bench-202503` (stratified across all 44 harm categories). Records with null source text in the original dataset were also excluded (4 records).

The full translation pipeline is available at: https://github.com/davidquicast/sorry-bench-translation-pipeline

Raw translations with metadata (model, latency, retry count, status per row) are available at: https://huggingface.co/datasets/tiny-aya-safety/sorry-bench-202503-cohere-translation-tier1
