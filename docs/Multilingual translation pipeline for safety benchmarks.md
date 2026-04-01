# Multilingual translation pipeline for safety benchmarks: a comprehensive specification

**The most reliable approach for translating SorryBench's ~9,240 harmful prompts across 10 languages is a hybrid pipeline combining dedicated machine translation models (which have no content refusal mechanisms) with LLM-based quality refinement.** This strategy, validated by leading multilingual safety benchmarks like PolyGuard (COLM 2025) and LinguaSafe (2025), avoids the critical problem where LLMs refuse to translate harmful content—a failure mode measured at **36.7–46.7% refusal rates** in recent studies. The total cost ranges from **$0** (NLLB-200 on free GPU platforms) to **~$112** (Cohere Aya Expanse API for all 9 target languages), with quality evaluation achievable through a multi-metric ensemble of COMET, chrF++, and back-translation verification. Every one of the 10 target languages (Arabic, Bengali, German, English, French, Hindi, Japanese, Russian, Spanish, Chinese) is covered by at least three viable model options, though Bengali and Hindi present the narrowest provider support among commercial APIs.

---

## 1. What prior safety benchmarks teach us about translation methodology

No off-the-shelf "translate safety benchmark to N languages" pipeline exists. Every major multilingual safety benchmark built its own translation workflow, and their methodological choices reveal a clear hierarchy of approaches.

**SorryBench itself already includes 5 translated language mutations** (Malayalam, Tamil, Marathi, Chinese Simplified, French) implemented via the `googletrans` Python library in its mutation pipeline. The base dataset contains **450 unsafe instructions across 45 fine-grained safety categories** (v2), with 20 linguistic mutations producing ~9,000 total records. The official repository is at `github.com/SORRY-Bench/sorry-bench`, with HuggingFace datasets at `sorry-bench/sorry-bench-202503`. No standalone multilingual translation of the full dataset exists beyond these 5 built-in mutations.

The gold-standard benchmarks used **professional human translators**: XSafety (Wang et al., ACL Findings 2024; `arxiv.org/abs/2310.00905`) hired professional translators to render 2,800 safety instances across the exact same 10 languages targeted by XSafetyRep, with two rounds of proofreading. MultiJail (Deng et al., ICLR 2024; `arxiv.org/abs/2310.06474`) used native speakers to manually translate 315 harmful prompts into 9 languages, requiring >97% verification pass rates. These approaches completely sidestep the LLM refusal problem but are expensive and slow for ~83,160 translation calls.

The most sophisticated automated pipelines come from two recent projects. **PolyGuard** (Kumar et al., 2025; `arxiv.org/abs/2504.04377`) used a three-tier system: TowerInstruct-7B for 9 supported languages, NLLB-3.3B for remaining languages, and GPT-4o in an agentic framework for high-fidelity test translations, with a GPT-4o-distilled Qwen2.5-7B quality classifier achieving 82% accuracy at filtering bad translations. **LinguaSafe** (Ning et al., 2025; `arxiv.org/abs/2508.12733`) developed the **TATER framework** (Task-Aware Translate, Estimate, and Refine): initial MT translation → quality estimation with attention to harmful intent preservation → LLM iterative refinement → human review. TATER reduced translation error rates from **71% to 12% for Bengali** and 36% to 3% for Malay.

The **Aya Collection** (Singh et al., ACL 2024; `arxiv.org/abs/2402.06619`) provides the best precedent for massive-scale automated translation: Cohere used **NLLB-200 3.3B** to translate 19 English instruction-tuning datasets into **101 languages (114 dialects)**, producing 7.53M translated prompts. Quality controls included language filtering, template pruning, and human-annotation-based data pruning that removed ~19% of instances.

Key open-source batch translation tools that can serve as building blocks include:

- **Large_dataset_translator** (`github.com/vTuanpham/Large_dataset_translator`): Google Translate with multithreading, ~1,500 rows/minute, supports LLM-based context-aware translation via Groq API
- **nllb-serve** (`github.com/thammegowda/nllb-serve`): REST API + CLI batch tool for NLLB-200, configurable batch size, GPU-accelerated
- **stopes** (`github.com/facebookresearch/stopes`): Meta's modular MT pipeline from the NLLB project, including preprocessing, bitext mining, and evaluation
- **EasyNMT** (UKPLab): Wraps Opus-MT, mBART-50, and M2M-100 with batch support, multi-GPU processing, and Docker deployment
- **Andrew Ng's translation-agent** (`github.com/andrewyng/translation-agent`): Three-step agentic workflow (translate → reflect → refine) with 5.7k GitHub stars

---

## 2. The refusal problem is quantified and solvable

When LLMs encounter harmful text in a translation request, they frequently treat it as a content generation request rather than a translation task—the single most critical engineering challenge for this pipeline.

The **SafeConstellations** study (Maskey et al., 2025; `arxiv.org/abs/2508.11290`) provides the most precise measurements: **LLaMA-3.1-8B refuses 46.7%** of benign translation tasks containing harmful content, while **GPT-4o refuses 36.7%**. Refusal rates are disproportionately high for low-resource languages (Hindi, Urdu, Nepali). Google's own research (reported by Slator, October 2024) found that "refusal to translate" is the most common form of LLM verbosity, with **Claude-3.5 being the most refusal-prone** and GPT-4/Aya-23 being the least verbose.

Paradoxically, a March 2026 study (`arxiv.org/html/2603.11914`) documented the opposite failure mode: some LLMs faithfully translate and even *elaborate* on harmful content (e.g., weapon manufacturing instructions) when framed as translation tasks, because "moral filters might not be activated in the context of a seemingly benign task." Analysis of RLHF training data revealed **no entries resembling translation-type alignment cases**, explaining both failure modes.

**Dedicated MT models completely avoid the refusal problem.** NLLB-200, Helsinki-NLP Opus-MT, Google Cloud Translation API, and DeepL have no alignment-based content filtering during inference. NLLB-200 was specifically tested for toxicity during development—Meta built the ETOX toxicity detection tool and Toxicity-200 word lists—but the model itself translates any content faithfully. M-ALERT (Friedrich et al., 2024; `arxiv.org/abs/2412.15035`) specifically chose Opus-MT models for translating 15,000 safety prompts per language precisely because dedicated MT models "have NO safety alignment/refusal mechanisms."

For cases where LLM-based translation is needed (higher quality for specific language pairs), these prompting strategies reduce refusal:

**The professional translator persona** is the most widely validated approach. A system prompt establishing the role of "professional certified translator" with explicit instructions to translate faithfully, without commentary or refusal, effectively bypasses content filters in most models. Setting **temperature to 0.0** further reduces verbose refusal behavior. The research context framing—identifying the work as academic AI safety research—provides an additional layer of legitimacy. CultureGuard (2025; `arxiv.org/html/2508.01710`) validated the **cross-lingual safety consistency filter**: a sample is kept only if the safety label of the original English text and the back-translated English text remain identical.

The **recommended architecture** is a cascading fallback system: attempt LLM translation first; for any refusals (detected via pattern matching on refusal phrases), automatically route to Google Translate API or NLLB-200; optionally use an LLM in a second pass to refine the MT output (the "refine MT" framing triggers far fewer refusals than "translate harmful content").

---

## 3. Model and API landscape covers all 10 languages, with important gaps

### Commercial translation APIs

**Microsoft Azure Translator** offers the best value among dedicated MT APIs at **$10/million characters** with a generous 2M characters/month free tier and support for all 10 target languages. **Google Cloud Translation API v3** charges $20/million characters with 500K characters/month free and supports 130+ languages with consistently strong quality across all targets, including Bengali and Hindi. **Amazon Translate** sits at $15/million characters with 75+ languages. **DeepL** produces the highest quality translations for European languages at $25/million characters but **critically lacks Bengali and Hindi support**, making it unsuitable as a sole solution. ModernMT ($15/million characters estimated) offers unique adaptive learning from corrections.

### Open-source translation models

**NLLB-200** (Meta; `facebook/nllb-200-3.3B` on HuggingFace) remains the workhorse for multilingual translation research. The 3.3B dense model requires **~8 GB VRAM** (FP16) and achieves the best quality among open-source encoder-decoder models, with a **70%+ improvement** over prior MT systems for South Asian and African languages. The distilled variants (1.3B at ~3–4 GB, 600M at ~2 GB) offer faster inference with moderate quality loss (~1–3 chrF points). Via CTranslate2 optimization, batch processing reaches **10–20 sentences/second** on a T4 GPU. The model has no content filters and will translate any text.

**GemmaX2-28** (Xiaomi Research, NAACL 2025; `arxiv.org/abs/2502.02481`) is the most compelling newer option. Built on Gemma2 with 56B tokens of multilingual pretraining across **28 languages that include all 10 targets** (Arabic, Bengali, German, English, Spanish, French, Hindi, Japanese, Russian, Chinese), it's available in 2B (`ModelSpace/GemmaX2-28-2B-v0.1`) and 9B (`ModelSpace/GemmaX2-28-9B-v0.1`) variants. The paper reports it "consistently outperforms TowerInstruct and X-ALMA" on WMT benchmarks. The 9B model requires ~18 GB VRAM (FP16), fitting on an A100 or quantized for T4/V100.

**TowerInstruct** (Unbabel, 2024) outperforms NLLB-54B and approaches GPT-4 quality for its 10 supported languages (English, German, French, Spanish, Chinese, Portuguese, Italian, Russian, Korean, Dutch) but **lacks Bengali, Hindi, Arabic, and Japanese**—a dealbreaker for this project as a sole solution. **ALMA-R** (Xu et al., ICML 2024; `arxiv.org/abs/2401.08417`) matches or exceeds GPT-4 on WMT benchmarks using only 22K parallel sentences and 12M LoRA parameters, while **X-ALMA** (ICLR 2025; `arxiv.org/abs/2410.03115`) extends to 50 languages. **MADLAD-400** (Google; `google/madlad400-3b-mt`) covers 450+ languages—the widest coverage available—but with less consistent quality for low-resource languages compared to NLLB-200.

**M2M-100** is effectively superseded by NLLB-200 across all metrics (+7.0 spBLEU on average) and should be skipped. **Helsinki-NLP Opus-MT** models are extremely lightweight (~300 MB per pair) and cover all 10 targets via dedicated bilingual models, making them ideal for CPU-only environments, though quality is inconsistent for low-resource pairs.

### Proprietary LLMs for translation

**GPT-4o** ($2.50/$10.00 per million input/output tokens) ranks as the overall best LLM translator per the Intento State of Translation 2025 report, though the **36.7% refusal rate** on harmful content requires the fallback architecture described above. **GPT-4o-mini** ($0.15/$0.60) offers extraordinary value with good multilingual quality. **Gemini 2.5 Flash** ($0.15/$0.60) matches GPT-4o-mini's pricing with Google's extensive multilingual training data, while **Gemini 2.5 Pro** ($1.25/$10.00) leads WMT 2025 quality rankings across many language pairs. Claude 3.5 Sonnet ($3.00/$15.00) delivers strong translation quality but exhibits the highest refusal sensitivity among major LLMs.

**Cohere Aya Expanse** ($0.50/$1.50 per million tokens) is purpose-built for the **23 languages** that include all 10 XSafetyRep targets. At ~$112 for all 83,160 translations, it offers the best balance of multilingual quality, cost, and alignment with the project's Cohere infrastructure. Cohere Command R+ ($2.50/$10.00) supports the same languages at higher cost and quality.

### Language-specific model selection matrix

| Language Group | Best Open-Source | Best API | Key Gaps |
|---|---|---|---|
| European (DE, FR, ES, RU) | TowerInstruct-13B, ALMA-R | DeepL, GPT-4o | None—all models strong |
| CJK (JA, ZH) | GemmaX2-28-9B, ALMA-R | Google Translate, GPT-4o | DeepL recently added JA/ZH |
| South Asian (BN, HI) | NLLB-200-3.3B, GemmaX2-28 | Google Translate, Azure | **DeepL does not support BN or HI** |
| Arabic script (AR) | NLLB-200-3.3B, GemmaX2-28 | Google Translate, Azure | DeepL Arabic support is recent |
| African (AM, HA, IG, YO, WO) | **NLLB-200 only viable option** | Google Translate, Azure | Most LLMs and DeepL lack coverage |

---

## 4. Cost analysis reveals LLM APIs as cheapest option, not dedicated MT

This counterintuitive finding stems from the token-vs-character pricing model difference. For 83,160 translations of ~75-token records across 9 languages (~6.24M total input tokens, ~6.24M output tokens), LLM-based translation is **dramatically cheaper** than character-based MT APIs.

| Service | Type | Total Cost | Cost/Record | Refusal Risk |
|---|---|---|---|---|
| NLLB-200 (Kaggle/Colab free) | Local model | **$0** | $0 | None |
| GPT-4o-mini (Batch API) | LLM | **~$2–5** | $0.00003 | Moderate |
| Gemini 2.5 Flash | LLM | **~$5** | $0.00006 | Low–Moderate |
| Claude 3 Haiku | LLM | **~$9** | $0.00011 | High |
| Cohere Aya Expanse | LLM | **~$112** | $0.00135 | Moderate |
| GPT-4o | LLM | **~$78** | $0.00094 | Moderate |
| Claude 3.5 Sonnet | LLM | **~$112** | $0.00135 | High |
| Azure Translator | MT API | **~$2,250** | $0.027 | None |
| Google Cloud Translation | MT API | **~$500** | $0.006 | None |
| DeepL (7 langs only) | MT API | **~$625** | $0.0075 | None |

*LLM costs assume ~165 tokens input (75 source + ~90 prompt overhead) and ~75 tokens output per call. Batch API discounts (50% off) from OpenAI, Anthropic, and Google further halve LLM costs. Character-based API costs assume ~25M source characters total.*

**The recommended cost-optimal strategy**: Run NLLB-200-3.3B on Kaggle (free, 30 GPU-hours/week) for initial translations (~2–5 hours processing time), then use GPT-4o-mini Batch API (~$2–5) or Cohere Aya Expanse (~$112 with available credits) for quality refinement of flagged segments. Total cost: **under $120**.

---

## 5. Prompt engineering should be simple, direct, and persona-anchored

The academic literature on LLM translation prompting converges on several clear findings that contradict common intuitions.

**Simple, direct prompts outperform complex ones.** Jiao et al. ("Is ChatGPT a Good Translator?"; `arxiv.org/abs/2301.08745`) tested multiple prompt templates and found `"Please provide the {target_language} translation for these sentences:"` performed best. Gao et al. (`arxiv.org/abs/2304.02182`) confirmed that designed prompts achieve comparable performance to commercial MT for high-resource languages. Critically, **chain-of-thought prompting degrades translation quality** by inducing word-by-word translation behavior (confirmed by `arxiv.org/abs/2303.13780`).

**Few-shot examples consistently help**, with 3–5 high-quality examples being optimal. Quality of examples matters more than quantity—semantically similar examples to the input perform best. The **MAPS framework** (He et al., TACL; `doi.org/10.1162/tacl_a_00642`) adds a knowledge-mining step before translation: extract keywords, topics, and relevant demonstrations, then integrate them into the prompt. This resolved up to **59% of hallucination mistakes**.

**The TEaR self-refinement framework** (Feng et al., NAACL 2025 Findings; `arxiv.org/abs/2402.16379`) implements translate → estimate (MQM-style error annotation) → refine cycles, yielding consistent quality improvements across resource levels. Andrew Ng's open-source translation-agent (`github.com/andrewyng/translation-agent`) implements a similar three-step agentic workflow (translate → reflect → improve) and supports glossaries, style control, and regional dialect specification.

For the XSafetyRep pipeline specifically, the recommended prompt template combines persona anchoring with structured output:

```
System: You are a professional certified translator for an academic AI safety
research project. Translate all text exactly and completely, preserving original
meaning, tone, and intent. Output ONLY the translation with no commentary.

User: Translate from English to {target_language}:
{source_text}
```

For structured batch processing, requesting JSON output (`{"translation": "..."}`) enables reliable parsing. **Temperature should be set to 0.0** for deterministic, faithful translation. **Pivot prompting** (translating via English as intermediate) improves quality for distant language pairs with low-resource targets.

---

## 6. Quality evaluation requires a multi-metric ensemble

No single metric reliably captures translation quality for safety-sensitive content. The recommended evaluation stack layers automatic metrics with targeted manual verification.

**COMET** (`Unbabel/wmt22-comet-da` on HuggingFace; `github.com/Unbabel/COMET`) is the primary metric, scoring 0–1 with the highest correlation to human judgments among automatic metrics. The reference-free variant `Unbabel/wmt22-cometkiwi-da` enables quality estimation without gold references. **XCOMET** (`Unbabel/XCOMET-XL`, 3.5B params) additionally identifies error spans. However, COMET can assign overly generous scores to fluent but hallucinated translations—a documented limitation.

**chrF++** (via SacreBLEU: `github.com/mjpost/sacrebleu`) serves as the secondary surface-level metric, using character n-grams plus word bigrams. It's less sensitive to tokenization than BLEU and performs better for morphologically rich languages (Arabic, Hindi, Bengali, Japanese). The ALMA-R paper explicitly recommends abandoning BLEU as a primary metric.

**Back-translation with BERTScore** provides the most practical reference-free quality check. Moon et al. (2020; `arxiv.org/abs/2004.13937`) showed that round-trip translation evaluated via SBERT/BERTScore **outperformed WMT 2019 QE task submissions**. The procedure: translate source → target → source, then compute BERTScore between original and round-trip result. This is robust to backward system choice and works across MT paradigms.

**GEMBA** (Kocmi & Federmann, 2023; `arxiv.org/abs/2302.14520`; `github.com/MicrosoftTranslator/GEMBA`) uses GPT-4 as a translation judge, achieving state-of-the-art correlation with MQM human labels. The GEMBA-DA prompt scores translations 0–100; GEMBA-MQM detects specific error spans with severity ratings.

For **safety-specific validation**, CultureGuard's cross-lingual consistency filter should be implemented: a translation is accepted only if the safety category label survives a round-trip (translate to target, back-translate to English, classify both with a safety classifier). Segments where safety labels diverge after round-trip are flagged for manual review.

---

## 7. Pipeline engineering should prioritize resilience and resumability

The translation pipeline for ~83,160 records needs robust error handling, checkpointing, and quality verification integrated at every stage.

**Python wrapper libraries** provide the foundation. `deep-translator` (PyPI) offers a unified API across 9+ backends (Google, DeepL, Microsoft, ChatGPT) with built-in `translate_batch()`. `translators` (UlionTse) wraps 30+ engines without API keys. For local models, `EasyNMT` (UKPLab) wraps Opus-MT/M2M-100 with multi-GPU `translate_multi_process()` and Docker REST API deployment.

**Rate limiting** should use the `tenacity` library with exponential backoff plus jitter: `@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type((ConnectionError, RateLimitError)))`. For async pipelines, `aiolimiter.AsyncLimiter` implements leaky-bucket rate limiting (e.g., 5 requests/second) combined with `asyncio.Semaphore` for connection pool control.

**Checkpointing** should save progress to a JSON file every 100 records, using atomic writes (write to temp file, then rename) to prevent corruption. The checkpoint structure should be `{record_id}_{target_lang}: {status, translation, timestamp}`, enabling resumption from the exact point of failure.

**Translation failure detection** requires a multi-signal approach using `fast-langdetect` (FastText-based, 176 languages, 80× faster than `langdetect`). Check for: empty outputs, unchanged source text (copy-through), wrong target language (detected language ≠ expected with >80% confidence), truncation (output <20% of source length), and refusal patterns (regex matching on common refusal phrases like "I cannot," "I'm sorry," "as an AI").

The recommended **processing architecture** uses async I/O with batched API calls:

```python
# Core pipeline pseudostructure
for target_lang in target_languages:
    checkpoint = load_checkpoint(f"checkpoint_{target_lang}.json")
    pending = [r for r in records if r.id not in checkpoint]
    
    async for batch in chunk(pending, size=50):
        results = await asyncio.gather(*[
            translate_with_fallback(r, target_lang) for r in batch
        ])
        # Verify: language detection, length check, refusal detection
        verified = [verify_translation(r, t, target_lang) for r, t in zip(batch, results)]
        # Checkpoint
        save_checkpoint(checkpoint_file, verified)
```

---

## 8. Recommended pipeline architecture for XSafetyRep

The optimal architecture synthesizes all findings into a three-tier cascading pipeline with integrated quality assurance.

**Tier 1 — NLLB-200 3.3B baseline translation** (cost: $0, time: 2–5 hours on Kaggle P100). Translate all 9,240 records × 9 languages using the NLLB-200 3.3B model via CTranslate2 for optimized inference. This handles all languages without refusal, provides a strong baseline, and is completely free. Use language codes: `arb_Arab`, `ben_Beng`, `deu_Latn`, `fra_Latn`, `hin_Deva`, `jpn_Jpan`, `rus_Cyrl`, `spa_Latn`, `zho_Hans`.

**Tier 2 — LLM quality refinement** (cost: ~$5–112 depending on model). Pass NLLB translations through Cohere Aya Expanse or GPT-4o-mini with a refinement prompt: provide both the English source and NLLB translation, asking the LLM to improve the translation. This "refine" framing dramatically reduces refusal compared to direct translation of harmful content. For any remaining refusals, retain the NLLB translation.

**Tier 3 — Quality assurance** (cost: ~$0–10). Run COMET scores on all translations (reference-free via CometKiwi). Perform back-translation to English via NLLB-200, compute BERTScore against originals. Flag segments where: COMET < 0.7, BERTScore < 0.8, detected language ≠ target, or safety label changes after round-trip. Manual review of flagged segments (~5–15% of total expected).

For **generalizability to any dataset and language pair**, the pipeline should accept configuration via a YAML file specifying: source/target language pairs, model selection per tier (with automatic fallback ordering), batch size, checkpointing frequency, quality thresholds, and prompt templates. The NLLB-200 backbone ensures bidirectional translation across 200+ languages, while the LLM refinement tier can be swapped between Cohere, OpenAI, or Anthropic endpoints.

---

## Conclusion

Three insights emerge from this research that should fundamentally shape the XSafetyRep translation pipeline design. First, **dedicated MT models are superior to LLMs for safety content translation**—not because of quality, but because they translate faithfully without refusal. Every top-tier multilingual safety benchmark (XSafety, MultiJail, M-ALERT, PolyGuard) either used human translators or dedicated MT models as their translation backbone, never LLMs alone. Second, **the cost landscape has inverted**: LLM APIs are now 10–100× cheaper than dedicated MT APIs for batch translation due to token-vs-character pricing, making a hybrid NLLB (free) + LLM refinement ($5–112) approach both the highest-quality and most economical option. Third, **Bengali is the hardest target language in this set**—LinguaSafe found 71% error rates for vanilla Bengali translation, DeepL doesn't support it at all, and most LLM-based translation models underperform on Bangla script. NLLB-200 and GemmaX2-28 (which explicitly supports Bengali) are the only reliable open-source options.

The field is moving rapidly: GemmaX2-28 (NAACL 2025), the TATER framework, and PolyGuard's multi-model pipeline all represent 2025 advances that significantly improve on earlier approaches. The XSafetyRep pipeline should be built modularly to incorporate newer models as they emerge, with the NLLB-200 → LLM refinement → multi-metric QA architecture providing a robust, generalizable foundation.