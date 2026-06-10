---
title: "From Manuscript Transcriptions to JSON: A Configurable Extraction Pipeline"
date: 2026-07-15
description: "Initial notes on a series about porting unstructured university library data to a new metadata backend: schema-constrained extraction from noisy transcriptions, early vision-vs-text experiments, and what prefilling buys you on the local HPC cluster."
---

This post opens a series I plan to write while porting a large body of unstructured university library data into a new backend metadata editing system. The migration will not be a one-shot ETL script: legacy catalog records need to go through the new system's API, and the payloads have to be extracted and validated before anything can be ingested. I expect batch inference, fine-tuning experiments, and evaluation to run on the local university HPC cluster, since hosted APIs are off the table.

The first hard problem looks like manuscript transcriptions spanning material from the **7th century onward**: noisy historical text with complex Latin nomenclature, abbreviations, and transcription styles that shift across centuries and catalogers. All of it needs to become valid JSON against a fixed schema. This post is my initial thinking on how to approach that extraction problem, plus a few early experiments that already changed the direction. Later posts in the series will cover implementation, API integration, and what actually worked at scale.

---

## The Setup

Most information-extraction advice assumes thousands of labeled documents and a benchmark to optimize against. The setup I am walking into looks different: a **small validation set** (roughly 100 to 1000 examples), a **fixed JSON schema**, transcriptions of manuscripts dated from the 700s through the early modern period, and on-premise inference on the cluster.

A typical catalog entry is not clean prose. Eight centuries of cataloging practice show up in one block: Latin titles and incipits, scholastic and paleographic abbreviations, watermark cross-references, binding notes, uncertain readings, and transcription conventions that differ depending on when and by whom the entry was written:

```text
M.ch.f. 124

Thomas Aquinas: Summae theologicae pars I

Pergament u. Papier 322 Bl. 307 x210 mm Ostfranken oder Leipzig
1463

Bl. 1 u. 11 sind aus Pergament. Wasserzeichen: Turm, Piccard II, 337. ...
Bastarda. Rubriziert. 2r: Darstellung des Thomas in Q (deckende Farben. Blattgold).
...
316r: finitus per Io(annem) Lanckheym. 1463. in die saneti Sebastiani.
...
Benützte Ed.: S.Thomae Aquinatis Opera omnia iussu Leonis XIII. ed., t.4-5, Romae 1888-1889.
```

The target is structured metadata: shelfmark, author, title, material, dimensions, dating, script, decoration, collation, binding, colophon, marginalia, cited editions. Valid JSON. Schema-compliant types and enums. Output the archive can ingest without fixing every field by hand.

That is **constrained semantic parsing**: unstructured transcription in, valid JSON out, with very limited supervision.

---

## The Wrong First Question

A natural starting point is *which paper should I read to fine-tune on my validation set?*

In the small-data, fixed-schema, noisy-text regime, several lines of work matter more than another NER benchmark:

| Paper / line of work | Why it matters |
| :--- | :--- |
| SFT memorization & generalization (2024/25) | Whether a small supervised set generalizes or just memorizes |
| Outlines / grammar-constrained decoding | Guarantees syntactically valid JSON at generation time |
| DSPy (2024) | Optimizes prompts and pipeline modules against a validation metric |
| Structured outputs / Instructor-style APIs | Typed objects internally, JSON at the boundary |
| Inference-time scaling / test-time compute | More decode-time compute on hard cases instead of retraining |
| DocIE / KIE benchmarks | Evaluation methodology, less direct implementation value |

**Read first:** SFT memorization and generalization work. The core question is whether tuning on a few hundred JSON examples will hold up on unseen documents.

**Implement first:** grammar-constrained decoding. That is probably the single largest practical win for schema-shaped output.

---

## Why Fine-Tuning Should Not Be Step One

In the small-data regime, formatting failures and schema violations are often cheaper to eliminate with decoding constraints than with more gradient steps. Fine-tuning feels like progress when GPUs are available on the cluster. The papers increasingly argue it should come later.

Without constraints, a model can emit broken JSON. With a grammar-backed decoder, illegal tokens cannot be produced. The model may still hallucinate field *values*, but it cannot break surface syntax.

```text
Base instruct model
      │
LoRA (optional, if semantic errors persist)
      │
Grammar-constrained decoding
      │
JSON Schema validator
      │
Repair pass (optional)
```

Fine-tuning belongs in the stack when the model consistently misreads *meaning* (confuses colophon with dating, drops nested physical description) after constraints and prompting are in place. Not when the output forgot a closing brace.

---

## Early Experiments: Vision Models, Schema Size, and Prefilling

Before committing to the on-premise stack above, I tried a few hosted tools that output structured data from documents in a single pass. **Datalab Lift** is one example: a vision model that ingests multi-page inputs and returns JSON in one shot. That whole-document view is genuinely useful when a field spans pages (a colophon on the last leaf, a watermark note that continues on the next catalog card). The catch is that my input is already **raw transcription text**, not scanned images. Feeding text through a vision-oriented pipeline still produced OCR-style misreadings on strings that were already correct. For this migration, a text-native or properly tuned multimodal model looks like the better bet, not a document-vision stack pretending the bytes are pixels.

Single-pass multi-page extraction also has a performance cliff. Without deliberate preprocessing, batching, or chunking, latency and cost climb fast as page count grows. The feature is worth keeping in the design space, but it needs infrastructure around it, not just a bigger context window.

Our target schema is not a flat dozen fields. It has **more than 50 properties**, with nested objects (e.g. `manuscript` → `chapter`). On a **10B** instruct model, results were promising but I suspect the model would do better if I chunked the extraction problem instead of asking for the full tree at once. That fits the CMR idea below: decompose before you map to the final API payload.

One cheap win showed up before any fine-tuning. **Enriching the JSON Schema itself** helped: filling in `description` fields on each target property with catalog-specific context (what "colophon" means in this archive, how dating ambiguity is encoded). Prompting with that richer schema improved field quality noticeably, with no weight updates. The schema is not just validation glue; it is part of the prompt.

The strongest result so far, across every pipeline variant I tried: **prefill the output JSON as far as regex and heuristics allow**, then let the model complete the remainder. Shelfmarks, dimensions, leaf counts, four-digit years, and similar patterns are high-precision from rules. Starting generation from a partially filled document beats blank-slate extraction every time. I would build around that regardless of whether the backend is a 10B local model, grammar-constrained decoding, or a future fine-tune.

---

## Pipeline Shape

The manuscript case should generalize across the library migration: raw text will need to map to a predefined schema across different record types, input formats, and nesting depths. The useful literature treats extraction as a **pipeline**, not a single prompt.

For historical manuscripts, the LLM only needs to appear in a few stages:

```text
OCR / raw text
    → cleaning & segmentation
    → section detection
    → entity extraction        ← LLM
    → canonical object
    → normalization & validation
    → target schema mapping
    → repair (optional)        ← LLM
```

Regexes, layout heuristics, and lookup tables handle surprising amounts of catalog boilerplate (`Bl.`, `mm`, `Bastarda`, four-digit years). The model is for semantic glue, not every token.

**Do not extract directly into the target schema.** Use a canonical metadata representation (CMR) and a mapper:

```text
Raw document → LLM / rules → CMR → mapping layer → domain JSON schema
```

```json
{
  "works": [{ "author": "Thomas Aquinas", "title": "Summa theologicae", "part": "pars I" }],
  "physical_description": {
    "materials": ["Pergament", "Papier"],
    "extent": "322 Bl.",
    "dimensions_mm": { "height": 307, "width": 210 },
    "script": "Bastarda"
  },
  "dating": { "year": 1463, "place": "Ostfranken oder Leipzig" }
}
```

One backend schema might expose `Shelfmark`, `Author`, `Title`, `Dating`; another wants `InventoryID`, `Creator`, `WorkName`, `Century`. Only the mapper changes. The extraction engine, constrained decoder, and evaluation harness stay put. That indirection seems worth building in early, since the new system's API schema may still shift as the port proceeds.

---

## Papers I'm Starting From

**DSPy** (Khattab et al., 2024): define a signature (`catalog entry → DocumentMetadata`), a module, and a metric; the optimizer searches prompts, demonstrations, and decomposition. Change the signature for a different record type; the loop stays.

**Snorkel / data programming** (Ratner et al.): weak labels from heuristics (`.*Bl\.` for leaf count, `\d+ x \d+ mm` for format, `\b14\d{2}\b` for dating) complement a small gold validation set.

**Grammar-constrained decoding** (Outlines, LM Format Enforcer): guarantee valid structure at generation time.

**Structured generation / Instructor-style APIs**: typed objects internally, JSON at the boundary.

**Agent Lightning** (2025): relevant once the pipeline has multiple validation and repair stages to optimize jointly.

Together these describe a platform-shaped architecture:

```text
Raw input (PDF / OCR / XML)
        │
Document Parser → Section Detector
        │
Extraction Engine (LLM + constrained decoding)
        │
Canonical Metadata Object
        │
Validation & Confidence Scoring
        │
Domain-specific Mapper → JSON Schema Output
```

What changes per record type: schema definition, mapping rules, examples. What stays reusable: parser, extraction engine, constrained decoding, evaluation framework.

---

## Benchmarking

Generic NLP benchmarks rarely predict whether a record can enter the production system without manual correction. **Per-stage metrics** make failures diagnosable:

| Stage | Metric |
| :--- | :--- |
| Section detection | accuracy / boundary F1 |
| Field extraction | precision, recall, F1 per field |
| Schema compliance | % records passing JSON Schema |
| End-to-end acceptance | % insertable without manual fix |

The metric that usually matters most is **end-to-end acceptance**: how many outputs the API can consume as-is. For free-text fields, normalized Levenshtein or CER on titles and colophons is more honest than exact match.

---

## Planned Build Order (On-Premise HPC)

No hosted API means picking a strong open-weight instruct model, running batch evaluation and fine-tuning through the cluster scheduler, and keeping experiments reproducible. Early runs on a 10B model and hosted vision tools already suggest prefilling and schema-rich prompting before LoRA. This is the order I plan to try:

1. **Regex prefilling** into a partial JSON skeleton. Measure how much remains for the model.
2. **Baseline** with a careful prompt and enriched JSON Schema descriptions. Measure schema validity and field F1.
3. **Grammar-constrained decoding** against the JSON Schema. Expect a large validity jump.
4. **Chunked extraction** for nested 50+ field schemas before scaling model size.
5. **LoRA fine-tuning** only if semantic errors dominate after steps 2–4.
6. **DSPy or structured prompt search** against the validation metric.
7. **CMR + mapper** before hard-coding domain field names into prompts.
8. **Weak labels** via Snorkel-style rules for high-precision patterns.

Grammar decoding plus modest LoRA looks like a reasonable bet for reliability per GPU hour. Full-model fine-tuning on hundreds of examples carries memorization risk and would fight schema evolution whenever the API payload changes.

---

## Open Questions

* Whether a CMR prompted on one catalog convention generalizes to others (German nineteenth-century descriptions vs English auction catalogs).
* Whether weak labels from regex libraries help more than they hurt on ambiguous readings like `Ostfranken oder Leipzig`.
* How much inference-time compute (reranking, repair passes) can substitute for additional supervised examples on a cluster budget.
* How to chunk a 50+ field nested schema on a 10B model without losing cross-field dependencies (colophon vs dating, chapter vs manuscript).
* Whether multi-page single-pass extraction is worth the batching complexity once preprocessing and cluster scheduling are in place.

These are workflow design questions as much as model questions.

---

## Practical Takeaways (For Now)

These are the bets I am taking into the build, based on the literature review and early experiments so far:
1. **Treat small validation sets as a distinct regime.** Read SFT generalization work before committing GPU weeks to fine-tuning.
2. **Enforce JSON at decode time.** Grammar-constrained generation removes a failure mode prompting alone does not fix.
3. **Separate canonical extraction from target schema mapping.** That is how the pipeline adapts when the backend API schema shifts.
4. **Put the LLM only where rules stop.** Pattern-rich boilerplate belongs in heuristics; colophons and provenance notes belong in the model.
5. **Prefill JSON from regex before any model call.** The best ROI so far; works across pipeline variants.
6. **Treat the schema as prompt material.** Rich `description` fields on properties improved extraction without fine-tuning.
7. **Match model modality to input.** Raw transcription text should not go through a vision pipeline that re-introduces OCR errors.
8. **Chunk large nested schemas.** 50+ fields on a 10B model likely needs decomposition, not one-shot generation.
9. **Optimize for end-to-end acceptance**, not token F1 on a generic leaderboard.
10. **Stack techniques.** LoRA + constrained decoding + validation + optional repair beats any single paper in isolation.

---

### References

* Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* (2024). [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
* Ratner et al., *Snorkel: Rapid Training Data Creation with Weak Supervision* (VLDB 2017). [PDF](https://www.vldb.org/pvldb/vol11/p269-ratner.pdf)
* Willard & Louf, *Efficient Guided Generation for Large Language Models* (Outlines, 2023). [arXiv:2307.09702](https://arxiv.org/abs/2307.09702) · [Code](https://github.com/dottxt-ai/outlines)
* Su et al., *One Embedder, Any Task: Instruction-Finetuned Text Embeddings* (Instructor, 2022). [arXiv:2212.09741](https://arxiv.org/abs/2212.09741)
* Chu et al., *SFT Memorizes, RL Generalizes* (2025). [arXiv:2501.17161](https://arxiv.org/abs/2501.17161)
* Lin et al., *Debunk the Myth of SFT Generalization* (2025). [arXiv:2510.00237](https://arxiv.org/abs/2510.00237)
* Luo et al., *Agent Lightning: Train ANY AI Agents with Reinforcement Learning* (2025). [arXiv:2508.03680](https://arxiv.org/abs/2508.03680)
* Gat, [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) (grammar-constrained decoding).
* [Marvin](https://github.com/prefecthq/marvin), [Guidance](https://github.com/guidance-ai/guidance), [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) (typed / structured generation tooling).
* [DocIE benchmark](https://xllms.github.io/DocIE/) (XLLM ACL 2025); [DocRED dataset](https://arxiv.org/abs/1906.06127) (document-level relation extraction).
* Full list: [All references](/readinglist/).

---
