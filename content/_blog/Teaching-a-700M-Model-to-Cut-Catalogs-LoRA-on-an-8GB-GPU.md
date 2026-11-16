---
title: "Teaching a 700M Model to Cut Catalogs: LoRA Ablations on an 8 GB GPU"
date: 2026-11-16
description: "LoRA-tuning small Liquid models to split library catalog text into exact database units: what moved exact match from 0 to 82 out of 93, and why rank, context, and target modules mattered less than the data."
---

In the [manuscript extraction post](/blog/From-Manuscript-Transcriptions-to-JSON-A-Configurable-Extraction-Pipeline/) I argued that fine-tuning should not be step one. For one narrow sub-problem of the library migration it turned out to be exactly the right step, and a small adapter on a consumer GPU ended up doing most of the work.

The sub-problem: a catalog field holds a run of literature references glued together, and the database wants one row per reference. The model has to return **exact, in-order substrings** of the input, one per line. No paraphrasing, no normalizing, no "helpful" cleanup. Out of the box, both models I tried scored **0/93 exact**. They knew what a citation looks like, they just couldn't stop rewriting it.

---

## Setup

- **Models:** `LiquidAI/LFM2-700M` locally, a 1.7B Liquid model on the cluster's L40 nodes.
- **Hardware:** RTX 3060 Ti (8 GB) at home, L40 (48 GB) on the HPC side. Code and data sync over Tailscale with a small `rsync` script that skips adapters and caches (about 6 GB down to 5 MB).
- **Training:** PEFT `LoraConfig` + TRL `SFTTrainer`, bf16, gradient checkpointing, batch 1 with gradient accumulation 8, learning rate 2e-4. Each example is a chat pair: catalog text in, newline-joined spans out.
- **Data:** about 840 hand-checked clean rows, an OCR-noisy "dirty" twin of the same rows, and 131 reconstructed long catalogs with many spans each.
- **Eval:** a fixed 93-row clean held-out set. **Exact** means the whole predicted list matches. I also track bag-of-spans F1 and break results down by how many spans a row has.

Experiments are TOML configs (datasets with their own train/eval fractions, an oversample block, LoRA knobs), so every row in the tables below maps to one file.

---

## The Ablation Ladder

The first series trained on the dirty data, changing one thing per run:

| Run | Change | Exact (dirty gold) | Span F1 |
| :--- | :--- | :--- | :--- |
| 1 | baseline: r8, 512 tokens, ~3 epochs | 31/93 | 0.57 |
| 2 | context 1536 | 33/93 | 0.59 |
| 3 | rank 16 (α=32) | 37/93 | 0.63 |
| 4 | 5 epochs, hard rows oversampled | 41/93 | 0.68 |
| 5 | adapt MLP too (`up`/`down`/`gate`) | 43/93 | 0.62 |

Every lever helped a little, and the gains came almost entirely from rows with 1 to 4 spans. Rows with 5 to 7 spans went from 2/32 to 7/32, and rows with 8 or more stayed at **0/6** no matter what I turned. Run 5 is the one that bugged me: exact went up, F1 went down. More adapted layers made the model more willing to emit spans, including spurious ones, so precision dropped. Capacity was saturating and starting to buy the wrong behavior.

Then I re-scored the same adapters on a clean version of the gold, and retrained the run-5 recipe on clean data:

| Run | Exact (clean gold) |
| :--- | :--- |
| No adapter (700M) | 0/93 |
| Run 5 recipe, trained on dirty | 56/93 |
| Run 5 recipe, trained on clean | **71/93** |
| Same recipe, 1.7B model | 81/93 |
| Same recipe, 1.7B, 4096 context | 75/93 |
| 700M, clean + reconstructed long catalogs, 4096 context | **82/93** |

Turns out the biggest single jump in the whole project (+15) came from changing nothing about the model. Training on OCR noise taught the adapter to reproduce noise patterns that the clean eval punished. Rank, context length and target modules together had bought +12 on the dirty series.

The last row is the result I like most. The 700M model on the 8 GB card matched and slightly beat the 1.7B run, once the training mix contained enough long catalogs (hard rows oversampled 3x, 2406 training rows total). It also scored the first nonzero long-tail result: **2/6** on 8+ span rows, and 29/32 on 5 to 7. Data shape moved the plateau that parameters couldn't.

---

## What the Adapter Is Actually Doing

To make sure I understood what I was configuring, I wrote a toy `LoRALinear` next to the real pipeline, loosely following the torchtune LoRA tutorial. The forward pass is the whole idea:

```python
frozen_out = self.linear(x)
lora_out = self.lora_b(self.lora_a(self.dropout(x)))
return frozen_out + (self.alpha / self.rank) * lora_out
```

The base weight stays frozen, and only the low-rank pair `A` (in × r) and `B` (r × out) trains. PEFT initializes `B` to zero, so step 0 is exactly the base model. With r=16 and α=32 the update is scaled by 2, which is why going from r8 to r16 at fixed α isn't a pure "more capacity" change: it also changes the effective step size of the adapter. I didn't separate those two effects, honestly, and I should have.

For this task the adapter mostly learns two things the base model already half-knows: output format (one span per line, nothing else) and boundary decisions (where a title ends and the next author begins). That is a good fit for a low-rank update. The long-tail failures look different: on 8+ span rows the model drops a prefix or glues two titles, which reads more like a long-range copying problem than a knowledge gap.

Memory on 8 GB was manageable once gradient checkpointing was on. The 700M base in bf16 is about 1.4 GB of weights, the r16 adapter over all seven projections is tiny next to that, and activations at 4096 tokens are what actually decides whether a run fits.

---

## What This Connects To

The setup is one frozen base and many small task adapters. Each adapter is a few megabytes, trains overnight on a consumer card, and can be swapped per workload. That is a different cost model from full fine-tuning, and it raises the next systems question: what does it take to serve many of these at once, or to keep updating one as new catalog conventions show up?

It also changed how I think about small models. A 700M model that fits comfortably on an 8 GB card was competitive with a model more than twice its size once the data was right. That matters if this kind of extraction ever has to run close to the scanner instead of on a cluster.

Things I haven't measured yet and would check next:

- **One seed per config.** On 93 rows, one example is about 1.1 points. Runs 2 and 3 might be inside the noise.
- **Why 4096 context hurt the 1.7B run** (81 to 75). My guess is that longer packed sequences changed the effective number of updates per epoch, but I haven't verified it.
- **Training throughput.** I have quality numbers but no tokens/s or peak-memory table per config. Profiling the train step is the next artifact.
- **Constrained decoding** to force every output line to be a substring of the input. That would turn the exact-copy requirement into a guarantee instead of something the adapter has to learn.
- **The visual half.** A sibling project LoRA-tunes a small VLM to draw one box per literature entry on a page image, instead of the merged multi-line blocks YOLO layout detection gives me. It's scaffolded, no results yet.

---

## Takeaways

1. **Fix the data before scaling the adapter.** Clean training data was worth +15 exact. Rank, context and target modules together were worth +12.
2. **Bucket the eval by difficulty.** The aggregate score hid that 8+ span rows were stuck at zero for five runs straight.
3. **Watch precision when adding target modules.** Adapting the MLP raised exact but lowered F1 because the model over-split.
4. **Oversample the failure distribution.** Hard rows at 3x, plus a dedicated set of long catalogs, was the only change that moved the long tail.
5. **Try the small model with the right data before the big one.** 700M on 8 GB reached 82/93, against 81/93 for 1.7B on an L40.
6. **Keep every experiment as a config file.** Ablations stay comparable, and the same run moves from the home GPU to the cluster without edits.

---

### References

* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* Hugging Face [PEFT](https://github.com/huggingface/peft) (`LoraConfig`) and [TRL](https://huggingface.co/docs/trl/sft_trainer) (`SFTTrainer`).
* [torchtune LoRA fine-tuning tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/lora_finetune.html) (reference for the toy `LoRALinear`).
* Liquid AI, [LFM2 models](https://huggingface.co/LiquidAI) on Hugging Face.
* Chu et al., *SFT Memorizes, RL Generalizes* (2025). [arXiv:2501.17161](https://arxiv.org/abs/2501.17161)
