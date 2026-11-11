# The MTP draft head: what it keeps today, and a self-distillation design (owner note 179 item 3)

2026-11-04, run 176 (agent). Part of the Intel port. A measurement and a design only: **no training until the owner
says go.** The draft head never changes the served text (the verify keeps only what the model itself would emit),
so every change here is judged by speed alone.

## 1. What the head keeps today (H121: image ri-m3rtneo:221, the rotation, 10 minutes, 225 requests)

- Depth 3 (three drafts a verify step). 12,259 verify steps, 36,777 drafts, 25,778 kept: **0.701 of the drafts,
  3.10 tokens a step** (1 + 2.10 kept).
- **By draft position** (the engine's /metrics: a position counts when every earlier one was kept):

  | position | reached | kept | conditional acceptance |
  |---|---|---|---|
  | 1st draft | 12,259 | 10,304 | 0.841 |
  | 2nd draft | 10,304 | 8,592 | 0.834 |
  | 3rd draft | 8,592 | 6,882 | 0.801 |

- **By prompt family** (the rotation's prompts by their token count and finish; m3/draft_by_family.py over the
  new `drafts= kept=` field of each request's log line, 8b14a2d):

  | prompt tokens, finish | requests | kept share | tokens a step |
  |---|---|---|---|
  | 6, stop | 21 | 0.817 | 3.45 |
  | 7, length | 21 | 0.673 | 3.02 |
  | 13, length | 21 | **0.526** | **2.58** |
  | 16, length | 21 | 0.798 | 3.39 |
  | 63, stop | 21 | 0.727 | 3.18 |
  | 66, stop | 40 | 0.768 | 3.30 |
  | 151, length | 20 | 0.647 | 2.94 |
  | 348, tool_calls | 20 | 0.671 | 3.01 |
  | 454, tool_calls | 20 | 0.737 | 3.21 |
  | 462, tool_calls | 20 | **0.888** | **3.66** |

  Free-running prose (the length-capped stories) keeps least; structured output (tool calls, short answers) most.
- **What a gain is worth:** the verify step is ~36.4 ms (H121's decode-step lines) and nearly flat in the tokens
  kept, so tok/s scales with tokens a step: **+0.1 token a step = +2.7 tok/s (+3.2 %)** at today's 85 tok/s
  aggregate; every draft kept (4.0 a step) would be +29 %. Moving the 1st-position acceptance from 0.84 to 0.90
  alone (the later ones unchanged in ratio) gives ~3.30 tokens a step: **~+6.5 %**.

## 2. Two things to try, the cheap one first

1. **The head's own vocabulary projection at higher precision (no training, ~1 run).** The served container stores
   `mtp.draft_head.weight` (the head's 248,320-row output projection) at 2 bits (`rtn codes=u2 zero=u8 group=128
   scale=f16`, 79.5 MB a card); the trunk's own `output.weight` is int8. Quantisation noise in the draft's
   logits costs acceptance where the top two candidates are close -- the prose families above. A re-convert of
   that one tensor at int4 or int8 (+80-240 MB of VRAM, out of the expert slab), or sharing the trunk's int8
   output projection if the head was tied to it in training, is measured by one rotation's acceptance. Cost: one
   tensor's convert (`--reuse` copies everything else) and a gemv form for the new dtype if libxe lacks one.
2. **Self-distillation of the head on Flash-Next's own outputs (training; the owner's go needed).**

## 3. The self-distillation design

- **What the head reads and predicts.** For position t the MTP head takes the trunk's final hidden state h_t and
  the embedding of the token at t + 1, and predicts the token at t + 2. Self-distillation trains it to match the
  **served model's own** distribution at t + 2 (the trunk's logits there), not the original training data, so
  the drafts follow what the verify will accept -- including the quantised model's quirks.
- **The training pairs, from our engine on intel-host (no new compute path):** the KL mode already teacher-forces a
  corpus through prefill and records every position's logits (`--kld-record`). Two additions: (a) dump the
  trunk's final hidden rows (2,560 bf16 = 5 KiB a position) beside them; (b) keep the logits as top-32 (ids +
  probabilities, ~256 B a position) instead of the full 248K row (~1 MB) -- the full rows for 1M positions would be
  ~1 TB. Corpus: ~/code/lora-data/v2's Flash-Next replies plus the agreement / rotation prompts' completions
  (prose, code, tool calls), ~0.5-1M positions: **~3-6 GB of pairs**, ~1-2 hours of the cards at prefill speed.
- **Where to train:** amd-host cards 2,3 with the LoRA stack, booked with that session in ~/code/SCHEDULE.md. The
  head is small: its fc layers, one transformer block and the 248K x 2,560 output projection, ~0.65 B parameters
  if that projection is trained, ~0.03 B without it. **A full fine-tune of the head without the output projection
  (or a LoRA on the block and an adapter on the projection) fits one card;** the loss is the KL between the head's
  distribution and the recorded top-32 (renormalised), at the draft depth's three positions (the head reads its
  own previous draft for positions 2-3, so those pairs come from running the head itself -- a second dump pass, or
  train position 1 first and measure).
- **Back into the .rad:** the trained tensors as a bf16 safetensors file; `rad-convert --reuse` of today's container
  with the recipe's `mtp.*` rules pointed at the new file (the head re-quantised as today, or at int4 / int8 if
  item 2.1 shows the precision matters); the rest copied. No engine change.
- **Expected gain:** published self-distillation of draft models and heads (EAGLE's training on target-generated
  data, Medusa-2's self-distillation, the MTP heads of the DeepSeek-V3 / Qwen3-Next line fine-tuned on in-domain
  outputs) reports acceptance-length gains of roughly 5-20 %; I could not verify a figure for Flash-Next's head
  from here, so the honest range for our 3.10 tokens a step is **+0.1 to +0.4 tokens (+3 to +13 % tok/s)**, the
  upper end only for prose-heavy traffic. The cheap item 2.1 may take part of it first.
- **Cost and time:** the dump ~2 runs (~$120: the hidden-row hook, the top-k writer, the dump on intel-host), training
  ~half a day of amd-host cards 2,3 (the LoRA session's slot), the convert + copy ~2 hours, the measurement one
  rotation. Disk: ~6 GB of pairs (/srv/bulk/ri), one 122 GB container on intel-host's store.
- **Gate:** speed only -- the rotation's median and aggregate against H121's 89.2 / 84.7, the M-pair's texts
  identical (the head cannot change them; identical texts confirm it).
- **Licensing:** a retrained head is a modified Flash-Next weight, distributed (if ever) under the same Qwen
  Community License as the model today; nothing here is published, and the owner decides any use beyond intel-host.
