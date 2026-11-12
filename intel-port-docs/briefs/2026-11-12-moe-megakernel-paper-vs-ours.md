# The fused MoE decode megakernel paper against our expert path (owner note 251)

2026-11-12, run 335 (agent). Part of the Intel port. **Desk only: no build, no card time.** The paper is arXiv 2609.04244,
"An Efficient Fused Mega-kernel for Quantized MoE Decoding" ("MonoMoE"); its code is in FlashInfer
(`csrc/fused_moe/monomoe`). Our numbers come from the kernel scoreboard's current decode table (P110, image 292, MTP on:
**29.3 ms of device time a rank-step**), the expert-layout brief (2026-11-05), the megakernel brief (2026-11-05,
sections 6-8) and STATE.

**Bottom line: no item reaches the 2 % bar, so there is no build proposal.** The paper's gains come from three things:
- removing launch boundaries and padding,
- FP8 weights feeding tensor cores with no decode,
- TMA prefetch.

On the B70 the first is already cheap inside recorded tapes, the second does not exist (no FP8 math on Xe2), and the
third was tried (T1c) without gain. The one transferable idea not done here is folding SwiGLU and the activation
quantiser into gate_up's epilogue. It is worth ~1 %.

## 1. What the paper does

- **Hardware and format:** H200 (132 SMs, 4.8 TB/s). E4M3 FP8 weights with 128 x 128 block scales; activations
  quantised in the kernel at 128-element granularity.
- **Models:** Qwen3.5-35B / 122B, GLM-5.2, DeepSeek-V3.1, each 256 experts with top-8; batch 1-8 decode.
- **One persistent launch a MoE layer:**
  - one CTA per SM, a normal (not cooperative) launch, CUDA-Graph capturable;
  - routing, top-k and metadata inside the same kernel;
  - weight-major work: CTAs own row stripes of an expert's weights and apply them to the whole token tile. Tokens sit
    on the MMA's small dimension, unsorted and unpadded;
  - gate + up, the activation and the quantised cast to down's input stay in registers and SLM, with one global store;
  - down accumulates with `atomicAdd`, then a final weighted reduction;
  - per-input readiness flags, no grid barrier; prefetch depth 2 (TMA), including the next expert's first tiles.
- **Kernel speed against vLLM's grouped GEMM at B 1-8:** 1.02-1.54x. Effective DRAM rises from 10-45 % to 21-58 % of
  peak (Qwen3.5-35B).
- **End-to-end time per output token:** 3-19 % lower.
- **Where the gain comes from:** much of it is the grouped GEMM's padding and preprocessing, 13-27 % of its kernel time
  (5.4 us of 20.4 at B = 1).

## 2. Our decode expert path (P110, per rank-step)

| launch a MoE layer | kernel | us a launch | ms a step | share | floor (608 GB/s) | effective rate |
|---|---|---|---|---|---|---|
| router + top-k + scatter (+ the block quantiser, rqp) | `router_topk_scatter_kernel` (one work-group, 256 GRF) | 15.0 | 0.76 | 2.6 % | ~5 us | -- |
| gate_up | `moe_i4i8_run_kernel<5,16,..>` format C | 107.6 | 5.50 | 18.8 % | 36.8 us (22.4 MB) | 208 GB/s = 34 % |
| SwiGLU + int8 quant of down's input | `gated_quant` (128-column groups, optional Hadamard) | -- | ~0.36 (two entries) | ~1.2 % | -- | -- |
| down | `moe_i4i8_run_kernel<3,2,2,..>` | 57.6 | 2.94 | 10.0 % | 18.4 us (11.2 MB) | 194 GB/s = 32 % |
| the weighted combine | `gather` | -- | 0.26 | 0.9 % | -- | -- |

That is about **5 launches a layer and ~9.8 ms a step (~33 %)**, of which gate_up + down are 8.4 ms at ~3x their byte
floor. Inside a recorded tape a kernel boundary costs **~0.5-0.8 us** (M302's 0.77 us; run 176's 1,504 boundaries a
rank-step at 0.52 us).

## 3. Item by item, ranked by what each would save here

| # | the paper's technique | here | expected gain on our numbers | verdict |
|---|---|---|---|---|
| 1 | **gate + up + SwiGLU + quantised cast in the producer, one store** | **NOT done**: gate_up writes bf16, then `gated_quant` reads it back | ≤ 0.36 ms (the quantiser's launches) + 48 boundaries x ~0.7 us = 0.03 ms → **≤ 0.39 ms, ~1.3 % gross**. gate_up is issue-bound (below), so the added epilogue ALU returns part of it: **~0.2-0.3 ms net, ~1 %** | under the bar; the only transferable idea (section 4) |
| 2 | down fused with the routing-weight scale + scatter-add | NOT done | the combine's 0.26 ms (0.9 %) + a boundary | **rejected: `atomicAdd` makes the sum order nondeterministic**, against the exact-path rule; a deterministic order needs the combine anyway |
| 3 | one persistent launch a layer, work queue / flags | **tried** (megakernel 1a / 1b, upstream's hc merge, the cooperative launch, runs 189-206) | launch boundaries a layer 5 → 1: 4 x 48 x ~0.7 us = **0.13 ms (0.5 %)** | the B70 lost 2.7-4.2x on every fused grid: one SLM / GRF budget sized by the largest phase (8x less residency for the column phase), spills, and ~54 % of thread time waiting on flags (megakernel brief §6.4, §7). Not worth it for 0.5 % |
| 4 | routing and top-k inside the MoE kernel | **partly done**: router GEMV + top-k + scatter + quantiser are one kernel (rqp) | ≤ 1 boundary a layer: **0.03 ms (0.1 %)** | merging it would put a one-work-group top-k ahead of a wide grid, the dependency-wait pattern of §6.4 |
| 5 | weight-major stripes, tokens unsorted and unpadded | **different and fine**: expert runs (rows sorted by expert in the scatter), 1-4 rows, no padding | the paper's biggest win (removing 13-27 % of padding / preprocessing) has no counterpart: **we do not pad** | nothing to take; our per-work-group fixed work (the activation staging, 29 % of a launch by ablation) was attacked by the layout repack T1 / T1b: bit-identical, no faster (96-109 %) |
| 6 | block-scaled FP8 weights on tensor cores | not applicable | -- | Xe2 has no FP8 math (note 131); format C is w4 codes → int8 levels (an SLM table) → DP4A. The decode kernels are **instruction-issue bound** on the 4-bit lookup: the select-chain form was +33 / +35 % slower (M294), the expanded int8 form (E5) needs 2x the bytes (≥ 242 us at our measured 343-370 GB/s) |
| 7 | deeper outstanding loads, prefetch of the next expert's tiles | **tried**: T1c (sub-group block loads, wide scale loads, more loads in flight) | -- | bit-identical, 98-102 % (P102). The gate_up weight stream is already near its floor (ablation abl3: −28 us of a 29 us floor); the rest is lookups, staging and fixed work, not load latency |

The paper's effective rate, ~49 % of peak at B = 4, compares with **~32-34 %** for ours. The gap is item 6 (the 4-bit
decode's instruction stream), not anything a fused launch removes. The route past it is a format with fewer
instructions a weight. Format B (int4 x int8) was dropped by the owner (notes 148 / 150), and E5 doubles the bytes.

## 4. If item 1 were ever built (recorded, not proposed)

- **Exactness is reachable.**
  - `gated_quant_row` takes four bf16-rounded values a lane in 128-column groups and has no row-wide reduction (one wave
    a group; `r4d_fused_quant_fp8.cpp`).
  - A gate_up epilogue that rounds its outputs to bf16 first, then runs the same `silu(g) * u`, the optional width-128
    Hadamard and `quant_group<I8>` in the same order, would give the same codes and scales bit for bit.
- **What it needs on Xe2:**
  - **Pairing:** the work-group must produce matching gate *and* up column blocks of a whole 128-column group. Gate
    column j and up column j sit in different halves of gate_up's output, so the column split changes. Each column's
    dot product is unchanged, so this stays exact.
  - **Lane layout:** gate_up's lane-to-column mapping differs from the quantiser's four-values-a-lane wave, so it
    needs shuffles or an SLM transpose of 128 x rows.
  - **Cost:** about 8-16 KB of SLM at 4 rows, which lowers gate_up's residency a little. gate_up runs ~2.2x the card's
    thread capacity at 4 rows, so it has room.
- **Expected:** ~1 % of the step, with a CPU row test, a card row check, the M-pair and a rotation as gates. It sits
  under note 251's 2 % bar, so it goes on the list for the owner, not to the coordinator as a proposal.

## 5. Sources

- arXiv 2609.04244v1 (read 2026-11-12): Sections 3-4, the B = 1-8 kernel and end-to-end tables, the Qwen3.5-35B
  latency and bandwidth table.
- The paper's own inconsistency: its introduction quotes DeepSeek-V3.1 at −1.2 % to +3.7 %, while Section 4.4 says
  12.8 / 8.3 / 4.7 / 3.5 %.
- briefs/kernel-scoreboard.md: the P110 decode table, rows 1, 3, 8 and 14.
- briefs/2026-11-05-expert-layout.md: the ablations abl1-abl4.
- briefs/2026-11-05-megakernel.md: §6.1-6.4 and §7.
- STATE: M294 (the SEL decode), T1 / T1b / T1c (P102), note 231 (E4 / E5).
