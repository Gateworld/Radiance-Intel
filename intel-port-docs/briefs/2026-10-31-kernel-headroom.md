# Radiance-on-Intel (2x Arc Pro B70, TP2, Qwen3.8-Flash-Next): remaining headroom per hot kernel, with an R9700 comparison

This is a read-only analysis on amd-host, written 2026-10-31. No GPU was touched and nothing was built. Each figure carries a tag:
**M** = measured, read from a file (file:line, or the profile table it comes from);
**C** = computed here from M figures;
**E** = estimate (an assumed rate or model, given with its reasoning).
Every "achievable" figure is **E**. It is the time the kernel would take if it reached the best rate measured for its kernel class on this card. It is a ceiling per kernel, not a forecast, and the gains do not all add up independently.

---

## 0. Method

### 0.1 Decode profiles (tape profiler)
- Each table is cumulative over both ranks. Per rank-step = Δ(ms) / Δ(logits_i8 launches), with one logits launch per rank per step (briefs/2026-10-28-bottleneck.md §1 conventions).
- **I use the difference between the last two tables, not the last table alone.** Every `prof 1` run is the *first* request after the model loads, so the first half carries the heat engine promoting experts. In P47 the expert kernels fall from 172.5/155.6 us (first table) to 135.4/136.8 us (second half). Windows:
  - P47: tables at serve.log:801 → 840, i.e. 174 − 82 = **92 rank-steps**, 4,386.3 ms → **47.68 ms device per rank-step** (C).
  - P44: tables :801 → :840, 92 rank-steps.
  - P36: the last two tables (:944 → :976), **124 rank-steps**, 2,837.1 ms → **22.88 ms per rank-step** (C).
  - The script is scratchpad/prof.py.
- **P47 is the base for MTP on**: image :125, hc level 8, LL all-reduce, 4-row verify. The current service setting (`svcmtpx-gnt0-hc8-rp`) adds two things on top of P47:
  - **gnt0**: the GDN update falls from 74.9 to 33.0 us, from P44 (M, STATE.md:2149). That is −1.52 ms per rank-step (C).
  - **The router pair** (`rp`, 29101e1): 48 fewer bf16 GEMV launches. It measured +1.0 % end to end (M135, STATE.md:1562), so about −0.5 ms (E).
  - **Adjusted device time ≈ 45.7 ms per rank-step (C/E).**
- **Unprofiled step, MTP on.** M152 run 2 = 56.93 tok/s (M, results-m3/card01/M152-out.txt). P47 ran 87 verify steps per rank for 255 decoded tokens, so **2.93 tokens a step** (C). The texts are identical to M97, so acceptance is the same. That gives **~51.5 ms per step** (C), and ~5.8 ms of host, sync and residual misses is not in any kernel row.
- **MTP off.** M150 run 2 = 40.33 tok/s → **24.8 ms per step** (M/C). P36 is level 5 without gnt0. gnt0 at one row is worth +5.2 % (M112 vs M113) ≈ 1.3 ms, i.e. GDN ≈ 22 us a call (E). The router pair is ≈ −0.25 ms (E). Adjusted device ≈ 21.3 ms, plus a ~3.5 ms host gap.
- **moe8** (RADIANCE_XE_MOE_V2R_SMALLK=4) is *not* in the current setting. P44 measured its down projection at 114.8 us vs P47's 136.8 at the verify (M). It costs −1.9 % MTP off (M111, STATE.md:2136-2148).

### 0.2 Prefill profile
- **I use B58 instead of B51.** B58 is the same configuration plus STAGE, which is exactly the "12.2 → 9.3 ms a call" change. Both are "xe launch profile (exit)" tables, and B58's other kernels match B51's to within 0.2 %. B58: 7K 12.94 s, 20K 20.47 s (results-m3/card01/B58/serve.log, M).
- The table covers **both prompts and both ranks**:
  - the 7K prompt is 5,604 new tokens;
  - the "20K" prompt is 16,184 tokens with 5,568 cached, so 10,616 new (B51/serve.log:600,614, M).
  - Per rank = /2.
  - **The 7K share is the token fraction 5,604/16,220 = 0.3455**, for every kernel that is linear in tokens.
- **Attention (attn_split) is apportioned linearly too.** Each query's keys come from its QSA block table, capped at `max_blocks × block_size` (xe_attn.cpp:89-92, read). The indexer budget is 2,048 (audit.md:153). So per-query work is flat once context passes 2,048. This is inferred; a quadratic split would give the 7K only ~12 %. The linear split also matches the wall times better: device time per rank comes out at 9.2 s of 12.94 s for 7K and 17.4 s of 20.47 s for 20K, so the non-device gaps are similar.
- **qsa_score (the indexer) scans the whole context**, so it is apportioned by Σ tokens × context (12 %, E).
- **7K device time per rank ≈ 9.20 s (C) against 12.94 s wall.** The remaining **~3.7 s is not in any kernel**:
  - host and pass boundaries;
  - first-request expert staging (21.5 GiB h2d per rank over the two prompts, B58/serve.log:637,660, M);
  - rank imbalance;
  - profiler stamps.
- Pass structure (C, from launch counts):
  - 7 passes per rank: 4,096 / 1,472 / 36 for the 7K, then 2,624 / 4,096 / 3,840 / 56 for the 20K;
  - 36 GDN layers × 2 ranks × 7 = 504 chunk-scan launches (M);
  - attn_split runs once per 256-query block: 66 a layer-rank (M).

### 0.3 Shapes and bytes per call, per rank (C unless marked)
- **Experts (IQ4R, w4nl64a8h, 4.125 bits a weight; audit.md §6):**
  - gate_up 512 × 2560 → 675,840 B a routed expert; down 2560 × 256 → 337,920 B. The expert unit is 992 KiB (STATE.md:2286-2290, BM2, M).
  - At the verify, the counter (M151, STATE.md:184-188, M) shows 39.4 rows to **26.5 distinct experts a layer**, so 1.48 rows an expert. v2m reads an expert's weights once for all its rows (xe_moe_planes.cpp:289-293, read). That gives **17.9 MB (gate_up) and 8.95 MB (down) a layer**.
  - MTP off: 10 experts, 6.76 / 3.38 MB.
- **Trunk W8A8 int8:** **~1.48 GB a rank-step** over 216 launches (bottleneck brief §1.2 row 1), about 1.49 GB with the MTP layer's extra launches. Activations are ≤ 4 × K bytes a call, negligible. lm_head: 248,320/2 × 2560 × (1 + 2/128) = **322.8 MB**. The 1-row form runs at 565 GB/s, a check on that size.
- **hc mixes (E4M3, n 2560, hc 4 → HN 10240, LR 320):**
  - W_down 320 × 10240 = 3.28 MB plus f32 scales per 128 → **3.38 MB**.
  - W_up 10240 × 320 plus scales per 80 → **3.44 MB**.
  - Together that is the "6.55 MB per connection" (r4d_hc_bf16.cpp:28-30 says "each mixing matrix 6.55 MB ... 1.27 GB per step", which is the bf16 size; at E4M3 it is half). There are 97-99 connections a rank-step, and the matrices are replicated on both ranks.
- **MTP draft head (w2a8):** 248,320/2 × 2560 × 2 bits + scales/zeros ≈ **79.5 MB** a pass, 3 passes a step.
- **GDN state:** about 24 heads per rank × 128 × 128 × f32 = 1.57 MB, read and written, so **~3.3 MB a call** (the brief's figure).
- **All-reduce (decode):** 4 rows × 2560 × bf16 = 20 KB payload, which the LL form doubles to 40 KB on the wire. Bytes are not what binds it.

### 0.4 Reference rates on this card (from briefs/2026-10-28-bottleneck.md, M)
| Class | Best measured rate used as the target |
|---|---|
| Clean GEMV ceiling | 596 GB/s |
| int8 big-shape GEMV (M1) | 530-573 → **550 GB/s** |
| lm_head (same kernel, 1 row) | 555-565 GB/s |
| E4M3 GEMV (round-2 probe at the hc shapes) | 232-242 → **237 GB/s** |
| byte-decoded u4 GEMV (M6) | **343 GB/s** |
| int8 XMX | 177 TOPS (measured) |
| f16/bf16 XMX | 183 TFLOPS (nominal) |
| Two-rank exchange handshake | ~15-19 us (P36 1-row `ar` = 17.7 us, M) |

For latency-bound kernels without a byte rate, the target is a per-launch floor: the profiled tiny-kernel floor is ~3-4 us, and the unprofiled L0 command floor is 0.6 us (brief §1.1).

### 0.5 AMD R9700 side
- **Searched read-only:** ~/code/radiance-src*, ~/code/radiance-test, /srv/models/test/reports. **There is no per-kernel profile of production.**
- **Measured per-kernel figures that exist** (gfx1201; tagged M-AMD):
  - **hc_read kernel A:** 25.4 us for 6.55 MB = 258 GB/s; **kernel B:** 13.0 us = 504 GB/s; the shipped hc_read_a: **546 GB/s**. Source: radiance-src/libr4d/r4d_hc_bf16.hip:349-350, :1059.
  - **hc_up W_up:** 14.9 us for 6.55 MB = **440 GB/s** (r4d_hc_bf16.hip:1035-1036). The bf16 lm head on the same part reaches 631 GB/s (:326).
  - **hc prefill:** E4M3 hc_down PF2 711 → **611 us at a 2048-row chunk** (:2531-2533); hc_read with up_mix_wide **767 us at 2048 rows** (:1771-1772).
  - **Expert gate_up at decode:** fastest dispatch **576 GB/s** (resident, ~19.7 MB a dispatch); **mean 270 GB/s** including link misses (r4d_moe.hip:905-915).
  - **Router selection:** 5.4-8.6 us at a 32-row verify (r4d_moe.hip:2848-2852).
  - **GDN update:** ~57 us fixed + 3.2 us a token at 8 seqs × 8 candidates (24 MB state); 14.44 us for a 12 MB call without the NT hint (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip:66-77, :836-840).
  - **Prefill AR (fused, wht6 3/8-byte wire):** gather 491 us and write 317 us at a 2048-row chunk (r4d_ar_entry.hip:764-768).
  - **ISA ceilings:** fp8 WMMA 412, iu8 407, f16 207 TF/s (r4d_gemm_w4a8_nt_m64.hip:5-16). Measured W4A8 at M = 64: 72-78 TF/s.
- **Whole-step anchors (M-AMD):**
  - **15.28-15.32 ms per decode step**, two cards, production flags (radiance-src-170/docs/MOE-W4.md:41-42);
  - 3.19 tokens a step (2026-10-16-amd-host-radiance-tp2.md:244), which with 218 tok/s gives ~14.6 ms (the caller's anchor);
  - **8K first token 1.38 s** (tp2.md:24), so **~0.97 s for 5,604 tokens** (E, linear). MOE-W4.md's 6,539 tok/s at 26K would give 0.86 s.
- **No measured per-kernel R9700 figure exists for:** the int8 trunk GEMV, lm_head, the expert down projection, the w2a8 draft head, attention/QSA decode, the GDN chunk scan, or prefill expert GEMMs.
- **For kernels without one, the R9700 estimate (E-AMD) is:**
  - the same bytes at ~545 GB/s (FP8/int8 class), with the 4-row verify reading weights once;
  - for prefill, FLOPs on native fp8/int8 WMMA at a realistic ~150-200 TF/s;
  - all scaled so the sums land near the anchors.

---

## 1. Decode, MTP on (P47 warm half, adjusted for gnt0 + rp); step ≈ 51.5 ms, device ≈ 45.7 ms per rank-step

| # | Kernel (P47 name) | calls / rank-step (M/C) | us / call (M) | ms / step (C) | bytes / call (C) | achieved GB/s (C) | best class rate (M ref) | target ms (E) | **saving ms (E)** | R9700 est. ms | B70/R9700 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `gemv_i8a8<1,4,0>` trunk W8A8, 4 rows | 221.8 | 40.9 | **9.07** | 6.7 MB avg (1.49 GB / step) | 164 | 550 GB/s plus ~120 small-launch floors | ~3.3 | **5.8** | ~2.9 (E-AMD: 1.49 GB at 545 + floors) | 3.1x |
| 2 | `moe_planes_v2m<1,4,2,16>` expert down | 48.4 | 136.8 (moe8: 114.8 M, P44) | **6.62** | 8.95 MB | 65 | 343 GB/s u4 (M6) + ~4 us | 1.46 | **5.2** (1.05 of it measured via moe8) | ~0.86 (E-AMD: 8.95 MB at ~500) | 7.7x |
| 3 | `moe_planes_v2m<5,4,2,16>` expert gate_up | 48.4 | 135.4 | **6.55** | 17.9 MB | 132 | 343 GB/s | 2.53 | **4.0** | 1.49 resident (M-AMD 576 GB/s) to 3.2 (M-AMD mean 270 GB/s) | 4.4x / 2.0x |
| 4 | hc down group: `hc_v2_down_rows<4>` 26.2 + gains `hc_v2_down<2>` 9.9 + `hc_v2_norm` 6.9 (106.8 calls) | 98.8 | 43.6 (sum) | **4.30** | 3.38 MB (W_down E4M3) | 78 | 237 GB/s E4M3 probe | 1.41 | **2.9** | ~0.8 (E from M-AMD 546 GB/s, norm fused in kernel A) | 5.4x |
| 5 | `hc_up_mix<4,1,1,1,1>` | 97.8 | 32.5 | **3.18** | 3.44 MB (W_up E4M3) | 106 | 237 GB/s (the 1-row form already reaches 232, P36) | 1.42 | **1.75** | ~0.77 (E from M-AMD 440 GB/s) | 4.1x |
| 6 | `ar_kernel<0,0,4>` LL all-reduce + hc write | 106.8 | 31.3 (first half 37.4) | **3.34** | 40 KB wire | ~1.3 effective | protocol floor 17.7 us (P36 1-row, M) | 1.89 | **1.45** | ~1.2 (E-AMD, ~11 us a call, no figure) | 2.8x |
| 7 | `gemv_w2a8<1>` MTP draft head, 3 passes | 3.0 | 521.4 | **1.56** | 79.5 MB | 152 | 550 GB/s int8 class | 0.43 | **1.13** | ~0.44 (E-AMD) | 3.6x |
| 8 | `gemv_bf16<4>` (router 512×2560, ssm_ab, qsa_qk, …) | 148.2 (~100 after rp) | 13.8 | **2.04** (~1.5 after rp, E) | router 2.6 MB; most others < 0.3 MB | ~65 | 550 GB/s + ~3 us floors | ~0.5 | **~1.0** | ~0.35 (E-AMD: the router is fused with top-k on bf16 WMMA) | ~4x |
| 9 | Elementwise tails: `hc_write` .41, `hc_quant_rows` .41, `gather` .32, `rmsnorm` .26, `quant_act_fp8` .22, `silu_mul` .21, `rope` .21 | ~490 | 3.4-6.2 | **~2.0** | ~0 | — | launch floor (0.6 us unprofiled) / fusion | ~1.0 | **~1.0** | ~0.5 (E-AMD: ar_hc_write and ar_rmsnorm fusions) | ~4x |
| 10 | `attn_split` .50 + `qsa_select` .30 + `qsa_score` .25 + `attn_merge` .20 + small QSA ~.10 | ~75 | 13-33 | **~1.35** | ~0.5 MB | — | latency floor | ~0.6 | **~0.75** | ~0.3 (E-AMD) | ~4.5x |
| 11 | `router_topk_scatter<16,8>` | 51.4 | 19.0 | **0.97** | ~4 KB | — | ~5 us (M-AMD 5.4 us at 32 rows) | 0.25 | **0.72** | ~0.2 (E from M-AMD) | ~5x |
| 12 | `logits_i8<4>` lm_head, 4 rows | 1.0 | 1,290.2 | **1.29** | 322.8 MB | 250 | 555-565 GB/s (its own 1-row form) | 0.58 | **0.71** | ~0.55 (E-AMD at 545-631) | 2.3x |
| 13 | `r4d_gdn_recurrent_update` (gnt0, P44) | 36.3 | 33.0 (M, P44) | **1.20** | 3.3 MB | 100 | 596 GB/s plus the serial chain: ~15 us | 0.54 | **0.65** | ~0.55 (E from M-AMD's cost model) | 2.2x |
| 14 | Draft-only selection: `rowtopk_stage2` .245 + `stage1` .124 + `merge` .11 + `mtp_rows_gemv` .263 | 12 | 36-88 | **0.74** | small | — | latency | ~0.3 | **~0.45** | ~0.2 (E-AMD) | ~3.5x |
| | **Total of rows 1-14** | | | **~44.2** | | | | **~16.7** | **~27.5** | **~11.1** kernel sum (E-AMD) | ~4x |

Notes on the totals:
- The remaining ~1.5 ms of device time sits in about 30 sub-1 % kernels.
- **If every row reached its class rate (E):** device time 45.7 → **~18.2 ms**. The step would be ~24 ms at ~5.8 ms of host/sync, i.e. **~122 tok/s on this greedy prompt** (2.93 tokens a step), against 56.9 now.
  - This is an upper bound. Exact rewrites have historically recovered less than their predictions: GEMV NX gave +0.7 % and U gave +0 % (STATE.md:1961-1984), and the rows form gave 15 % on the trunk kernel (P45, STATE.md:2213-2216).
- **R9700:** the E-AMD kernel sum is ~11.1 ms against a measured 14.6-15.3 ms step. The rest (~3.5-4 ms) fits expert misses (AMD's own measured mean is 270 vs 576 GB/s best, worth +1.7 ms) plus launch/host time.
  - **Whole-step ratio B70/R9700 ≈ 51.5 / 15.0 = 3.4x (C).**
  - Even after every Intel row reaches its class rate (~18.2 ms device), ~1.6x remains. That residual is the hardware part: no FP8 or codebook matrix path, so E4M3 and u4 are decoded in software.

### 1.1 Why each kernel sits below its rate
**R** = read from the source, **I** = inferred.

1. **`gemv_i8a8` at 4 rows.**
   - R (xe_gemv_i8.cpp:57-105): per 16-byte weight chunk, each of the 4 rows reloads its activation chunk, does 4 DP4A, 3 cross-lane steps, and the scale FMA from lane 0 of each 8-lane group. So the per-column work is repeated 4 times for weight bytes loaded once. The default path still uses the emulated `ds_bpermute` (template `NX = false` in P47's name).
   - I: the cost is per-row ALU and shuffle issue, not bytes. It is 1.90x the 1-row time for the same bytes.
   - Measured partial levers: the 1-row NX/U tries were exact and changed ~0; the rows form `gemm_i8a8_rows` ran 34.8 us (P45) instead of 40.9.
   - A DPAS form with an 8-row tile (half the tile wasted at 4 rows) or a "lane-owns-a-group" mapping is the next form (I).
2. **Expert down (v2m<1>).**
   - R (xe_moe_planes.cpp:133-162, 811-814): each work-group stages the activation rows as f16, their per-128 scales, a 256-entry byte → two-level table and 256 E4M3 values, then passes a barrier. All that serves 64 output rows × 128 bytes = **8 KB of weight**. The down projection has 40 work-groups per routed expert.
   - I: prologue-bound, at 65 GB/s, the worst rate of any big kernel in the step. Per expert the cost stays ~5 us at 1 or 4 rows (P36 4.6 us an expert; P47 5.2).
   - The measured lever is moe8 (4 rows a team at K ≤ 512): −22 us a call at the verify, but −1.9 % at one row. So a **row-count-conditional moe8 is a measured ~1.05 ms with no MTP-off loss**.
3. **Expert gate_up (v2m<5>).**
   - R: the same prologue, but over 64 rows × 1,280 B (80 KB), so it is better amortised (132 GB/s). Per byte it does a table read, an activation read and two f16 FMAs (xe_moe_planes.cpp:136-158).
   - I: the software codebook decode and the grid (8 work-groups per routed expert → ~210 live work-groups) leave it at 39 % of M6's rate.
   - The single prompt's residual misses are inside the window and cannot be separated (I).
4. **hc down group (level 8).**
   - R (r4d_hc_bf16.cpp:1498-1525, 272-300): the grid is 160 work-groups of 2 waves; each wave walks one 10,240-byte W_down column with 2 loads in flight. Each 16 codes are decoded once by `e4m3x4_s8` (software E4M3) but every row reloads and re-widens its 16 bf16 N values.
   - R: the norm and the inject gains are separate launches (6.9 + 9.9 us).
   - I: latency-bound (few bytes in flight) plus software decode. AMD's comment on the same kernel says the same: in-flight loads are "the whole performance of kernel A" (r4d_hc_bf16.hip:345-350).
5. **`hc_up_mix<4>`.**
   - R: W_up rows are staged in LDS behind a barrier (ST), with per-row `d` in LDS and the codes made in the kernel (r4d_hc_bf16.cpp:1100-1147).
   - I: the 1-row form already reaches the E4M3 probe rate (232 GB/s, 14.8 us), so the 4-row form's extra 17.7 us is per-row work, not bytes.
6. **`ar_kernel` LL.**
   - R (xe_allreduce.cpp:291-311): every payload word goes out as an uncached 8-byte store carrying a sequence number, then the reader polls each word with an acquire fence. That doubles the wire bytes.
   - I: bound by PCIe peer-store latency and inter-rank skew (the first rank to arrive spins inside the kernel). It fell from 37.4 to 31.3 us between the halves, which suggests skew is a large part. The 1-row protocol floor is 17.7 us.
7. **`gemv_w2a8` (draft head).**
   - R (xe_gemv_w2a8.cpp:1-40): it uses the trunk GEMV's mapping. Per lane: 4 code bytes against 16 activation bytes, an integer zero point and a scale per 128.
   - I: each lane loads 4 B of weight to 16 B of activation, so the weight stream is a quarter of the vector width per load, and the code is latency-bound at 152 GB/s.
8. **`gemv_bf16<4>`.**
   - R: the router (2.6 MB) is the only large shape. The rest (ssm_ab N 48, qsa_qk, gate column N 1) are launch floors (STATE.md:1985-1990).
   - I: ~100 launches at 13.8 us each, mostly fixed cost.
9. **Elementwise tails.** I: pure launch cost at 3.4-6.2 us profiled (0.6 us unprofiled floor), so this row partly overlaps the host-gap term. AMD fuses `ar_hc_write` and `ar_rmsnorm` (audit.md §4 table); here they are separate launches.
10. **Attention chain.** R (xe_attn.cpp:80-130): one work-group per (split, kv head, query row); 12 sub-groups each decode the same E4M3 KV per element (`ld_kv` → `e4m3()`) and run a 5-step butterfly per key. I: latency-bound.
11. **`router_topk_scatter`.**
    - R (ports/r4d_moe.cpp:2268): the launch is **`dim3(1)`, a single work-group** for the selection and the sort. AMD's lane builtins (DPP/readlane) are emulated through `select_from_group` (m3/note111-inventory.md, cited in STATE.md:2219-2223).
    - I: a serial chain on one core.
12. **`logits_i8<4>`.** I: the same per-row repeat as row 1. The 1-row form runs at 565 GB/s and the 4-row form at 2.26x its time.
13. **GDN update.** R (brief §1.2 row 4; the KT 8 fused-conv launcher): 24 work-groups on 32 cores, 5+ barriers, 2 LDS reductions. I: a serial chain; gnt0 already removed the cache-pollution cost.
14. **Draft selection.** I: top-R over 124K logits per draft pass, as 3 dependent launches.

---

## 2. Decode, MTP off (P36 warm window; ≈ 21.3 ms device after gnt0 + rp, 24.8 ms step)

| # | Kernel | calls (M/C) | us (M) | ms / step (C) | bytes / call (C) | GB/s (C) | target (E) | **saving ms (E)** | R9700 est. ms (E-AMD) | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `hc_v2_down<0>` (with gains) + `hc_v2_norm` | 97.1 | 24.1 + 7.1 | 3.03 | 3.38 MB | 108 | 237 GB/s, norm fused → 1.38 | **1.65** | ~0.6 | 5x |
| 2 | `moe_planes_v2<1,2>` down | 48.0 | 45.7 | 2.19 | 3.38 MB | 74 | 343 GB/s + 4 us → 0.66 | **1.53** | ~0.33 | 6.6x |
| 3 | `moe_planes_v2<5,2>` gate_up | 48.0 | 46.3 | 2.22 | 6.76 MB | 146 | 343 → 0.95 | **1.28** | ~0.56 (576 M-AMD) | 4x |
| 4 | `gemv_i8a8<1>` trunk | 216.2 | 19.7 | 4.27 | 6.8 MB avg | 347 | 550 + small floors → ~3.3 | **~1.0** | ~2.9 | 1.5x |
| 5 | Elementwise tails (hc_write .37, hc_quant .35, gather .26, rmsnorm .24, quant .20, silu .19, rope .16, gated_quant .15) | ~470 | 3.1-5.4 | 1.92 | ~0 | — | fusion → ~1.0 | **~0.9** | ~0.4 | ~5x |
| 6 | `router_topk_scatter` | 48.1 | 18.3 | 0.88 | — | — | 5 us | **0.64** | ~0.2 | 4.4x |
| 7 | `gemv_bf16<1>` (~1.0 after rp) | 146 → ~98 | 8.4 | 1.23 | router 2.6 MB | ~114 | → ~0.45 | **~0.55** | ~0.3 | ~3x |
| 8 | Attention/QSA chain | ~108 | 3-25 | ~1.1 | — | — | ~0.6 | **~0.5** | ~0.25 | ~4x |
| 9 | GDN update (gnt0 at 1 row ≈ 22 us, E) | 36.0 | ~22 | ~0.8 | 3.3 MB | ~150 | ~12 us | **~0.36** | ~0.45 | ~1.8x |
| 10 | `ar_kernel` (1 row) | 97.1 | 17.7 | 1.72 | 10 KB | — | ~15 us | **~0.26** | ~1.0 | 1.7x |
| — | `hc_up_mix<1>` | 96.1 | 14.8 | 1.42 | 3.44 MB | **232** | already at probe rate | ~0 | ~0.75 | 1.9x (hardware: software E4M3) |
| — | `logits_i8<1>` | 1 | 571.8 | 0.57 | 322.8 MB | **565** | at rate | 0 | ~0.55 | 1.0x |

- **MTP-off total saving ≈ 8.7 ms (E)**: device ~21.3 → ~12.6 ms, step ~16 ms, **~62 tok/s** against 40.3 now. That is in the brief's 59-66 range.
- R9700 MTP off has no anchor (production runs MTP), so the column shows per-kernel byte estimates only.

---

## 3. Prefill, 7K prompt (5,604 tokens; B58; seconds per rank; 7K share = 0.3455 of the profile)

The 7K wall time is **12.94 s (M)**. Kernel device time per rank is ~9.20 s (C), and **~3.7 s is outside the kernels**: host, pass boundaries, first-request expert staging, rank imbalance.

| # | Kernel (B58) | calls (both ranks, both prompts) | per call (M) | **7K s / rank (C)** | FLOPs or bytes, 7K per rank (C) | achieved | best class rate (M ref) | target s (E) | **saving s (E)** | R9700 est. s | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `moe_planes_mx` expert matrix form (COMPACT + STAGE) | 2,112 | 9,331 us | **3.40** | 10.6 TFLOP (5,604 × 48 × 10 × 1.966 M MAC × 2) | 3.1 TFLOPS (1.7 % of f16 XMX) | f16 XMX 183 nominal; realistic 30-50 TFLOPS with in-loop u4 decode | 0.25-0.4 kernel; **~1.0 if link-bound** (staging ~1 s a rank, brief §3) | **2.4-3.1** | ~0.35 (E-AMD residual to the anchor; fp8 WMMA 412 TF/s ISA, M-AMD W4A8 72-78 TF/s at M 64) | ~10x |
| 2 | `attn_split` (decode kernel at prefill) | 1,584 | 9,750 us | **2.67** | ~1.4-1.7 TFLOP (≤2,048 keys × 256 × 12 heads × 2 products × 12 layers) | ~0.6 TFLOPS (0.3 %) | f16 XMX; brief target 0.3 s for a blocked DPAS form | 0.3 | **2.37** | ~0.10 (E-AMD) | ~27x |
| 3 | `ar_kernel` LL, large messages | 1,358 | 3,400 us | **0.80** | 2.78 GB payload (5,604 × 5,120 B × 97), 5.6 GB on the LL wire | 3.5 GB/s payload | ~13.5 GB/s peer copy (M-AMD r4d_ar_oneshot_2rank_exact.hip:333; Intel h2d 14.5 GB/s M) | 0.21 (0.08 with AMD's 3/8-byte wht6 wire) | **0.6-0.7** | ~0.11 (E from M-AMD 317-491 us per 2048-row fused call) | ~7x |
| 4 | `hc_pf_dpas<1>` + `<0>` (+ `hc_v2_norm` .05, `hc_write` .05) | 970 + 970 | 1,644 / 1,119 us | **0.46** (+0.10) | 7.1 TFLOP (6.55 M MAC × 2 × 97 × 5,604) | 15.4 TFLOPS | f16 XMX ~50 % → ~92 | 0.08 | **~0.43** | ~0.19 (E from M-AMD 611-767 us per 2048-row call × 2.74 × 97) | ~3x |
| 5 | `gemm_i8a8_dpas` trunk | 2,928 | 907 us | **0.46** | 16.6 TOP | 36 TOPS (20 % of measured 177) | 60 % of 177 → 106 TOPS | 0.16 | **0.30** | ~0.09 (E-AMD at ~180 TOPS) | ~5x |
| 6 | `cs_jm` GDN chunk scan (joint_matrix, jm64) | 504 | 4,090 us | **0.36** | 0.80 TFLOP | 2.2 TFLOPS | the FLOP floor is 4.4 ms; a serial chain over 64-token chunks | ~0.1 | **~0.25** | ~0.05 (E-AMD) | ~7x |
| 7 | `gemm_bf16_dpas` (router, ssm_ab, qsa_qk …) | 2,228 | 608 us | **0.23** | ~0.8-1.0 TFLOP (E) | ~4 TFLOPS | ~40 TFLOPS | 0.03 | **~0.20** | ~0.03 (E-AMD) | ~8x |
| 8 | `moe_gemm_bf16` (10 protected experts, HIP WMMA port) | 176 | 5,799 us | **0.18** | ~0.2 TFLOP (E) | ~1.2 TFLOPS | joint_matrix bf16 | 0.02 | **~0.15** | ~0.01 | ~15x |
| 9 | `qsa_score` (indexer over the full context) | 168 | 7,166 us | **0.07** (context-weighted; 0.21 if linear) | — | — | WMMA on AMD | ~0.02 | **0.05-0.19** | ~0.01 | ~7x |
| 10 | `gemv_i8a8<4>` tails + `kkt_solve` + `gather_wide` + `hc_quant_rows` | — | — | **0.18** | — | — | — | ~0.08 | ~0.1 | ~0.03 | — |
| — | Outside the kernels (host, pass boundaries, cold staging, rank imbalance) | — | — | **~3.7** | — | — | — | ? | (not a kernel) | ~0.05 | — |
| | **Total** | | | **12.94 wall** (9.20 device) | | | | | **~7.2-7.9 kernel** | **~0.97 anchor** (M-AMD 8K 1.38 s scaled) | **~13x** |

- **If every row reached its target (E):** 7K device ~9.2 → ~1.5-2 s; with the 3.7 s non-kernel gap unchanged, **~5.5 s first token** against 12.94 now.
- **The non-kernel 3.7 s is then the single largest item.** It has not been decomposed. It needs a per-pass host/device timeline (an unprofiled B-run with the pass timer), which this read-only analysis cannot produce.

Why each sits below its rate:
- **mx.** R (xe_moe_planes.cpp:456-469 and STATE.md:226-231, 307-321):
  - per 64-K group the block stages its activation rows as f16 and dequantises 64 weight rows' u4 codes through the 16-level table into the packed B layout;
  - then 4 f16 DPAS steps; each tile is stored to LDS and every element is scaled by a_scale × e4m3(scale) in the lane's accumulators.
  - I: barriers and scalar rescales per 64-K group dominate, not the XMX. Run 108 proposed 128-K groups to halve the barriers. STAGE removed the per-group recomputation (−25 %).
- **attn_split.** R (xe_attn.cpp:80-130): the decode kernel run once per 256-query block; one work-group per query row; scalar fp32; each of 12 sub-groups decodes E4M3 per element. No matrix units.
- **ar.** R: as in decode, the LL form doubles the bytes; a word-by-word poll.
- **hc_pf_dpas.** R (r4d_hc_bf16.cpp:4537 onward): on the matrix units. I: the E4M3 → f16 software decode of both matrices per row tile, and N rebuilt per tile. AMD's PF2 comment (r4d_hc_bf16.hip:2525-2540) names exactly these two repeats as what took its form from 711 to 611 us.
- **gemm_i8a8_dpas.** R (xe_gemv_i8.cpp:108-140): per 128-K group the weight block is scattered into LDS for the packed-B layout; the int32 tile is stored to LDS and rescaled per element per group. I: LDS round-trips per 128-K.
- **cs_jm.** R (r4d_gdn_chunk_scan_jm.inc header): grid V/64 × H × 1 = 2 × 24 = 48 work-groups (1.5 a core), 4 barriers a chunk, 95 KB LDS, serial over 64-token chunks. I: occupancy- and latency-bound.

---

## 4. Rankings by achievable saving (all E)

### Decode, MTP on (ms per verify step; step ≈ 51.5 ms)
1. Trunk `gemv_i8a8<4>`: **5.8**
2. Expert down `v2m<1>`: **5.2** (1.05 measured via moe8)
3. Expert gate_up `v2m<5>`: **4.0**
4. hc down group (rows + gains + norm): **2.9**
5. `hc_up_mix<4>`: **1.75**
6. `ar_kernel` LL: **1.45**
7. `gemv_w2a8` draft head: **1.13**
8. `gemv_bf16<4>` (after rp): **~1.0**
9. Elementwise tails: **~1.0**
10. Attention/QSA chain: ~0.75
11. `router_topk_scatter`: 0.72
12. `logits_i8<4>`: 0.71
13. GDN update: 0.65
14. Draft selection: ~0.45

Total **~27.5 ms** of 45.7 device → ~24 ms step (≈ 2.1x tok/s, as a ceiling).

### Decode, MTP off (ms per step; 24.8 ms)
1. hc_v2_down + norm: 1.65
2. Expert down: 1.53
3. Expert gate_up: 1.28
4. Trunk i8: ~1.0
5. Tails: ~0.9
6. Router top-k: 0.64
7. bf16 GEMVs: ~0.55
8. Attention: ~0.5
9. GDN: ~0.36
10. ar: ~0.26

Total **~8.7 ms** → ~16 ms (~62 tok/s).

### Prefill (s per 7K first token; 12.94 s)
1. Expert mx: **2.4-3.1**
2. `attn_split`: **2.37**
3. `ar` LL: **0.6-0.7**
4. hc_pf pair: **0.43**
5. Trunk DPAS: **0.30**
6. Chunk scan: 0.25
7. bf16 DPAS: 0.20
8. Protected-expert bf16: 0.15
9. qsa_score: 0.05-0.19

Total **~7.2-7.9 s** → ~5.5 s first token. A further ~3.7 s outside the kernels is unattributed.

### Biggest B70/R9700 per-kernel gaps (decode, per step)
| Kernel | Gap | Closable on Xe2? |
|---|---|---|
| Expert down | ~7.7x (6.6 vs ~0.9 ms) | Mostly. It is prologue-bound, not decode-bound. To M6's rate the residual is ~1.7x (software u4 decode). |
| hc down group | ~5.4x | To ~1.8x. The E4M3 decode is software (no FP8 units: hardware-bound part). |
| Router top-k | ~5x | Fully. A single work-group with emulated lane ops; no hardware reason. |
| Expert gate_up | 4.4x | To ~1.7x (u4 software decode). |
| hc_up_mix<4> | 4.1x | To ~1.8x. |
| gemv_bf16, tails, attention | ~4x | Mostly. These are launch and fusion costs. |
| `w2a8` | 3.6x | Fully. The int8 class reaches 550 GB/s. |
| Trunk i8 at 4 rows | 3.1x | Fully. Int8 DP4A/XMX exists; the cost is per-row repeated work. Ends at ~1.1x. |
| `ar` | 2.8x | Partly. PCIe peer latency; the floor is ~17.7 us vs AMD's ~11 (E). |
| lm_head at 4 rows | 2.3x | Fully (the 1-row form is at rate). |
| GDN | 2.2x | Mostly. |

**Prefill:**
- attention ~27x (E): closable with a blocked joint_matrix form;
- mx ~10x: closable to the link floor (~1 s); the remainder is PCIe ×8 and residency;
- ar ~7x: closable with a payload-only or compressed wire;
- the non-kernel gap of ~3.7 s vs ~0 on AMD.

---

## 5. Caveats
- The P47/P44 windows are the second half of a cold first request, so some expert misses (in-kernel PCIe reads, 150-220 us each per the brief) can remain inside the moe_planes times. A warm run-2 profile would tighten rows 2-3.
- 26.5 experts a layer comes from the M151 counter over a run on this prompt; other prompts vary (the rotation's H11 gave 29.8).
- The 7K prefill split is a token-share apportionment of a two-prompt profile (E). A 7K-only profiled B-run would replace it.
- The trunk byte total (1.48-1.49 GB) is the brief's figure, not re-derived from per-tensor shapes here.
- Every R9700 per-kernel figure not tagged M-AMD is an estimate. The AMD hc figures in r4d_hc_bf16.hip quote 6.55 MB per matrix; the rates are applied to this model's E4M3 bytes, and whether those comments were written against bf16 or E4M3 matrices is not stated on the lines.
- The decode AR on AMD has no per-call figure.
