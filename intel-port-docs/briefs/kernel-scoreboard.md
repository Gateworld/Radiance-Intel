# Kernel scoreboard (owner note 171): every hot kernel against its floor

Part of the Intel port (2x Arc Pro B70, TP2, Qwen3.8-Flash-Next). A living table, updated whenever a profile or a
kernel change lands; no separate runs for it. Started run 164 (2026-11-03) from the latest profiles: decode P87
(rtneo:180, `svcmtpx-gnt0-hc8-rp-gnx-prof`, the tape profiler's last cumulative table: 356 verify steps a rank,
12,806.5 ms device -> **35.97 ms a rank-step**; the first ~90 steps carry the heat engine's warm-up, so the
per-launch figures are a touch high) and prefill B66 (rtneo:186, the 7K prompt + its logprob form, the matrix-unit
expert form opt-in: `xe launch profile (exit)`, 14,142.9 ms device over both ranks -> **~7.07 s a rank of the
10.85 s first token**; the logprob request adds ~10 % to the launch counts).

Conventions. **Floor** = bytes / 608 GB/s for memory-bound kernels (the card's nominal; the best measured here is
~565 GB/s on the 1-row lm_head), ops / the matrix units' peak for compute-bound ones (int8 177 TOPS measured on
the trunk GEMM's best shape; f16 ~92 TFLOPS = 50 % of nominal), or the protocol / launch floor where neither binds
(said in the row). **Gap** = time / floor. **Share** = of the rank-step's device time (decode) or of the 7K first
token's wall (prefill). Bytes and FLOPs per call are the headroom brief's (briefs/2026-10-31-kernel-headroom.md
§0.3). Calls a step are launches / 356 (P87). Counters: unitrace's metric groups load in the test image
ri-m3ut:2 (UT1, run 164: 13 groups; ComputeBasic, VectorEngineStalls with XVE_STALL_{BARRIER,CONTROL,SBID,
INSTFETCH,PIPESTALL,OTHER}, VectorEngineProfile, MemoryProfile with GPU_MEMORY_BYTE_READ/WRITE). The first per-kernel query is **B90** (run 164: the 7K under
`unitrace -q`, the RenderBasic group = this build's default, image ri-m3ut:3 = rtneo:188 + unitrace; 27,510
instances, 231 kernel+shape keys, 18.5 s of GpuTime over both ranks). Under the query the kernels run ~1.5x long
(the 7K took 34.3 s), so only the ratios are read; logs/ut_query.py summarises perfquery.metrics.<pid>.csv (per
kernel and shape: GpuTime-weighted means of the % metrics, sums of the counts). "matrix" = ALU2 utilisation (the
systolic pipe), "ALU1 / ALU0" = the vector pipes' instruction-slot utilisation.

## Decode, MTP on (P87, ms a rank-step; the step ~51 ms, ~2.9 tokens a step)

| # | kernel / variant | calls a step | us a launch | ms a step | floor (how) | gap | share | counters | last change | next idea |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `moe_i4i8_run_kernel<5,16,1,0,1>` expert gate_up, format C (w4nl64a8h, int8 rows, DP4A) | 51.1 | 114.7 | **5.86** | 29.4 us: 17.9 MB (26.5 distinct experts x 675,840 B) / 608 GB/s | 3.9 | 16 % | — | format C default, note 160 (run 135, 128d9b8) | note 167: the streaming floor, down first; 8 lane partials folded per g -- fewer partials, wider weight loads |
| 2 | `gemv_i8a8_kernel<1,4,1>` trunk W8A8, 4 rows | 220 | 24.8 | **5.46** | 11.1 us: 6.77 MB avg (1.49 GB a step) / 608 GB/s | 2.2 | 15 % | — | rows form (run ~124, P45) | per-launch floor ~5 us x 220: fuse the pairs (rp did 48); GL (opt-in, on in the rotation's bestr) |
| 3 | `moe_i4i8_run_kernel<3,2,2,1,1>` expert down, format C | 50.9 | 67.3 | **3.43** | 14.7 us: 8.95 MB / 608 GB/s | 4.6 | 9.5 % | — | format C default (note 160) | note 167 first target (down): 2 partials at K <= 512 already; the 256-wide N rows need 4 teams a run |
| 4 | `hc_up_mix_kernel<4,1,1,1,1,0>` hc up mix (E4M3 W_up 10240 x 320) | 97 | 31.5 | **3.06** | 5.7 us: 3.44 MB / 608 GB/s | 5.5 | 8.5 % | — | hc level 8 (note 158, run 125) | int8 rows (hci8, scored 0.044 KL -- at the bar's edge); E4M3 -> f16 decode per element is the cost |
| 5 | `ar_kernel<0,0,4>` LL all-reduce + hc write (form 5 since run 163: 20.2 us, P86) | 106 | 25.1 (form 5: 20.2) | **2.66** (2.14) | ~8 us: the 16-byte LL probe moved 16 KiB in 7.9 us (PEER2); one-way store 2.07 us | 3.1 (2.5) | 7.4 % | — | form 5 default (run 163, 6919054 / 4c24c9d) | fold the hc write's second pass; a 32-byte store width probe; note 168 item 3 for prefill |
| 6 | `hc_v2_down_rows_kernel<4,0>` hc down, 4 rows | 98 | 25.7 | **2.52** | 5.6 us: 3.38 MB / 608 GB/s | 4.6 | 7.0 % | — | level 8 (note 158) | same as row 4: the E4M3 decode; int8 N rows (hci82, 64-token chunks only so far) |
| 7 | `gemv_w2a8_rows_kernel<1,2>` MTP draft head, 3 passes | 3.0 | 513 | **1.55** | 131 us: 79.5 MB / 608 GB/s | 3.9 | 4.3 % | — | w2a8 rows form | 2-bit decode in software: a 256-entry table like the format C path; or the draft gate (note 165, opt-in) skips a pass |
| 8 | `r4d_gdn_recurrent_update_kernel` (gnt0) | 35.9 | 31.6 | **1.13** | 5.4 us: 3.3 MB state read + write / 608 GB/s | 5.8 | 3.1 % | — | gnt0 (note ~120) | serial over the step's rows; the fused-recurrent reference (briefs/2026-11-01-fla-*) |
| 9 | `hc_v2_down_kernel<2,1,0>` hc gains | 97 | 10.2 | **0.98** | ~3 us launch/latency | 3.4 | 2.7 % | — | — | fuse into row 6 (one launch a connection) |
| 10 | `router_topk_scatter_kernel<16,8,0>` | 51 | 18.7 | **0.95** | ~5 us (AMD's 5.4 us at 32 rows) | 3.7 | 2.6 % | — | router pair rp (29101e1) | one work-group does the whole 4-row top-k: wider sub-group sort |
| 11 | `gemv_bf16_kernel<0,0,4>` router / ssm_ab / qsa_qk GEMVs | 51 | 16.4 | **0.84** | 4.3 us (router 2.6 MB / 608) | 3.8 | 2.3 % | — | rp | fuse the router into the hc read (same input rows) |
| 12 | `logits_i8_rows_kernel<4,2>` lm_head, 4 rows | 1.0 | 722 | **0.72** | 531 us: 322.8 MB / 608 GB/s | 1.4 | 2.0 % | — | rows form | at the floor's edge; nothing cheap left |
| 13 | `hc_v2_norm_kernel` | 106 | 6.8 | **0.72** | ~3 us latency | 2.3 | 2.0 % | — | — | fuse with hc_write (row 15) |
| 14 | `gemv_bf16_pair_kernel<0,0,4>` | 48 | 12.6 | **0.61** | ~4 us | 3.2 | 1.7 % | — | rp | — |
| 15 | tails: `hc_write` 0.40, `hc_quant_rows` 0.37, `qsa_select` 0.31, `attn_split` 0.30, `mtp_rows_gemv` 0.27, `rowtopk_stage2` 0.25, `qsa_score` 0.23, `hc_v2_down<0>` 0.19, `attn_merge` 0.18, `gemv_i8a8<1,1,1>` 0.17 | ~600 | 3.6-90 | **~2.7** | launch floor 0.6 us unprofiled | — | 7.5 % | — | — | fusion (note 171's "fewer launches") |
| | **sum of the rows** | | | **~32.5 of 35.97** | **floor sum ~9.5** | 3.4 | 90 % | | | |

Largest absolute time over its floor (decode): row 1 gate_up (5.86 − 1.50 = **4.4 ms**), row 2 trunk (5.46 − 2.44 =
3.0), row 3 down (2.7), row 4 hc up (2.5), row 6 hc down (2.0), row 5 all-reduce (1.8). Note 167 (format C's
experts toward the streaming floor, down first) is already queued on rows 1 and 3.

### Decode, MTP on, current (P110, 2026-11-10, image 292 = every default incl. the five fusion flags, the GRF / cache bundle, W2A8_R=16, NEO 26.40 + IGC 2.42.6, the scalarized format C kernels; `svcmtpx-gnt0-hc8-rp-gnx-prof`, 78.8 tok/s under the profiler against 86.2 plain)

Device time 10,415 ms over 356 logits launches = **29.3 ms a rank-step** (P87, 11-04: 12,806 ms = 36.0 ms; -19 %). ms a rank-step = the kernel's ms / 356.

| # | kernel | launches a step | us a launch | ms a step | share | P87 ms | note |
|---|---|---|---|---|---|---|---|
| 1 | `moe_i4i8_run_kernel<5,16,..>` format C gate_up | 51.1 | 107.6 | 5.50 | 18.8 % | 5.86 | the scalarized form (M368 / H150 null); M398 (image 304, every fusion default): 109.2 us x 51.1 = 5.58 ms. **FLOOR RECONCILED (run 279, owner note 218 stage 0): 26.5 distinct experts a layer (P87's counter) x the two parity classes' average expert = (768 + 512) / 2 columns x 1,320 B (1,280 of codes + 40 of E4M3 scales) = 844,800 B -> 22.4 MB a launch = 36.8 us at the card's 608 GB/s; gap 2.9x at 107.6 us (3.0x at 109.2), ~3.6 ms a step over; at the format's 343 GB/s probe rate (M6) 65 us, 1.65x.** The earlier 29.4 us / 17.9 MB counted the 512-column class alone (675,840 B; the P87 table above still carries it); note 167's "1.3x" counted ~34 MB a launch (~50 experts: the 6-user verify). Note 218's stage 2 gate (<= ~70 us a launch) is 320 GB/s effective, near the format's best measured streaming rate |
| 2 | `gemv_i8a8_kernel<1,4,1,0,5>` trunk W8A8, 4 rows (dp4a, WCC 5) | 220 | 18.6 | 4.09 | 14.0 % | 5.46 | dp4a builtin 26.3 -> 19.1 (run 207), now 18.6 |
| 3 | `moe_i4i8_run_kernel<3,2,2,..>` format C down | 51.0 | 57.6 | 2.94 | 10.0 % | 3.43 | M398: 59.0 us = 3.00 ms. Floor reconciled the same way (run 279): 26.5 x (506,880 + 337,920) / 2 B (2,560 columns x K 384 / 256, 198 / 132 B a column) = 11.2 MB -> 18.4 us at 608 GB/s (not 14.7); gap 3.1x; 32.7 us at 343 GB/s |
| 4 | `hc_v2_gu_kernel<4>` hc up-mix + gains (fusion default) | 97 | 28.3 | 2.74 | 9.4 % | 3.06 + 0.98 | the GU pair replaced rows 4 + 9 |
| 5 | `hc_v2_down_rows_kernel<4,0,2>` hc down, 4 rows (D16) | 98 | 25.3 | 2.48 | 8.5 % | 2.52 | |
| 6 | `ar_kernel<0,0,5>` LL all-reduce, form 5 | 106 | 19.5 | 2.06 | 7.1 % | 2.66 | |
| 7 | `r4d_gdn_recurrent_update_kernel` (gnt0) | 35.9 | 31.7 | 1.14 | 3.9 % | 1.13 | |
| 8 | `router_topk_scatter_kernel` (256 GRF) | 51 | 15.0 | 0.76 | 2.6 % | 0.95 | |
| 9 | `logits_i8_rows_kernel<4,2>` | 1 | 711 | 0.71 | 2.4 % | 0.72 | |
| 10 | `hc_v2_write_norm_kernel` (write + norm fused) | 94 | 7.2 | 0.68 | 2.3 % | 0.72 + 0.40 | |
| 11 | `gemv_bf16_kernel<0,0,4>` | 51 | 13.0 | 0.67 | 2.3 % | 0.84 | |
| 12 | `gemv_bf16_pair_kernel` | 48 | 11.6 | 0.56 | 1.9 % | 0.61 | |
| 13 | `gemv_w2a8_wide_kernel<1,0>` draft head, 3 passes (W2A8_R=16) | 3 | 185.2 | 0.56 | 1.9 % | 1.55 | the wide form: 513 -> 185 us |
| 14 | `hc_quant_rows` 0.36, `attn_split` 0.26, `qsa_select` 0.26, `quant_act_fp8` 0.26, `gather` 0.26, `mtp_rows_gemv` 0.25, `rmsnorm` 0.22, `rope` 0.22, `gated_quant` 0.20 + 0.16, `attn_merge` 0.17, `gemv_i8a8<1,1>` 0.16 | | | ~2.8 | 9.6 % | | |

Largest absolute time over the floor now (the floors at 608 GB/s, the step's measured bytes; the realistic rate for each class in brackets; rows 1 and 3 at the reconciled floors of run 279): row 1 ~3.6 ms (36.8 us x 51.1 = 1.88 ms floor; ~2.2 at the format's 343 GB/s probe rate), row 2 ~1.6 ms (6.77 MB x 220 = 1.49 GB a step = 2.45 ms at 608; measured 4.09), row 3 ~2.0 ms (18.4 us x 51 = 0.94 ms floor; ~1.3 at 343 GB/s), rows 4-5 (hc, ~4.1 ms together over their 1.1 ms of bytes: the E4M3 decode), row 6 (the handshake: ~1.2 ms over a ~8 us floor a launch). Every exact single-kernel idea on rows 1-6 has been tried this milestone (see "What did not work" in the report skeleton); the remaining room is structural (the all-reduce protocol, the expert layout, ESIMD), each recorded for the owner.

## Prefill, 7K first token (B66: 5,604 new tokens in 4096-token chunks; s a rank; wall 10.85 s)

| # | kernel / variant | calls (both ranks) | us a launch | s a rank | floor (how) | gap | share of 10.85 s | counters | last change | next idea |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `moe_i4i8_dpas_kernel` format C experts on int8 DPAS (64 sorted rows x 64 cols a group, K groups of 64) | 576 | 7,195 | **2.07** | 0.06 s: 10.6 TOP (5,604 x 48 x 10 x 1.966 M MAC x 2) / 177 TOPS; memory 0.02 s (12.2 GB of expert codes a rank / 608 GB/s) -- compute-bound | 34 | 19 % | B94 (5b62eaa): **SLM bank conflicts 0**, stall 48-50 % (B90 52-56), active 44-52 % (39-47), occupancy 85-93 %, matrix pipe 1.2-1.7 %, ALU1 29-39 % (23-33: the select chain), SEND 1.23e11 (1.03e11) and load-store cache accesses 8.1e10 (4.6e10): the per-lane row reads and the per-group scale loads -- the next cut hoists the scales; under the query the launches read 11.0 / 6.5 / 9.3 / 5.3 ms (B90 10.6 / 6.4 / 8.8 / 5.2) while the plain B-run is 4.9 % faster on the chunk. B90 (e7613c3): conflicts 3.3e10 a 7K, stall 52-56 %, matrix 1.3-1.8 %, DRAM 42-48 GB/s | 5b62eaa (run 166, note 172 a): the level table -> a register select chain, B stores 16 consecutive words a sub-group, A stores one 256-byte run -- SLM bank conflicts 3.3e10 -> 0, bits identical; B93 7K 10.49 s / 20K 16.86 against e7613c3's 10.85 / 17.02 (a per-code sub-group shuffle was tried first and lowered through SLM: +20 % a launch, B92). Before: e7613c3 (run 164, note 168 item 2; B89 10.85 / 17.02 against the run form's 13.56 / 21.76; KDP2 KL 0.0256 / top-1 97.4 %) | the decode no longer touches memory; **run 170, cut 1 (5f3802d): the accumulator scaled in registers instead of through sC -- exact (B111 on 197: the 7K logprobs byte-identical to B109's), neutral in time (7.71 / 11.73 vs 7.63 / 11.62), so the counted SLM round trip was not the wall. B112 (ut:6, the counters on the new form): SLM bytes 4.0 -> 2.3 TB a 7K (reads −30 %, writes −66 %: the round trip gone, no spill), the launch 10.79 -> 10.95 ms, ALU1 4.54e11 -> 4.59e11 instructions, stall 48 -> 49 %: THE SLM PORT IS NOT THE WALL; cut 2 (the 2-D tiling) not built. The wall is the instruction stream on ALU1 (2.4e9 a launch, the 4-bit decode's select chain re-run by every row run, ~18 an expert) at a 49 % stall. B113 (image 198, the decode skipped, timing only): the 7K 7.12 s against 7.71 -- THE DECODE IS ~0.6 s OF THE 7K, under a fifth of the kernel; decode-once-a-run is worth <= 0.4 s and is not the lever. The ~2.7 s rest is the per-group pipeline (staging sends, 2 barriers a group, the MMAs and tile loads, the scale loads). B114 (image 199, the scaling's loads skipped): 7.35 s -- the scale loads are ~0.36 s of the 7K (~11 % of the kernel; hoisting them is a real, exact item). B115 (the B staging skipped): 6.56 s -- the B staging is ~1.15 s of the 7K (~35 %: the decode 0.6 + the code loads, packing and 8 SLM stores a lane a group 0.55), done once a work-group but again by every row run (~3.4 for the 7K): a 128-row work-group halves it (~0.5 s, exact). B116 (the MMAs and tile loads skipped): 7.18 s -- ~0.53 s (~16 %). The table: decode 0.59, scale loads 0.36, B staging 1.15 (decode inside), MMAs 0.53 -- 2.04 s named of ~3.3; the remainder ~1.3 s is the A staging, the two barriers a group x 24 x 8 sub-groups, the loop and epilogue. Design (run 171): the scale-load hoist (~0.3 s) + a 128-row work-group (the staging and barriers once per 128 rows: ~1.0-1.2 s), both exact -- the kernel ~3.3 -> ~1.8 s, the 7K ~7.7 -> ~6.2**; next (owner note 172): the 128-K-group rework if this row still ranks first -- (1) B decoded once an expert run instead of once a 64-row block, (2) 128-K groups, 2 accumulators; grf256 only if spill-free |
| 2 | `attn_split_kernel<8,1,0,0,1>` (the decode kernel at prefill, one work-group a query row, scalar f32) **-> run 170: replaced opt-in by sycl-tla's FMHA (attn_qsa_fmha, B107: the 7K 9.60 -> 7.93 s, the 20K 15.72 -> 12.14; KFL1 mean KL 0.0078 / top-1 99.8 % inside note 149's bar); the owner makes it the default** | 576 | 6,145 | **1.77** | 0.03 s: ~1.5 TFLOP / ~50 TFLOPS f16 XMX realistic (0.009 at nominal) | ~60 | 16 % | B90: active 61 %, stall 31 %, occupancy 72 %, ALU1 47 % / ALU0 21 % / **matrix 0 %**, DRAM 3 GB/s: vector-ALU-bound (the E4M3 decode and scalar f32 math), 1.7e12 vector instructions for the 7K | unchanged since the decode form | note 168 item 1: a chunk_prefill port (vllm-xpu-kernels, sycl-tla) -- blocked DPAS, E4M3 K/V decoded once per tile |
| 3 | `ar_kernel<0,0,5>` LL all-reduce at prefill, large messages **-> run 170: the bandwidth form (proto 6, opt-in: the payload once + a position-dependent checksum in tagged words, re-read on a tear -- the card tears 11-41 of 200 large launches, every one caught; B108: the 7K 9.60 -> 9.30 s, the 20K 15.72 -> 15.19, exact), the tags' ~0.3 s of the 7K gone; the rest of the row is the rendezvous and the ranks' skew** | 776 | 2,165 | **0.84** | 0.25 s: 2.78 GB payload / 11.3 GB/s peer stores (PEER2; h2d 14.5) | 3.4 | 7.7 % | B90: active 1.3 %, **stall 94 %**, occupancy 12 %, DRAM 6 GB/s, matrix 0 %: a polling kernel -- the time is the link, not the card | form 5 (run 163) | note 168 item 3: a bandwidth (not LL) form for chunks: wide stores, no tags, one flag a message |
| 4 | `gemm_i8a8_dpas_kernel<1>` trunk | 1,056 | 722 | **0.38** | 0.094 s: 16.6 TOP / 177 TOPS | 4.0 | 3.5 % | B90: active 46 %, stall 50 %, occupancy 94 %, matrix 5.8 %, ALU1 26 %, DRAM 71 GB/s, SLM conflicts 1.2e10 over 72 launches, shared-function hold 11 % | — | LDS round trips per 128-K group (headroom brief §3); the same conflict fix as row 1 |
| 5 | `cs_jm::kernel<64,0,1,1>` GDN chunk scan (joint_matrix, jm64) | 216 | 3,329 | **0.36** | 0.0044 s FLOP floor; latency-bound: 48 work-groups serial over 64-token chunks | ~80 | 3.3 % | B90: active 8.9 %, stall 66 %, **occupancy 19 %**, ALU 1.6 / 4.1 / matrix 0.6 %, **i-cache misses 9.5e8 over 216 launches** (4.4 M a launch: the grf256 kernel's code thrashes the instruction cache), SLM conflicts 1.8e9 | jm64 (run ~150) | more groups (split V or H further), or the chunk-parallel form in vllm-xpu-kernels' gdn_attn |
| 6 | `hc_pf_dpas_kernel<1>` + `<0>` hc at prefill | 388 + 388 | 1,528 / 941 | **0.30 + 0.18** | 0.077 s: 7.1 TFLOP / 92 TFLOPS | 6.2 | 4.4 % | B90: `<1>` active 45 %, stall 53 %, occupancy 61 %, matrix 3.4 %, ALU1 32 %, DRAM 55 GB/s, SLM conflicts 5.5e9; `<0>` matrix 6.9 %, DRAM 87 GB/s | PF DPAS (note 153) | the E4M3 -> f16 decode per tile and N rebuilt per tile (AMD's PF2 fix) |
| 7 | `gemm_bf16_dpas_kernel<0,0>` router / ssm_ab / qsa_qk | 776 | 591 | **0.23** | ~0.02 s (~0.9 TFLOP / 50) | ~10 | 2.1 % | — | — | the router on the int8 path |
| 8 | `moe_gemm_bf16_kernel` protected experts (bf16 WMMA port) **-> run 268: `moe_bf16_dpas_kernel` (joint_matrix bf16, the block map of row 1; opt-in RADIANCE_XE_MOE_BF16_DPAS=1, 020501a): 43 µs a launch against 5,625 (B187 / B190: the three prompts' device time 25.4 -> 23.7 s, −6.6 %; B189 the 7K 3.48 -> 3.34 s, the 20K 6.72 -> 6.41, the text equal, the first token's logprob 0.031 off); not bit-identical (the K sum's order), MB2: the outliers are cancellation (ratio >= 270k), the new form nearer fp64 on 34 of 44; the KL replay (KMB1 / KMB2) decides the default** | 288 (B187) | 5,625 -> 43 | 1.62 s (both ranks, B187) -> ~0.01 | the 288 launches' MACs: ~2 x live rows x 640 x 2560 a launch (the live rows from B191's trace) | — | 6.4 % (B187) -> ~0 | — | 020501a (run 268) | the 257-511-slot calls join once B191 shows what they are; T <= 256 stays on the GEMV form (decode's class) |
| 9 | `moe_i4i8_run_kernel` <5,16,...> + <3,2,...>: the T < 512 tails (the last chunk's small runs) | 192 + 192 | 1,066 / 717 | **0.17** | ~0.03 s | ~6 | 1.6 % | B90: active 9-15 %, stall 80 %, **shared-function access hold 40-57 %** (the data port holds the 16-byte lane loads), DRAM 40-43 GB/s | — | lower the DPAS cut-off (RADIANCE_XE_I4I8_DPAS_MIN) once padding rows cost less |
| 10 | `qsa_score_kernel<4,4>` indexer | 96 | 2,060 | **0.10** | ~0.01 s | ~10 | 0.9 % | — | — | WMMA on AMD; joint_matrix here |
| 11 | `gemv_i8a8<1,4,0>` 0.07, `r4d_gdn_kkt_solve` 0.06, `hc_v2_norm` 0.05, `hc_write` 0.05, `gather_wide` 0.04, `hc_read_a` 0.03 | — | — | **0.30** | — | — | 2.8 % | — | — | — |
| — | outside the kernels: host n-gram gather + build at the chunk boundaries (~0.5 s a chunk), pass boundaries, the stager's copies (12.2 GB a rank, synchronous in the memcpy call at 14.9 GB/s: ~0.3 s a 4,096-row chunk) and -- until run 173 -- the Level Zero backend's event records blocking the host at every stager event (3.5 s of a 3.6 s chunk, the rank thread issuing one layer at a time: RADIANCE_L0_RECORD_BARRIER=1 lifts it, B133: the 7K 7.71 -> 7.01 s, the 20K 11.73 -> 10.30, exact, opt-in), rank imbalance (B63 timeline) | — | — | **~3.8** | — | — | 35 % | — | — | note 168 item 4 + note 166: the gather for chunk N+1 on the CPU while chunk N runs. **Run 169: the gather was libref's page-faulting mmap walk (~0.9 + 0.45 s a 7K prompt; the test images had no libavx.so); libavx's io_uring row cache + read-ahead (B102, image 193): 43 + 1.5 ms, the 7K 10.49 -> 9.60 s (−8.5 %), the 20K 16.86 -> 15.72 (−6.8 %), exact.** What is left outside the kernels (~2.9 s of 9.60): the 36-row tail step ~1.1 s (the owner's tail checkpoint, note 135 item 2), the step-top gap ~0.24 s (the embedding all-reduce + the ranks' skew), two ~0.2 s host stalls (inside layer 0 and late in chunk 2), ~1.7 s of 1-100 ms gaps (skew, per-layer issue waits) |
| | **sum** | | | **~7.0 kernel + 3.8 other = 10.85** (run 169: the gather item takes ~1.3 s off "other": **B102 9.60 s** = the new baseline, owner note 175) | | | | | | |

Largest absolute time over its floor (prefill): row 1 the expert DPAS (2.07 − 0.06 = **2.0 s**), row 2 attention
(1.74), the non-kernel part (3.8, not a kernel), row 3 the all-reduce (0.59), rows 5 / 6 (0.36 / 0.40).

### Prefill, current profile (B192, 2026-11-10, image 292 = IGC 2.42.6 libxe + the 2.28.4 hcpf module; the 7K + 20K + 7klp prompts, both ranks; first tokens under the profiler 4.06 / 7.65 / 4.03 s) and the compiler comparison (B193 = the same code, image and runtime with the WHOLE libxe compiled by IGC 2.28.4; owner note 209 item 1)

Device time over the three prompts: libxe's kernels 25,491 ms (B193: 29,193, +14.5 %) + the hcpf module's 4,483 ms (2.28.4 in both).
Launches 88,918. us a launch (2.42.6 / 2.28.4), ms in all (2.42.6), share of the 29,974 ms.

| # | kernel | launches | us 2.42.6 | us 2.28.4 | 2.28.4 / 2.42.6 | ms (2.42.6) | share | note |
|---|---|---|---|---|---|---|---|---|
| 1 | `moe_i4i8_dpas_kernel` format C experts, int8 DPAS (V3 + V3b) | 2,880 | 2,437 | 3,057 | 1.25 | 7,018 | 23.4 % | at its plain-SYCL floor (round 3 dropped); ESIMD-only beyond |
| 2 | `ar_kernel<0,0,6>` the bandwidth all-reduce, large messages | 2,322 | 1,559 | 1,596 | 1.02 | 3,619 | 12.1 % | link-bound (~11 GB/s); overlap = lever #2 (owner) |
| 3 | `hc_pf_dpas_kernel<1>` + `<0>` (the 2.28.4 module) | 1,552 + 1,552 | 1,779 + 1,110 | — | — | 4,483 | 15.0 % | the 2.42.6 build hangs (run 262); by rows 1 / 4 / 5 a working one would be 15-27 % faster |
| 4 | `cs_jm::kernel<64,0,1,1>` GDN chunk scan, grf256 | 864 | 3,061 | 3,874 | 1.27 | 2,645 | 8.8 % | ISA (igcdump-pf2426/gdn): 14,000 instr, **510-539 GRF variables spilled at 256 GRF** (1,080-1,134 at 128); jm32 spills 248-275, jm16 110-125 -> the v-block A/B (B194 / B195) |
| 5 | `gemm_i8a8_dpas_kernel<1>` trunk W8A8 (V3b) | 4,608 | 562 | 711 | 1.27 | 2,588 | 8.6 % | 1,448 instr, 0 spills at 128 GRF, 73 a64 + 33 a32 sends; A / B staged through SLM (item 3's candidate) |
| 6 | `gemm_bf16_dpas_kernel<0,0>` router / ssm / qsa GEMMs (V3b) | 3,488 | 485 | 556 | 1.15 | 1,691 | 5.6 % | 1,443 instr, 0 spills |
| 7 | `moe_gemm_bf16_kernel` protected experts, WMMA port | 288 | 5,627 | 5,454 | 0.97 | 1,621 | 5.4 % | lever #3 replaces it (opt-in: 43 us a launch) |
| 8 | `qsa_score_kernel<4,4>` grf256 | 360 | 2,495 | 2,287 | 0.92 | 898 | 3.0 % | 2.28.4 faster by 20 ms in all |
| 9 | `r4d_gdn_kkt_solve_kernel<3>` | 864 | 638 | 625 | 0.98 | 552 | 1.8 % | |
| 10 | `gemv_i8a8_kernel<1,4,0,0,0>` (the decode GEMV at the tails) | 17,712 | 29.7 | 29.9 | 1.01 | 526 | 1.8 % | |
| 11 | `ar_kernel<0,0,5>` LL form, the small messages | 1,164 | 435 | 471 | 1.08 | 506 | 1.7 % | |
| 12 | `moe_i4i8_run_kernel<5,16,..>` + `<3,2,..>` format C tails | 576 + 576 | 889 + 567 | 783 + 499 | 0.88 | 838 | 2.8 % | 2.28.4's build issued 672 of each (more tail calls), so its totals rose |
| 13 | `hc_v2_norm` 494, `hc_write` 470, `gather_wide` 424 (2.28.4: 376), `hc_read_a` 207 (297), `hc_quant_rows` 156, `gated_quant_fp8` 135 (121), `gemv_bf16` 127, `attn_split` 115, `hc_up_mix<8>` 117 (130), `gdn_conv_prep` 125 (111) | | | | | ~2,370 | 7.9 % | |

The compiler verdict: 2.42.6 wins every heavy kernel (rows 1, 4, 5, 6 by 15-27 %); 2.28.4 wins only rows 8, 9 and a few small ones (~110 ms = 0.4 % in all) -> no per-unit mix (owner note 209 item 1 closed, run 270). The 7klp logprobs are bit for bit equal under both compilers (B155 / B159 / B175).
Largest absolute time over its floor now: row 1 (~6.9 s over a ~0.2 s TOPS floor), row 3 (4.5 s, the compiler-stuck unit), row 2 (link-bound), row 4 (2.6 s, latency / spill-bound), rows 5 + 6 (4.3 s against ~0.4 s of DPAS floor: item 3's 2D block loads).

Item 3 measured (run 270, B196 / B197 on image 293, `RADIANCE_XE_GEMM_DPAS_T=1`, commit 80e65d4, opt-in; the 7K logprobs bit for bit B155's):
- row 6 `gemm_bf16_dpas_t_kernel` 233.6 us a launch against the staged 484.9 (-52 %; the family 1,691 -> 829 ms);
- row 5 `gemm_i8a8_dpas_t_kernel` 548.0 us against 561.6 (-2.4 %): the int8 kernel is bound by its per-128-K fold (32 extractions, 32 int -> float, 32 FMAs and 12 scale loads a sub-group a group; ~1,600 fold instructions against 256 DPAS at K 2,048), not by the staging; next: the scale loads a group ahead (building), then the fold's ALU;
- the staged bf16 kernel on one 64-column block (the column tails) costs 241 us = half the full launch: the K loop's per-sub-group latency, times ~8 waves of work-groups, is what these kernels pay -- not throughput;
- device time 25,491 -> 24,498 ms (-3.9 %); first tokens 7K 3.48 / 20K 6.58 / 7klp 3.32 against 3.48 / 6.71 / 3.43: under the 5 % bar, so off by default.
- chunk scan (row 4), the v-block A/B B194 / B195: `<32>` is 5 % SLOWER on the first tokens than `<64>` though it spills half as much (logprobs identical) -- the serial structure, not the spill traffic, is the cost; v64 stays.

## Counters, the first reading (B90, 2026-11-03, prefill)

- **No prefill kernel is near a hardware limit.** DRAM read peaks at 87 GB/s (hc_pf_dpas<0>) against 608; the
  matrix pipe peaks at 6.9 % (hc_pf_dpas<0>), 5.8 % on the trunk GEMM, 1.3-1.8 % on the expert DPAS kernel.
- **The expert DPAS kernel (row 1) is LDS- and vector-bound:** 3.3e10 SLM bank conflicts over 192 launches with
  occupancy 84-94 % and stalls 52-56 %. The A tile (uint4 pairs a thread into 64-byte rows) and the packed B
  writes (`sBw[((k4/8)*4 + t)*128 + (k4%8)*16 + ln]`) conflict; the table decode and the per-tile rescale run
  on ALU1 (23-33 %). Fixing the layouts is the cheapest next step on the largest prefill gap.
- **Attention (row 2) is vector-ALU-bound** (ALU1 47 %, ALU0 21 %, matrix 0 %, DRAM 3 GB/s): the port to the
  matrix units (note 168 item 1) is the right fix, not a tuning.
- **The all-reduce at prefill (row 3) is idle polling** (stall 94 %, active 1.3 %, DRAM 6 GB/s): the time is the
  link (2.78 GB payload at ~3.5 GB/s effective against the probe's 11.3 GB/s peer stores) -- note 168 item 3.
- **The GDN chunk scan (row 5) thrashes the instruction cache** (4.4 M misses a launch) at 19 % occupancy: a
  code-size problem (the grf256 jm64 kernel) before a FLOP problem.
- **The run-form tails (row 9) stall on the data port** (shared-function hold 40-57 %): the 16-byte lane loads
  queue; at decode this is the kernel's design (memory-bound), at prefill it is the reason to lower the DPAS
  cut-off once padding costs less.
- Decode's counters: the per-instance query (`-q`) did not finish a 256-token request in 900 s (M251, run 164:
  hundreds of 5-40 us launches a step, each queried); next: time-based sampling (`unitrace -k`, logs/mutstep.sh
  with that prefix) or a 32-token request.

## Parameter sweeps kept

| kernel | knob | swept | kept | where |
|---|---|---|---|---|
| `moe_i4i8_dpas_kernel` | accumulation order (plain vs tree by g mod 8) | B66 7.19 ms vs B68/B69 11.54 ms a launch (grf256 + strided sweep) | plain | run 164 |
| `moe_i4i8_dpas_kernel` | `RADIANCE_XE_I4I8_DPAS_MIN` rows | 512 only | 512 | run 164 |
| `cs_jm` | chunk / tile | jm64 vs jm32 | jm64 | run ~150 |

## Log

- 2026-11-11 (run 294, owner note 234): groups A / B are the DEFAULT (commit 9f2884f, image 311; RADIANCE_XE_GEMV_MULTI2 unset or 1 lists both, 0 opts out, a / b one group). The flagless reference on 311: M407 91.36 tok/s single-user (texts = M285), M408 -prof 84.42 with device time 27.4 ms a rank-step (M403 on 309: 28.2) and the table showing gemv_i8a8_bf16x2 47.7 a step and gemv_i8a8_x3_bf16 12.0 a step by default, gemv_bf16_pair / gemv_i8a8_x3 gone, the standalone trunk GEMV 96.4 a step. The ar + write + norm design is shelved (owner), the 221 us boundary accepted as device-side; the decode queue is empty of exact items of 1 % or more. Next: note 235 (the fp16 GDN state cache, non-exact: accuracy first).
- 2026-11-11 (run 285, owner note 232 groups A / B, image 310, opt-in RADIANCE_XE_GEMV_MULTI2=1 / a / b; commit f77876f): two more paired decode GEMV launches, exact (CPU row tests + the card row check GSH1 byte-identical). A = the FFN block's fused router launch with the shared expert's gate_up in its grid (`gemv_i8a8_bf16x2_kernel<1,4,5>`: 47.7 a step at 14.0 us, in place of gemv_bf16_pair 11.8 + a standalone gemv_i8a8 13.4 + a boundary; the standalone gemv_i8a8<1,4,1,0,5> falls 144.5 -> 96.3 a step). B = the attention's x3 with the QSA indexer's qk projection (`gemv_i8a8_x3_bf16_kernel<1,4,5>`: 12.0 a step at 41.8 us, in place of gemv_i8a8_x3 35.7 + gemv_bf16 19.1 + a boundary). M406 (-prof): device time 27.5 ms a rank-step against M403's 28.2 (-2.5 %); M404 (A alone) 91.06 and M405 (A + B) 91.01 tok/s single-user against the flagless reference M402 88.99 (+2.3 %), texts = M285 -- B is null on the single prompt. The rotation pair H157 / H158 decides; the defaults are the owner's.
- 2026-11-10 (run 279, owner note 218 stages 1-2, CLOSED): the whole-kernel ESIMD form of gate_up (fc_tile_probe's E3 /
  E3b, logs/run279-esimd-design.md) is BIT-IDENTICAL to today's kernel on card1 at 4 tokens / 27 experts and 1 token /
  10 experts, and 1.55-1.66x SLOWER (today 73.7 us a launch at the probe's 4-token mix, E3 121.7, E3b 115.5; 30.0 /
  48.5 / 45.9 at 1 token). 2.8x fewer instructions a MAC ran at ~4x fewer instructions a cycle: the 4-bit decode's
  table lookups go through the one address register (serial a0 moves), the same verdict as P100's for down. The
  prefetch / hoisting round recovered ~5 %. Row 1's remaining gap is the decode's cycle cost in every form we can write
  (SLM gathers, register-indirect lookups); the ESIMD route is stopped, the probe stays as the record, and note 219 A's
  relayout alone (fewer instructions, the same lookups) is not worth doing. Rows 1 / 3 stay as they are.
- 2026-11-10 (run 279, owner notes 217 / 218): pairs 4 / 5 (the paired GEMV launches) are the DEFAULT too (commit
  00fad49); the flagless reference on image 304: 88.63 tok/s (M397), the table (M398) with gemv_i8a8_bf16_pair /
  gemv_i8a8_x3 in place of 111 standalone GEMV launches a step, 28.3 ms of device time a rank-step. Rows 1 and 3's
  floors reconciled to the two parity classes' average expert (22.4 / 11.2 MB a launch: 36.8 / 18.4 us; gaps 2.9x /
  3.1x) -- the 29.4 / 14.7 us figures counted the smaller class alone. Note 218 stage 0: a whole-kernel ESIMD form of
  gate_up compiles AOT under IGC 2.42.6 / ocloc 26.40 at 2,540 instructions a thread for 2x today's work (7.9 a dp4a
  against 22.1), 0 spills at 128 GRF (logs/run279-esimd-design.md).
- 2026-11-10 (run 279, owner note 215): the three exact decode fusions (gated_quant_pair, the qk_norm_rope_gate
  prologue, router_topk_scatter_quant) are the DEFAULT (commit 593f6e6; =0 opts out). The new decode reference on
  image 303, no flags: 87.62 tok/s single user (M395), the kernel table (M396) with the three fused kernels in place
  of the six they replaced. The four-flag rotation (H154, image 302) vs the three flags (H152): +1.0-1.2 % on every
  measure, every prompt length up; vs the bundle alone (H153) +2.7-2.9 %. Pairs 4 / 5 (gvm) stay opt-in for the owner.
  The router kernel (row 10): 15.3 µs x 48 a pass by default now; the bench variants RB4 (the sort with the slot list
  in registers) and RB4a (native sub-group scans) measured exact and NOT faster (base 9.9-10.7 µs at 4 rows in the
  bench, the variants 10.7-21.5 / 11.0; 6 µs slower at 1 row for the register form) -- record and stop: the kernel's
  time is the launch floor, the one-wave selection (5.0 µs) and the sort's barriers. Owner note 217: pairs 4 / 5 go
  default too (the commit and the new reference in STATE).
- 2026-11-10 (run 274): the map's pairs 4 and 5 as one mechanism -- several projections of one activation in one GEMV
  launch (flag gvm): `gemm_nt_q_x3` for the attention's q|gate, k, v and `gemm_nt_q_bf16_pair` for the GDN in-projection
  with its bf16 a|b; each row the standalone GEMV's row copied verbatim, the standalone kernels untouched (their 18.5 /
  13.1 µs re-read unchanged with the flag off, M394). The profile (M391, image 302): the pair kernel 35.8 launches a
  pass at 46.1 µs and the x3 13.4 at 35.7, the standalone int8 GEMV -76 launches a pass and the bf16 -36; the small
  GEMVs (one kv head, 96 a|b rows) run beside the big ones. +1.32 tok/s on the same image (88.50 vs 87.18 with the
  three earlier pairs; +1.5 %). The four flags together: 88.50 against ~85.6-85.9 plain, +3.0-3.4 %; 90 is 1.5 away.
  The decode table's row 10 (the router) is the largest small-kernel item left: 15.4 µs x 48 a pass (2.4 %), one
  256-GRF work-group. The same-image rotation of the three flags (H152 / H153): +1.6-1.8 % median, +1.7 % aggregate.
- 2026-11-10 (run 273, owner note 214 item 3): the decode step's adjacent-launch fusions, each exact by construction
  (the unchanged kernel bodies in one grid), each byte-for-byte on the B70 and texts = M285 at 1 / 4 tokens, each
  opt-in and committed. Pair 1 `gated_quant_pair` (the routed + shared gated quantisers, flag gqp): +0.35 tok/s (+0.4 %,
  image 298). Pair 3 as libr4d's `qk_norm_rope_gate` (q norm, k norm, both ropes in one launch; ported in m1, never
  listed for want of a libref schema; flag qkp): +0.92 (+1.1 %, image 299) -- the q / k rmsnorm<1024> pairs and the
  rope pairs leave the decode table. Pair 2 `router_gemm_topk_scatter_quant` (the block input's quantiser beside the
  router's one 256-GRF work-group, flag rqp): the profile shows the pair at 15.1 µs where router_topk_scatter<16,8,0>
  took 15.0 and quant_act_fp8<1,1> 5.2 (row 10 of the decode table: the router's 15 µs now also covers the quantiser);
  +1.04 (+1.2 %, image 301). The set ~+2.3 (~2.7 %) if additive; the three-flag config and the rotation are next. Lesson
  recorded: a fusion design names the profile kernels that must disappear and appear from the kernel table -- pair 2's
  first form paired the quantiser with the standalone router op the step never takes (+0.00, caught by the profile).
- 2026-11-08 (run 268): prefill row 8 (the protected experts' bf16 GEMM) moved to the matrix units as an opt-in
  (020501a; B190: 43 µs a launch against 5,625, the device time of the three prompts −6.6 %); the KL replay decides its
  default. B187 / B190's exit tables (image 289 / 290, the warm-up on, both ranks over the 7K + 20K + 7klp prompts):
  moe_i4i8_dpas 2,426-2,435 µs x 2,880 = 27.6-29.4 %; ar<6> 1,512-1,516 x 2,322 = 13.8-14.8 %; cs_jm<64> 3,056-3,059 x
  864 = 10.4-11.1 %; gemm_i8a8_dpas 561 x 4,608 = 10.2-10.9 % (753 before V3b); gemm_bf16_dpas 485 x 3,488 = 6.6-7.1 %
  (602 before V3b); qsa_score 2,497 x 360 = 3.5-3.8 %; gdn_kkt 638 x 864 = 2.2-2.3 %; gemv_i8a8 29.7 x 17,712 = 2.1-2.2 %;
  moe_i4i8_run 893 x 576 = 2.0-2.2 %; hc_v2_norm 318 x 1,552 = 2.1 %; hc_write 163 x 2,880 = 2.0 %. Rows 1-7 and 9-11
  of the table above still carry B66 / B90-era figures; the shares here are the current ranking.
- 2026-11-03 (run 164): file started from P87 (decode) and B66 (prefill). Counters: the metric groups load
  (UT1); no kernel query yet.
- 2026-11-03 (run 164): prefill row 1 committed as the default (e7613c3; B89 7K 10.85 s, KDP2 inside the bar).
- 2026-11-03 (run 164): B90, the first per-kernel counters (prefill, RenderBasic): the counters column and the
  reading above; row 1's next idea changed to the LDS conflict fix.
- 2026-11-03 (run 167): prefill row 2 -- the port's engine integration built (op attn_qsa_fmha, the FMHA in its
  own module kernels/xe/libxe_fmha.so, opt-in); its first two engine runs (B97 / B98) refused the chunk on the
  op's two raw operands and faulted the card in the abort -> PAUSED; the row's figures unchanged.
- 2026-11-03 (run 166): prefill row 1 -- 5b62eaa (conflict-free, bit-exact, 7K −3.3 %) kept; two further
  bit-exact cuts tried on top and NOT kept (neutral end to end, B95 / B96): the per-group scale loads hoisted
  (4 groups' E4M3 scale bytes in one load, the activation scale once per two groups) and a two-group staging
  round (128 K of A, both groups' B tiles, one barrier pair per two groups, a lane's code reads in one line).
  Reading: the time is in what every sub-group repeats -- the 4 KB packed-B set re-read from LDS by all 8
  sub-groups every group (~32 KB of SLM reads a group a work-group) and the decode's instruction stream; next:
  a 2-D sub-group tiling (32 rows x 16 columns a sub-group: B read once per column tile, A twice) then B decoded
  once an expert run.
- 2026-11-03 (run 165): prefill row 2 (attention) -- the port's kernel verified standalone: sycl-tla's FMHA with
  the per-query page mask does a 4,096-row chunk in 3.78 ms (dense 1.96) against ~100 ms today (FQ1); an
  LDS-staged joint_matrix form of our own reached only 129 ms a layer (ATP1): LDS staging is the wall. The row
  moves to ~0.09 s a rank once wired in (next run).
- 2026-11-04 (run 169): the "outside the kernels" row (note 168 item 4 + note 166) -- the gather was libref's
  page-faulting mmap walk (the images had no libavx.so); libavx's io_uring row cache + read-ahead served
  instead (image 193, `--kernels libxe:libavx:libref`): the gather 43 + 1.5 ms a 7K prompt against ~900 + ~450,
  the 7K 10.49 -> 9.60 s, the 20K 16.86 -> 15.72, exact (logprobs, M-pair). Owner note 175 made it the default
  and B102 the baseline. What is left outside the kernels (~2.9 s): the 36-row tail step (~1.1 s, the owner's
  tail checkpoint), the step-top gap (~0.24 s), two ~0.2 s host stalls, ~1.7 s of 1-100 ms gaps. The engine's
  --profile-ops timer (B103) hangs a tail step under the glue: not a tool for this.
- 2026-11-04 (run 170, 03:05): row 1's first cut, the accumulator scaled in registers (5f3802d; the sC
  store / read-back behind RADIANCE_XE_I4I8_DPAS_SLMC=1): exact on the card (B111, image 197: the 7K logprobs
  byte-identical to B109's) and NEUTRAL in time (7K 7.71 s, 20K 11.73 against B109's 7.63 / 11.62). B110's
  counters had put half the kernel's ~20 GB of SLM traffic a launch on that round trip; removing it bought
  nothing, so the SLM port is not the wall it looked -- the next counters (ut:6 = 197 + unitrace, B112) say
  whether the bytes fell (the decode's instruction stream is then the cost; cut 2's 2-D tiling would not pay)
  or not (the compiler spilled the work-item read). Row 1 stays first at 2.07 s.
- 2026-11-04 (run 170, 03:10): the counters on cut 1 (B112, ut:6 = 197 + unitrace, `unitrace -q`): the
  gate_up instance's SLM bytes 2,733 / 1,264 GB -> 1,903 / 426 GB over 192 launches (the C round trip's 830 GB
  gone from each side; no spill), SEND −21 %, and the launch 10.79 -> 10.95 ms, ALU1 instructions 4.54e11 ->
  4.59e11, stall 48 -> 49 %. Reading: THE SLM PORT IS NOT THE WALL -- B110's "SLM-bandwidth-bound" was wrong
  (the bytes were a symptom); cut 2 (the 2-D sub-group tiling) would not pay and was not built. The wall is the
  ALU1 instruction stream (2.4e9 a launch: the 4-bit decode's select chain, 256 nibbles a lane a 64-K group,
  decoded again by every row run of the expert) at a 49 % stall with high occupancy. Next (run 171, the start
  of the 128-K-group rework): lvl() is already the packed four-word form (3 selects + shift + mask a nibble,
  5b62eaa), so the lever is decoding once an expert run (~3.4 row runs an expert for the 7K at 256 experts a
  layer); first the decode's share, measured by a timing-only step with the decode skipped (B113,
  RADIANCE_XE_I4I8_DPAS_SLMC=2, wrong numbers by construction). The gate for a real form: the 7K logprobs
  byte-equal.
- 2026-11-04 (run 170, 03:26): the decode's share, measured by ablation -- B113 (image 198, the 4-bit level
  decode skipped by the test switch RADIANCE_XE_I4I8_DPAS_SLMC=2, numbers wrong by construction): the 7K first
  token 7.12 s against B111's 7.71. The decode is ~0.6 s of the 7K, under a fifth of row 1's ~3.3 s (gate_up
  2.07 + down 1.26 a rank); decoding once an expert run (~3.4 row runs an expert for the 7K) would save <= 0.4 s
  for a scratch the arena cannot hold whole -- not the lever. The remaining ~2.7 s is the per-64-K-group
  pipeline (the B staging's 64 SLM word stores a lane, the A run, two barriers a group x 24 groups a
  work-group, 8 MMAs + 10 SLM tile loads, the per-group scale loads, 32 FMAs) at a 49 % stall and 85-93 %
  occupancy. Run 171: the same ablation for the pipeline's three parts (the scaling, the B staging, the MMAs:
  slmc = 3 / 4 / 5, one build, three short 7K steps) before any design of the owner's 128-K-group rework.
- 2026-11-04 (run 170, 03:42): B114 (image 199 = d1adb20's three more switches, the scaling's loads skipped,
  timing only): the 7K 7.35 s against B111's 7.71 -- the per-group scale loads (4 expert scales + 8-32
  activation scales a lane a 64-K group) are ~0.36 s of the 7K, ~11 % of row 1. With B113's decode (~0.6 s),
  ~1.0 s of the kernel's ~3.3 s is named; the B staging (B115) and the MMAs with their tile loads (B116) are
  the next two shares; what none of the four removes (the A staging, two barriers a group, the loop and the
  epilogue) is the remainder. Hoisting the scale loads is exact and small -- an item on its own.
- 2026-11-04 (run 170, 03:44): B115 (image 199, the B staging skipped, timing only): the 7K 6.56 s against
  7.71 -- the B staging (the 16-byte code load, the level decode, the packing, 8 SLM word stores a lane a
  64-K group) is ~1.15 s of the 7K, ~35 % of row 1; the decode (B113) is 0.6 of it, the loads + packing +
  stores 0.55. It is done once a work-group (sub-group w owns tile w/2, step w&1 -- no redundancy inside) but
  again by every row run of the expert (~3.4 for the 7K): a work-group of 128 sorted rows (16 a sub-group:
  two 8x16 accumulators a tile, 64 dwords a lane) halves the staging for the same MMAs -- ~0.5 s of the 7K,
  exact. With the scale-load hoist (~0.3 s) and the decode's remaining share, ~1.5 s of the kernel's ~3.3 s
  is now named; B116 (the MMAs and tile loads) names the next part.
- 2026-11-04 (run 170, 03:47): B116 (image 199, the MMAs and their tile loads skipped, timing only): the 7K
  7.18 s against 7.71 -- ~0.53 s, ~16 % of row 1. The ablation table is complete: decode 0.59 (B113), scale
  loads 0.36 (B114), B staging 1.15 with the decode inside (B115), MMAs + tile loads 0.53 (B116) -- 2.04 s of
  the kernel's ~3.3 s a 7K named; the ~1.3 s remainder is the A staging, the two barriers a 64-K group x 24
  groups x 8 sub-groups, the loop and the epilogue (the 49 % stall at 85-93 % occupancy is the barrier-paced
  pipeline, not a port). Design for run 171 (the owner's 128-K-group rework, from the measurement): the
  scale-load hoist (~0.3 s, exact) and a work-group of 128 sorted rows (16 a sub-group, two accumulators a
  tile; the B staging and the barriers paid once per 128 rows: ~1.0-1.2 s, exact) -- row 1 ~3.3 -> ~1.8 s, the
  7K first token ~7.7 -> ~6.2 s. Image 199 (= d1adb20) carries the switches; the gate for the real changes is
  the 7K logprobs byte-equal to B111's.
- 2026-11-04 (run 171, 04:06): row 1 software-pipelined (B117, image 200: the next group's global loads
  issued before this group's barrier and MMAs, consumed a group later): exact (the 7K logprobs byte-identical
  to B111's) but +4.7 % on the 7K (8.07 s) and +4.9 % on the 20K (12.31). sycl::group_barrier's fence covers
  global memory, so the loads in flight were waited for at the barrier (the latency moved, not hidden) while
  the double buffer cost ~42 dwords a lane. Not committed. Next on one image: the group's loads batched at its
  top (one exposed latency a group instead of three) under a local-fence barrier (xe::syncthreads_local:
  __spirv_ControlBarrier with WorkgroupMemory only), and the 16-sub-group work-group (128 rows; each sub-group
  decodes 4 words instead of 8, an expert's run staged half as often).
- 2026-11-04 (run 171, 04:30): two more exact forms of row 1 on image 201. B118 (the group's global loads
  batched at its top, the barriers fencing shared memory only, W = 8): byte-exact, neutral (7K 7.96 s, 20K
  11.73, the warm 6.51 against 7.71 / 11.73 / 6.71). B119 (W = 16: 256-thread work-groups of 128 rows, each
  sub-group staging half the B words): +15 % on the 7K (8.85 s), +10 % on the 20K, and NOT byte-exact (one
  logprob off by 0.039 with the same arithmetic -- a race across the local-fence barrier at 16 sub-groups is
  the only candidate), so both the W = 16 form and the local-fence barrier are dropped (the diff kept in
  logs/run171-batched-localbarrier-w16.diff). Reading: with three load schedules neutral (B111 / B117 / B118),
  the chain's cost is not global-load latency; it is the shared-memory round trips (stores before the
  barrier, the 10 tile loads feeding 8 dependent MMAs), the decode's dependent ALU chain and the barriers
  under threads in lockstep (XVE active 44 %, stall 49 %, occupancy 85-93 %). Next: 16 rows a sub-group under
  the large register file (RT = 16, launch_as_grf256): each B tile load feeds two MMAs and the staging, decode
  and barriers are paid once per 128 rows -- the last exact structural lever before this row is called at its
  form's floor.
- 2026-11-04 (run 171, 04:55): row 1's last exact structural lever, 16 rows a sub-group under the large
  register file (B121, image 202; the build log reports "allocated 256 regs and spilled around 3" for it and
  no spills for the default form or B117's prefetch form): 7K 8.71 s (+13 %), 20K 13.37 (+14 %) -- slower, and
  the same 0.039 logprob gap as B119's 16-sub-group form (the two 128-row forms agree with each other
  exactly; the run form, B122, is 0.21 / 0.25 from the 64- / 128-row forms, so the DPAS form's acceptance
  stays note 149's bar). The padding hypothesis for the gap is wrong (ref_moe.cpp pads only the tail); the
  cause is open. CONCLUSION FOR ROW 1: five exact restructurings (cut 1, the prefetch, the batched loads, 16
  sub-groups, 16 rows a sub-group) are neutral or slower; the 64-row form of run 170 is this structure's
  floor, ~3.3 s a 7K, bound by shared-memory round trips, the decode's dependent chain and barriers under
  lockstep threads -- not by any port. A faster expert prefill needs a different algorithm (the codes decoded
  once a layer into an int8 scratch the matrix units read directly -- ~0.5 GB a layer in VRAM, the staging
  buffers' size x3; or a dense int8 expert format for prefill), which is a design item for the owner, not a
  kernel tweak.
- 2026-11-04 (run 172, 05:05): decode rows 5 / 13 / 15 (the all-reduce, hc_v2_norm, hc_write), owner note 165
  item 2 (the all-reduce fused with the residual write and the RMSNorm that follow it). The residual half
  exists since run 99 (RADIANCE_XE_AR_HC_FUSE=1, the all-reduce's emit pass doing hc_write's update, every
  proto incl. the LL form 5) but was never a default: M256 on image 202 with it, against M254 without: 69.30
  vs 69.62 tok/s (−0.5 %, noise), texts identical, klog clean -- the hc_write launch's 0.40 ms a step is not
  visible at the token rate (the tape replay overlaps the launch, or the emit pass grows by as much). The norm
  half's ceiling is its launch, 0.72 ms of the ~51 ms step (1.4 %), minus the last-block tail that would
  compute it, so the pair cannot reach the owner's 1.5 % rotation gate: NOT BUILT; the fuse stays opt-in.
  Rows 5 / 13 / 15 keep their "fuse" ideas as sub-gate items. Next: note 165 item 3's measurement (L1).
- 2026-11-04 (run 173, 07:25): the "outside the kernels" row -- the host stall named and lifted. The engine's
  RADIANCE_DEBUG_STAGE timer, extended to print a prefill-sized pass's parts and the host time per op
  (before / issue / after the op), then the stager's bytes and copy submissions, then the Level Zero
  backend's host time inside memcpy / settle / barrier waits / event records, found that a 4,096-row chunk's
  3.57 s rank thread spent 3.5 s inside rad_event_record: on an in-order queue the record takes the queue's
  last event (DPC++ ext_oneapi_get_last_event), and the query waits for the queue's work, so the host never
  issued more than one layer ahead of the device (the copies themselves: 4 GB a chunk synchronous at 14.9
  GB/s, 0.3 s). RADIANCE_L0_RECORD_BARRIER=1 (l0.cpp real_record: a barrier instead) -- the records ~1 ms a
  chunk, the host issues a chunk in 0.13 s, the chunk is the device's time (3.30-3.42 s warm against
  3.56-3.68). B133 on image 210: 7K first token 7.71 -> 7.01 s (−9 %), 20K 11.73 -> 10.30 (−12 %), warm 7K
  6.71 -> 5.80; logprobs byte-identical; committed deee214, opt-in. What is left of a chunk is the device:
  the expert DPAS kernel (row 1, ~2.3 s of a 4,096-row chunk) -- the algorithm question in WAITING is the
  next lever. The shared-chunk slowdown of run 172 (6.3 s a chunk beside decoders with the same kernels)
  stays open at the device level.
- 2026-11-04 (run 174, 07:50): no kernel change; the milestone's table refreshed on image 210 (deee214) with
  the three opt-in switches (fmha + ar6 + rb): the 6-user aggregate 132 tok/s (C13; target 120), the rotation
  median 88.8 (H119; H110 79.5 before the switches), the recorded-pass pool 0.24 GB a card (H119's list
  counts), the misses 28.1 a step (residency; the <= 4 target is the slab's, not a kernel's). The expert
  prefill algorithm (decode once a layer into an int8 scratch) re-costed: ~1.2 GB a rank a layer of decoded
  B (every expert is routed by a 4,096-row chunk), net ~0.4 s a chunk (~−8 % on the 7K) for 8 % of the expert
  slab -- parked (STATE's run-174 entry has the brief); row 1 keeps its "different algorithm" note. Next by
  the owner's order: note 165 item 4's measurement (the distinct experts a verify step reads a layer and the
  union's shrink under S2-MoE's bias / MoE-Spec's budget), then note 167 (rows 1 / 3 toward the streaming
  floor, down first).
- 2026-11-04 (run 174, 08:55): decode rows 1 / 3 (the expert kernels' bytes a step) -- owner note 165 item 4,
  the expert-union shrink at verify, measured and gated. A 4-row verify step reads 24.6 distinct experts a
  layer of its 40 picks (M259's router dump; 27.2 on the rotation at 3.9 rows). S2-MoE's lean of the two
  lowest picks within 0.005 of probability shrinks the union by 3.4-4.0 experts (−14-16 %, ~1.1-1.3 ms a
  step by the launch-time scaling, at the owner's 1 ms bar); built opt-in as RADIANCE_XE_ROUTE_LEAN in the
  fused router kernel (image 212), the KL replay FAILS note 149's bar at 0.005 (mean KL 0.072 / top-1 95.6 %)
  and at 0.002 (0.0685 / 96.0 %): the loss is a tail of positions. Lossy -> waits for the owner; rows 1 / 3
  keep note 167 (exact: the kernels toward their streaming floor) as the next idea.
- 2026-11-04 (run 174, 09:20): prefill rows 2 (the attention op), 3 (the bandwidth all-reduce) and the
  "outside the kernels" row (the record barrier) are THE DEFAULTS from 09e7610 (owner note 176 item 1; image
  213): B134 with no flags 7K 7.87 s cold / 20K 10.32 / warm 7K 5.76, logprobs byte-identical; M261 decode
  identical; the op's seven-document replay (KFL2) mean KL 0.0028 / top-1 99.3 %. The prefill table's "sum"
  row now reads from B134: ~5.8 s warm, 7.0-7.9 cold (the cold spread is the first launches, the warm-up pass
  next).
- 2026-11-04 (run 175): the startup warm-up pass is the default (74e6dd0): the 7K first prompt after startup 4.93
  s (B135), truly cold 6.92 (B136). Prefill row 1's decode-once form (137ce32, opt-in): exact, neutral in time,
  −1.77 GiB of slab, −7 % on the rotation (H120). Decode rows 2 / 4 / 6 on the matrix units at 4 rows re-timed
  (P88-P90): trunk +11 %, hc −1.2 %: under the gate. Decode rows 1 / 3 by ablation (P91-P95): the staging ~29 %,
  the weight loads 23-32 %, the level lookups 9-14 %, the scale loads 1-3 % of the kernel; the loads moved before
  the staging (exact, P96) neutral -- the next idea is less work: the staging and the table once a run, the
  lookups by the select chain.
