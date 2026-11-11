# Where the Radiance-on-B70 port loses its time

Read-only synthesis of the M3 measurements (results-m3/card01), the eight cluster analyses after skeptic correction, and the four gap closures. Every number carries its source and whether it is **M** (measured, read from a file) or **C** (computed here). Where the skeptics overturned the original clusters I use the corrected figure and say so. Line numbers into STATE.md drift (the runner appends); P3/P4/P5/B10 `serve.log` lines are stable.

Conventions. One *rank-step* = one decode step on one card; the two ranks run concurrently, so a rank-step's device time is the step. P3/P4/P5 tape tables sum both ranks and are divided by the `logits_i8` launch count (one per rank per step): **940 / 381 / 380 rank-steps** (P3/serve.log:841, P4:760, P5:759). The model has **36 GDN + 12 attention layers** (P3/serve.log:124), so STATE.md's "35 GDN a rank-step / 70 a step" divisor is 2.6 % low everywhere it was used. Peak bandwidth: 608 GB/s datasheet; **596 GB/s** is round 1's measured read ceiling (round1/STATE.md:63-64) and is used as the floor divisor.

---

## 1. The MTP-off decode step (27.86 ms wall, 36.0 tok/s)

### 1.1 What the profiler says, and two corrections to how it was read

P3 (image :33, single warm 15-token prompt, 256 generated tokens): 65,536 lists, 39 kernels, **26,832.5 ms** of bracketed device time over both ranks (P3/serve.log:830, M) → 28.55 ms per rank-step at 940 rank-steps (C). Two things make that number not the warm step:

1. **It is a cold+warm run mean.** Differencing the eight cumulative tables (serve.log:664→830, C): run-1 windows 29.6 ms/rank-step, run-2 warm windows **26.77 / 26.79 / 28.34 ms**. Per kernel every window is flat to ±0.5 us/launch except the expert GEMVs (`moe_planes_v2<5>` 71.1→48.5 us, `<1>` 57.7→45.5 us: the heat engine promoting the prompt's working set in the first ~50 steps), the all-reduce (25.5→17.3 us) and `moe_gemm_bf16`, which is bimodal (101 / 262 / 344 us a launch by window; +1.28 ms/rank-step is the whole 26.8→28.3 swing).
2. **The windows carry stamp cost.** Each profiled command is three in-order commands (`WriteGlobalTimestamp`, kernel, `WriteGlobalTimestamp`; l0.cpp:1586-1613, M); the bracket includes the stamp→kernel dispatch gaps, 0 ≤ S_in ≤ 2 × 0.593 us × 1,380 commands = 1.64 ms/rank-step (0.593 us = measured L0 in-order per-command floor, round2/STATE.md:62-63). The profiled wall step was 32.00 ms vs M20's 29.32 ms for the same window with the same `ranks` phase → 0.97 us per stamp command all-in (C), so "3.1 us tiny-kernel floor" figures are profiled upper bounds; the unprofiled floor is 0.6 us.

Unprofiled wall step: **27.86 ms = ranks 0.26 + collect 27.60** (M20/out-M20.txt:12, M; mean/128). Warm device 26.78-26.93 ms → **host/sync gap 0.9-1.1 ms if S_in = 0, up to 2.7 ms if S_in is at its ceiling** (C). The step is ~90-96 % device, not the 98 % the clusters assumed; the measured serial host part is 0.18-0.37 ms (M15/M18/M20 `ranks`).

### 1.2 Decomposition that sums to the wall step

Rows are warm-window values (whole-run P3 values in the note where they differ). Floors are bytes-that-must-move at 596 GB/s; realistic ceilings are the best *measured* rate for that kernel class on this card.

| # | Item | ms | % of 27.86 | Bytes floor @596 | Realistic ceiling (measured class rate) | Why it is above the floor |
|---|---|---|---|---|---|---|
| 1 | Trunk GEMVs: `gemv_i8a8<1>` 4.67 (216 launches × 21.7 us), `gemv_bf16<1>` 1.25 (146 × 8.6), `logits_i8<1>` 0.58, `rmsnorm` 0.24 — P3:831,839,841,846 (M/C) | **6.74** | 24 % | 1,956 MB → **3.28** | 4.4-4.8 (M1 probe 530-573 GB/s on the big shapes, out-m1.txt:23-28) | i8<1> streams 1,478 MB at 316 GB/s = 53 %; the 96 big launches ~344 GB/s, the 120 small ones (N 256-1,280) are 0.78 ms at the launch/occupancy floor; 146 bf16 GEMVs take 1.25 ms for 0.26 ms of bytes; three dependent `select_from_group` shuffles per 128-K group (xe_gemv_i8.cpp:52-76); 0 B spill (K1-out.txt:528-531). lm_head is the one kernel at bandwidth (555 GB/s). |
| 2 | HyperConnection mixes on E4M3: `hc_read_a` 4.21 (97 × 43.5 us), `hc_up_mix<1>` 3.43 (96 × 35.8), `hc_write` 0.37 (96 × 3.9) — P3:832,833,842 | **8.01** | 29 % | 680 MB (replicated per rank, rad_block_hc.h:29-31) → **1.14** | 3.17 (round-2 probe at the identical E4M3 shapes: 232 / 242 GB/s, out-m1.txt:32-33) | 81 / 96 GB/s = **14-16 % of peak**. No FP8 units: every code goes through the scalar `rad_fp8e4m3_to_f32` (xe_lanes.h:153-160). `hc_read_a` runs 81 work-groups with 4 of 8 waves streaming and every WG redoing the grouped norm over h before any weight byte (r4d_hc_bf16.cpp:803-887, 3912); `hc_up_mix` 160 WGs with 4 barriers + fence + atomic arrival. 0 B spill (K1:534,537); the 66-70 KB spills are tiled kernels that are never launched (`tiled = false &&`, :3796). |
| 3 | Expert path (warm): `moe_planes_v2<5>` 2.33 + `<1>` 2.18 (48 × 48.5 / 45.5 us), `router_topk_scatter` 0.88, `gather` 0.26, protected-expert `moe_gemm_bf16` 0.81 (low mode; whole-run 1.60) — P3:834,835,838,840,843 | **6.46** | 23 % | 480 units × 1,268,480 B = 609 MB → **1.02** | 1.78 at M6's 343 GB/s (round2/STATE.md:137) + ~0.4 launch floors | 135 GB/s warm = 23 % of peak, 39 % of M6. Down class (K 256/384 at TP2, one 16-B load a lane) scales linearly with rows → per-work-group overhead, not a per-launch intercept; gate_up grid 80-120 WGs = 1.25-1.9 waves on 64 slots; router/gather are byte-free launch floors (r4d_moe.cpp:295-301); the bf16 pair is a WMMA HIP port with 3,968 B spill. Single-prompt misses 2.5 × 87.5 us = 0.22 ms sit inside these windows. |
| 4 | GDN recurrent update 2.08 (36 × 57.9 us) + attention side 1.36 (193 launches: `attn_split` 32 forced splits + `attn_merge`, QSA 5-kernel chain, 3 rope, q/k norms, gates) — P3:836,844-860 | **3.44** | 12 % | 36 × 3.3 MB + 1.7 MB KV → **0.20** | ~0.7 at the launch structure (229 × 3.1 us) | GDN moves 3.3 MB in 57.9 us = 57 GB/s (9.5 %); launched kernel spills 0 B (K1:388; the 32 KB rows are the un-launched CV=0/KT=2 family), and KT 4→8 / 3.3 KB→0 spill changed the step by nothing (M18/M20 26.7/26.7) → the time is the serial chain (24 WGs on 32 cores, 5+ barriers, 2 LDS reductions). Attention reads 512 B per token-layer-rank (24 KiB page / 4 tok / 12 layers) — pure latency. |
| 5 | Two-rank all-reduce `ar_kernel` (warm 97 × 17.3 us; whole-run 21.3 us = 2.06) — P3:837 | **1.68** | 6 % | 97 × 5,120 B → **0.03** (wire); flag round trip 1.5-2.6 us → 0.15-0.25 | ~1.5 (the protocol itself) | Uncached peer stores + release fence + flag + seq_cst fence + poll (xe_allreduce.cpp:236-256); the first rank to arrive spins inside the kernel, counted as device time. M2b's persistent form (24.0 us) is the same protocol, so a persistent kernel buys only the 0.06-0.29 ms launch floor. |
| 6 | Element-wise tails not in the clusters above: `quant_act_fp8<1,0>` 0.20, `silu_mul` 0.19, `gated_quant` 0.15, `moe_blockmap` 0.03, 9 unlisted 0.10 — P3:846-860 | **0.60** | 2 % | ~0 | 0.1-0.3 | ~160 launches at 3.1-4.2 us profiled windows; the unprofiled command floor is 0.593 us, so most of this row is where S_in hides. |
| 7 | Host issue / step-end sync / 66 list boundaries (wall − warm device) — M20/out-M20.txt:12; P3 windows | **0.93** (range 0.9-2.7) | 3-10 % | 0.26 (measured serial `ranks`) | ~0.5 | 12-38 us per list boundary or event-sync wake-up; shrinks only by composing fewer lists. |
| | **Total** | **27.86** | 100 % | **5.9** | **~15-17** | |

Bandwidth floor of the whole step: **5.9 ms = 3.3 GB at 596 GB/s → ~165 tok/s** (C). The step runs at an equivalent **123 GB/s = 21 % of peak**. At the best measured per-class rates plus launch floors the step is ~15-17 ms → **59-66 tok/s** MTP off; that, not 165, is the realistic ceiling on this card without new silicon (see §5).

Ambiguities: (a) S_in (0-1.6 ms) moves time between rows 6/7 and the kernel rows; (b) the `moe_gemm_bf16` mode (0.8 vs 2.1 ms) is routing-dependent and unexplained; (c) `ar_kernel`'s 17.3 us mixes protocol latency and inter-rank skew in an unmeasured ratio.

---

## 2a. The MTP-on step (53.6 ms unprofiled best on :36, 42.0 tok/s)

P5 (image :35, before DP4A/v2m): 21,522.6 ms over 380 rank-steps = **56.6 ms** profiled (P5/serve.log:748, M/C); unprofiled M22 54.5 best / 56.8 recent on :35, M26 53.6 best on :36 (STATE M22/M26 rows). Device-bound in the same sense as §1: profiled device 56.6 vs profiled wall 58.5 ms (256 tok / 38.2 tok/s − ttft, P5:779). No P6 profile exists on :36, so the :36 kernels are ~1-2 ms below these figures somewhere in `gemv_i8a8<4>` / `moe_planes`.

| Item (P5 per rank-step, C from P5:749-778 / 380) | ms | % | vs 1-row (P3) | What the 4 rows cost |
|---|---|---|---|---|
| `gemv_i8a8<4>` 220 × 41.3 us | 9.07 | 16 | 4.68 | **1.90×** for the same 1,478 MB (163 GB/s): per-column activation load, 4 DP4A, 3 shuffles, scale FMA repeated; 0 B spill, same grid |
| `moe_planes_v2<1>` 50.8 × 147.8 + `<5>` 51.0 × 139.2 | 14.61 | 26 | 4.96 | 29.8 distinct experts a layer instead of 10 (H11 routed/dispatch, M) = 1.81 GB → 3.0 ms at 596, 5.3 at M6's 343; v2 at P5 read all 40 (row, expert) pairs, v2m on :36 dedups ~27 % duplicates (unprofiled, ~1 ms per M26) |
| `hc_read_a` 105.7 × 69.8 | 7.38 | 13 | 4.21 | 1.60× — same 81-WG grid, 4× the columns per WG; `hc_up_mix<4>` 96.7 × 45.4 = 4.39 (1.27×) |
| `moe_gemm_bf16` 8 × 488 | 3.90 | 7 | 1.60 | 2.45× on ~52 MB a rank-step: emulated WMMA tiles |
| `ar_kernel` 105.7 × 34.4 (warm trend 29 us → 3.06) | 3.64 | 6 | 2.06 | +13.1 us a call for +15 KB payload = ~1.1-1.3 GB/s of uncached dword stores |
| GDN 35.8 × 74.6 | 2.67 | 5 | 2.08 | 1.29×; the increment bounds the per-token body at ≤4.7 us, fixed chain ≥53 us |
| `gemv_bf16<4>` 146 × 14.0 + `logits_i8<4>` 1,290 us | 3.34 | 6 | 1.83 | 1.63× / 2.21× |
| Draft passes (3 serial 1-row passes through the MTP head): `gemv_w2a8` 1.57, `rowtopk` ×3 0.47, `mtp_rows_gemv` 0.29, extra hc/ar/attention/i8<1> launches ~3.0 | 5.3-5.9 | 10 | — | ~1.9 ms a pass; the 2-bit head reads 79.5 MB at 153 GB/s. (`mtp_enter_tiled` on :33 was 19.7 ms a step, P4:749 — removed by the few-row rewrite) |
| Attention side + small kernels | ~2.6 | 5 | 1.9 | +3 attention launches a step from the head |
| **Total** | **56.6** | | **28.55** | |

Decomposition of the **+28.1 ms** over the 1-row step (C, skeptic-corrected): **48 % (13.4-13.6 ms) is 4-row forms re-doing per-column work on the same weights** (trunk 5.9, hc 3.4, bf16 experts 2.3, ar 1.2, GDN 0.6), **31 % (8.8 ms) genuinely more expert bytes**, **21 % (5.3-5.9 ms) the three draft passes**. Bytes floor of the MTP step: 4.49 GB → **7.5 ms at 596** (C); the step runs at 79 GB/s = 13 % of peak. Break-even mean acceptance at N=3: (56.6 − 28.55)/(3 × 28.55) = 0.33; measured 0.45-0.50 greedy, 0.60-0.69 on the rotation. N=2 vs N=3 is a wash on the greedy prompt, N=4 loops the trunk GEMV a second time (kMaxM = 4, xe_gemv_i8.cpp:37) and loses 5-7 tok/s (C).

## 2b. The rotation: why 30-31 tok/s and not 42

Three separable parts, two measured directly, one fitted (H11 = 5-min 11-prompt rotation, image :48, gate + 4 GiB headroom):

| Part | tok/s | Evidence |
|---|---|---|
| Service setting vs bench setting (gate + 4 GiB headroom + history pass) | 42.0 → **40.0-40.1** | M26 42.0 (:36 bench) vs C9 40.1 / M27 40.0 (service args), M |
| Residency at 72.9 % resident (18,288 / 25,078 units; 4 GiB headroom costs 880 slab slots per GiB) | 40.0 → **35.75** (taped warm median, requests 12-21) | 77.9 misses + 26.8 promotions a step (H11/stats-final, M); taped warm step 78-88 ms vs 58.7 ms single prompt (C from `step decode` lines and request lines). Misses are in-kernel zero-copy PCIe reads served by only that (token, expert) pair's work-groups inside a ~50 us expert GEMV → exposed, not hidden: fitted **150-220 us a miss** vs 87.5 us of pure transfer (H5→H11 ladder, C). Promotion/demotion copies run on the mover's own in-order queue, polled, never waited on by compute (mover.cpp:1483-1500, 413-440); 4.5 ms/step of blitter at 6.5 % duty — overlapped, not a floor term. Link floor at H11's counts: 6.8-9.1 ms a rank-step; fitted cost 11-17 ms; 4-12 ms of the +27 ms gap unattributed. |
| **Tape pool exhaustion → all passes live, host-bound** | 35.75 → **17.0** (requests 22-37); overall median **30.5** | Pool grows 1,034 → 8,000 alive lists during requests 12-15 (re-recordings), floor warning at ~98 s inside request 16 (H11/serve.log:1034-1035, one per rank), then one still-played LRU tape dropped per tape-less pass (ctx.cpp:1385-1391) → pool empty by request 22 (`l0 tape` lines :1065/:1136/:1200; no play line after :1200), `passes.issued_refused` 7,918 of 16,352; `ranks` phase 35-162 ms a step (live issue ~1,400 launches on 2 CPUs), `collect` unchanged at 77-94. Replayed exactly in H11t. H10 (tapes off): median 21.6, 148.9 ms mean step. |

So the service's rotation median loses ~2 to the setting, ~4-5 to misses and ~5 to the tape floor; on any run longer than ~2 minutes the floor part dominates and the median converges to 17-22. Decay/age-bar changes are worth ≤1 tok/s (H12 min-age 16: no change; 91 % of readmits return after ≥16 steps — the working set coming back one or two requests later).

---

## 3. Prefill: what bounds the first token

Current service figure: **27.33 s at 7K (5,604 tokens), 48.73 s at 20K** (B12, image :57 with the expert matrix form, results-m3/card01/B12-out.txt, M). The profiled B10 run (36.88 s, GEMV expert form, :56) is the only per-kernel table; B11 (bf16 GEMM on DPAS) took 3.3-3.8 s off it and B12 (expert matrix form) a further 5.28 s.

| B10 kernel (per rank = both-rank ms / 2; B10-serve.log:874-888, M) | s a rank | Status now | FLOP floor @183 TFLOPS f16 | Achieved |
|---|---|---|---|---|
| GDN chunk scan, 108 launches × 69.1 ms | **7.47** | unchanged (27 % of 27.33) | 0.797 TFLOP → **4.4 ms** | 107 GFLOP/s = 0.06 % |
| Expert GEMVs v2m (4-row runs) 10.46 | ~5.2 residual | matrix form landed (−5.28 s) | 13.2 TFLOP → 72 ms | 0.7 % (GEMV form) |
| `hc_read_a` 388 × 12.95 ms + `hc_up` 1.48 | **6.51** | unchanged | 7.1 TFLOP → 39 ms | 0.6 % |
| `gemv_bf16` 204,546 × 19.2 us | 3.93 | removed by B11 | ~3 ms | — |
| `attn_split` 288 × 8.18 ms | **2.36** | unchanged | 1.385 TFLOP → 7.6 ms | 0.3 % |
| `ar_kernel` 388 × 1.59 ms, `gemm_i8a8_dpas` 528 × 0.87, rest | 1.6 | | int8 trunk 16.6 TFLOP → 45 ms (94 ms at the measured 177 TOPS) | DPAS 36 TOPS |
| Wall − device (host, pass boundaries, profiling 0.5-1 s, rank imbalance at 230 vs 275 W caps) | 4.3 | | | |

Chunk structure is 4,096 / 1,472 / 36 tokens + a 1-token pass (scheduler.cpp:757-806; chunk-scan launches at 150.8 / 54.2 / 2.4 ms = 64 / 23 / 1 sub-chunks × 2.356 ms, C) — STATE.md's "2,048/2,048/1,508" is wrong.

**What bounds it:** every large kernel is a decode-shaped per-row form executed at M rows — the chunk scan is **3,168 serial 64-token steps a rank (36 layers × 88) at 2.356 ms each on 48 work-groups** (18.75 % occupancy), each tile product going through a software WMMA emulator (xe_lanes.h:605-640: 72 shuffles + 128 scalar FMAs per 16×16×16 tile) with 16.8 KB/thread spill (K1-out.txt:217); hc_read_a re-reads and re-decodes the 3.28 MB E4M3 W_down once per token row; attention is the decode kernel (one WG per query row, scalar fp32, E4M3 decoded per element by each of 12 sub-groups). Not bandwidth (two 28 GB weight sweeps = 95 ms), not launch count (B11 removed 96 % of launches for 12 % of time), not the link while staging engages.

**Floors.** Compute: **~0.17-0.22 s a rank** (C: int8 trunk 45-94 ms + experts and hc on the f16 pipe 72 + 39 ms + attention 7.6 + GDN 4.4 + bf16 3), the ranks concurrent. PCIe: **~1.0 s a rank** (13.38 GiB h2d measured, B10-serve.log:848, at 14.5 GB/s) — hidden today by the 3-layer-ahead staging ring and binding below ~9 ms a layer; B13 shows what happens when it is lost: +4.86 s (the 8,192-token budget moved the staging threshold while `--checkpoint-interval` kept chunks at 4,096). Realistic end state **~8 s** is a bottom-up stack of three unwritten kernel ports (chunk scan, hc at M rows, blocked attention) plus a profiled/tuned expert matrix form; calibration: Strata ~1,000 tok/s on one B70 = 5.6 s (IQ3_S, int8 prefill experts, 8,192 chunks; OWNER-NOTES note 50), production 2× R9700 0.5 s.

---

## 4. Ranked bottlenecks across all measures

| # | Bottleneck | Measure | Now | Achievable | Lever | Cost | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Tape pool exhausts VRAM → drop cascade → all passes live, host-bound (median 17.0 vs 35.75 taped) | rotation | 30.5 tok/s median; steps 149-254 ms | 35.75 median (+19 on affected requests) | Bound the pool ≤1 GB/card (compose a pass as a few large lists; per-list driver scratch ~3.8 MB × 1,182) **and** change the under-floor policy: remember the refused shape, stop dropping played LRU tapes (ctx.cpp:1385-1391) | Tape-layer change in l0.cpp; parked after runs 42-45 | **High** on effect (measured, replayed in H11t); medium on fix |
| 2 | GDN chunk scan at prefill (software WMMA emulator, 3,168 serial steps, spill) | prefill | 7.47 s | 0.4-0.5 s (predicted; floor 4.4 ms) | joint_matrix rewrite with parallel sub-chunks | Days + 2-3 card runs | Medium-high size / medium effort |
| 3 | hc E4M3 mixes in decode (14-16 % of bandwidth) | decode single (+MTP) | 8.01 ms | 3.2 ms | Clean GEMV forms (norm once, 8 waves, SLM decode table), fuse A+B, hc_write into ar | 2-3 kernel runs | Medium (probe rate measured, no engine bench) |
| 4 | Expert GEMV at 4 rows (26 % of the MTP step) | MTP | 14.6 ms | ~7 ms | P6 profile on :36 first; split-K / wider tiles for the down class | Kernel-structure change | Low-medium |
| 5 | 4-column trunk forms (1.90× / 1.63× / 2.21× for the same bytes) | MTP | 12.4 ms (tax 5.9) | 9-10 ms | Multi-column XMX GEMV (note 32) or hoist the per-group reduction; the existing DPAS kernel's measured rate gives 1.3-3.6 ms, not the 3.9 ms claimed | 1 kernel + oracle | Medium-low |
| 6 | Rotation residency (72.9 % resident, misses exposed at 150-220 us each) | rotation | 11-17 ms of an 85.6 ms step | 6 ms (headroom 2 GiB, 79.7 %) to 2-4 ms (87.5 %, capacity ceiling) | Headroom 4→2 GiB once the pool is bounded (H5: +1.75 tok/s measured); prefetch missed units so the read hides behind VRAM work | Flag / mover change | Medium |
| 7 | Expert GEMV at 1 row (23 % of peak) | decode single | 4.51 ms | 2.5 ms | Occupancy A/B, more rows or K per WG for down, scatter-side row→expert | 1 sweep | Medium-low |
| 8 | hc_read_a + hc_up at M rows (per-row re-decode) | prefill | 6.5 s | 0.5 s | Matrix-unit / batched-row form (run 60 item 1) | 1 kernel | Medium-high size |
| 9 | Expert matrix-form residual (unprofiled) + chunk size | prefill | ~5.2 s | 2.5 s | Profile MX form; raise checkpoint interval with the budget; keep staging | 1-2 runs | Low-medium |
| 10 | Trunk 1-column GEMV below bandwidth + small-launch floors | decode single | 6.74 ms | 4.85 ms | Hoist the reduction (probe design), fuse k/v into qg, batch bf16 GEMVs, fold router top-k | 2-3 runs | Medium |
| 11 | Protected-expert bf16 WMMA pair (bimodal, spilling) | MTP / single | 3.9 / 0.8-2.1 ms | 0.6 / 0.1 ms | Plain bf16 GEMV (m2 kernel 5) | 1 kernel | Medium |
| 12 | Attention at prefill (decode kernel per query row) | prefill | 2.36 s | 0.3 s | Blocked DPAS form over QSA-selected blocks | 1 kernel | Medium |
| 13 | GDN update chain + attention side-chain in decode | decode single | 3.44 ms | 2.0 ms | Merge reductions, context-aware splits, fuse rope/QSA/gates | 2-3 small kernels | Low-medium |
| 14 | Exchange protocol latency (17-34 us a call for 5-20 KB) | MTP / single | 3.6 / 1.7 ms | 2.0 / 1.2 ms | Not the persistent kernel (0.06-0.29 ms); leaner protocol unmeasured; MTP vector stores ~1 ms | 1-2 runs, stall risk | Low |
| 15 | Host/sync/list-boundary gap + tiny tails | decode single | 1.0-2.7 + 0.6 ms | ~0.6 ms | Fewer lists (same change as #1), tail fusion | with #1 | Low-medium |
| 16 | Multi-user MTP host-bound on 2 CPUs (35 s CPU / 19 s at 3 users ungated) | multi-user | gated 59.5/54.4 at 3/6 users | gate 2-3 users | Keep the gate; 6-8 CPUs after the heatsink | Hardware | Medium |

### Not fixable on this hardware (and why) vs fixable

**Not fixable:**
- **No FP8 math on Xe2.** Every E4M3 code (hc mixes 680 MB a rank-step, expert scales) is decoded by ALU; the best E4M3 GEMV measured on this card is 232-242 GB/s (41 % of peak) and the best byte-decoded u4 GEMV 343 GB/s (M6). The hc cluster's floor is therefore ~3.2 ms, not 1.14; the expert floor ~1.8 ms, not 1.0. Production's R9700 does these in native FP8 at 545 GB/s achieved.
- **Int8 matrix peak is 177 TOPS measured vs 367 datasheet** (round1/STATE.md:59; round2/PROMPT.md:14); the prefill compute floor doubles accordingly (94 ms vs 45 ms for the trunk). Still ~100× under today's kernels, so irrelevant until the kernels are rewritten.
- **PCIe Gen4 ×8 at 14.5 GB/s and 32 GB per card.** 31.8 GB of expert units a rank cannot all be resident (ceiling ~87.5 % at 512 MiB headroom); a miss costs ≥87.5 us of transfer and today 150-220 us exposed. Both root ports are ×8 max (H11/serve.log:549,552), so ×16 means different slots.
- **The two-rank exchange's ~15-19 us handshake** (M2b 24 us persistent, warm one-shot 17.3): 97 exchanges a step ≥ ~1.5 ms whatever the kernel; one-reduce-a-layer is forbidden by the HyperConnection's data dependence (rad_block_hc.h:8-16,76-78).
- **2 CPUs**: live pass issue costs 60-155 ms a step; only tapes (or more cores) hide it.
- Consequence: the step's bandwidth floor (5.9 ms → 165 tok/s) and production's 218 tok/s (4.6 ms step) are out of reach. The realistic per-class ceiling is **~15-17 ms = 59-66 tok/s MTP off, ~47-50 tok/s MTP on at 0.46 acceptance** (brief's 44-50), ~8 s for the 7K first token.

**Fixable (kernel form, measured gap to a measured rate on this card):** hc at 14-16 % vs a 40 % probe (#3); experts at 23 % vs 58 % (#4, #7); the 4-column tax (#5); the WMMA-emulated kernels (#2, #11, #12); the tape pool's VRAM and policy (#1, #6); launch structure (#10, #13, #15).

---

## 5. The single most important finding

Across every measure the port is **kernel-form-bound, not bandwidth-, compute- or host-bound**: the MTP-off step moves 3.3 GB in 26.9 ms (21 % of the 596 GB/s the card delivers to a clean GEMV), and the two largest items — the E4M3 HyperConnection mixes at 8.0 ms / 14-16 % of peak and the expert GEMVs at 4.5 ms / 23 % — have measured, same-shape, same-card probe rates (242 and 343 GB/s) that put ~7 ms of the 27 ms within reach of kernel rewrites that are not yet written; the MTP step adds 13.4 ms of 4-row forms that re-do per-column work on unchanged bytes; the prefill's largest item is one HIP kernel executing 3,168 serial 64-token steps on a software WMMA emulator at 0.06 % of the matrix units. But for the service as it actually runs (MTP on, varied traffic), the biggest **measured** loss is not a kernel: the tape pool's +4.5 GB/card drives the card under the 1 GiB floor at ~98 s, the one-LRU-tape-per-pass drop rule empties the pool by request 22, and every pass then runs live on two CPUs at 149-254 ms a step — median 17.0 vs 35.75 taped, replayed identically in H11 and H11t. Bounding the pool and fixing the under-floor policy is the one item whose worth is measured rather than predicted (+5.25 on the 5-minute median, +19 on affected requests, and no collapse on longer runs), and it is the precondition for any kernel gain to be visible in service; everything else in the ranked list — hc (4.4-4.9 ms), expert GEMV at 4 rows (~5-8 ms), the 4-column trunk form (1.3-3.6 ms), the chunk scan (~7 s) — is a prediction from a probe rate or a FLOP count, not a measurement, and should be read as such until a P6 (MTP on, image :36+) and an MX-form prefill profile exist.